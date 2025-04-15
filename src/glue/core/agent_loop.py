"""
Agent Loop module for the GLUE framework.

This module provides the core agent loop implementation that drives the execution
cycle of agents within the GLUE framework, including observation, reasoning,
planning, and action phases with proper state management.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Set, TYPE_CHECKING
import asyncio
import logging
import time
import uuid
import traceback
from datetime import datetime
import json
import re
from ..utils.json_utils import extract_json
from dataclasses import asdict, dataclass, field

from .types import AdhesiveType, MessageType, ToolResult, V1MessagePayload, Message
from .model import Model
# Conditionally import Team and Agent for type checking
if TYPE_CHECKING:
    from .teams import Team, Agent

# Set up logging
logger = logging.getLogger("glue.agent_loop")

# Define the Action structure used internally within the loop
@dataclass
class Action:
    type: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    adhesive: Optional[str] = None # Store adhesive as string value (e.g., 'glue')
    reason: Optional[str] = None
    content: Optional[str] = None # Used for final_response type

class AgentState(str, Enum):
    """Possible states of an agent in the execution loop"""
    IDLE = "idle"
    OBSERVING = "observing"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"
    HOLD = "hold"


class AgentLoop:
    """Core agent loop implementation with state management"""
    
    def __init__(self, agent_id: str, team_id: str, model: Model, team_ref: 'Team', agent_ref: Optional['Agent'] = None):
        """Initialize the agent loop
        
        Args:
            agent_id: Unique identifier for this agent/loop
            team_id: ID of the team this agent belongs to
            model: Model instance to use for reasoning/acting
            team_ref: Reference to the parent Team instance
            agent_ref: Reference to the specific Agent instance (if this loop is for an agent)
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.model = model
        self.team_ref = team_ref
        self.agent_ref = agent_ref
        self.state = AgentState.IDLE
        self.memory = {
            "observations": [],
            "thoughts": [],
            "plans": [],
            "actions": [],
            "results": []
        }
        self.waiting_for = None
        self.last_error = None
        self.tools = {}
        self.logger = logging.getLogger(f"glue.agent.{agent_id}")
        self.max_iterations = 50  # Safety limit
        self.iteration_count = 0
        self.start_time = None
        self.last_activity = None
        self.observers = []  # Callbacks for state changes
        self._main_task = None  # Main loop task
        self.current_task_id: Optional[str] = None
        self.current_task_adhesive: AdhesiveType = AdhesiveType.TAPE # Default
        self._current_state_context = None  # Added for _set_state context storage
        
    def register_tool(self, tool_name: str, tool_func: Callable) -> None:
        """Register a tool for the agent to use
        
        Args:
            tool_name: Name of the tool
            tool_func: Tool implementation function
        """
        self.tools[tool_name] = tool_func
        self.logger.info(f"Registered tool: {tool_name}")
        
    def register_observer(self, callback: Callable[[str, AgentState, Any], None]) -> None:
        """Register an observer for state changes
        
        Args:
            callback: Function to call on state change (agent_id, new_state, context)
        """
        self.observers.append(callback)
        
    def _set_state(self, new_state: AgentState, context: Any = None) -> None:
        """Set the agent state and notify observers
        
        Args:
            new_state: New agent state
            context: Optional context about the state change
        """
        old_state = self.state
        if old_state == new_state:
             # self.logger.debug(f"State already {new_state.value}, not changing.")
             return # Avoid redundant logging/notifications if state isn't changing
             
        self.state = new_state
        self.last_activity = time.time()
        self._current_state_context = context # Store context
        
        # Log state change
        self.logger.info(f"State change: {old_state.value} -> {new_state.value}")
        if context:
             self.logger.debug(f"State change context: {context}")
        
        # Notify observers
        for observer in self.observers:
            try:
                observer(self.agent_id, new_state, context)
            except Exception as e:
                self.logger.error(f"Error in observer: {str(e)}")
                
    async def start(self, initial_input: Optional[str] = None, task_data: Optional[Dict[str, Any]] = None) -> None:
        """Start the agent loop
        
        Args:
            initial_input: Optional initial input string (used if no task_data)
            task_data: Optional dictionary containing task details (goal, adhesive, id, etc.)
        """
        self.start_time = time.time()
        self.iteration_count = 0
        self._set_state(AgentState.IDLE)
        
        processed_input = None
        if task_data:
            self.current_task_id = task_data.get('task_id')
            # Ensure adhesive is enum, default to TAPE
            task_adhesive = task_data.get('adhesive', AdhesiveType.TAPE)
            if isinstance(task_adhesive, AdhesiveType):
                 self.current_task_adhesive = task_adhesive
            else:
                 try:
                      self.current_task_adhesive = AdhesiveType(str(task_adhesive).lower())
                 except ValueError:
                      self.logger.warning(f"Invalid adhesive '{task_adhesive}' in task data, defaulting to TAPE.")
                      self.current_task_adhesive = AdhesiveType.TAPE
            
            processed_input = task_data.get('goal')
            self.logger.info(f"Starting loop for task {self.current_task_id} with goal: '{processed_input}' and adhesive {self.current_task_adhesive.value}")
        elif initial_input:
            processed_input = initial_input
            self.logger.info(f"Starting loop with initial input: '{processed_input}'")
            # Default task ID and adhesive if started without explicit task
            self.current_task_id = f"adhoc_{str(uuid.uuid4())[:8]}"
            self.current_task_adhesive = AdhesiveType.TAPE 
        else:
            self.logger.info("Starting loop with no initial input or task.")
            self.current_task_id = f"idle_{str(uuid.uuid4())[:8]}"
            self.current_task_adhesive = AdhesiveType.TAPE

        try:
            # Add the processed input/goal as the first observation
            if processed_input:
                self.memory["observations"].append({
                    "id": str(uuid.uuid4()),
                    "type": "initial_input" if not task_data else "task_goal",
                    "content": processed_input,
                    "task_id": self.current_task_id, # Add task_id to observation
                    "timestamp": datetime.now().isoformat()
                })
                
            # Create the main loop task BEFORE trying to log its name or await it
            self._main_task = asyncio.create_task(self._run_loop(), name=f"agent_loop_{self.agent_id}_{self.current_task_id}")

            try:
                # Await the main loop task
                self.logger.debug(f"[{self.agent_id}] START: Awaiting main loop task {self._main_task.get_name()} for task {self.current_task_id}")
                await self._main_task
                # This log might not be reached if _main_task is cancelled externally
                self.logger.debug(f"[{self.agent_id}] END: Main loop task completed normally for task {self.current_task_id}. Final loop state: {self.state}")
                
                # --- Task Completion Feedback ---
                # Determine final status based on state when loop finishes naturally
                final_status = "unknown"
                final_result = None
                final_error = self.last_error
                termination_context = self._current_state_context if self.state == AgentState.TERMINATED else None
                
                if self.state == AgentState.TERMINATED:
                    reason = termination_context.get("reason", "unknown") if isinstance(termination_context, dict) else "unknown"
                    if reason == "max_iterations_reached":
                         final_status = "max_iterations_error"
                    elif reason in ["cancelled", "requested", "cleanup", "new_task_assigned_externally"]:
                         final_status = "cancelled" # Treat external new task assignment like cancellation
                    else: 
                         final_status = "completed" 
                elif self.state == AgentState.ERROR:
                     final_status = "error"
                else: # Should ideally not happen if loop awaited correctly
                     final_status = "aborted"
                     
                await self._send_feedback_to_lead(status=final_status, error=final_error, result=final_result)
                
            except asyncio.CancelledError:
                self.logger.info(f"[{self.agent_id}] Agent loop task explicitly CANCELLED (asyncio.CancelledError caught) for task {self.current_task_id}")
                self._set_state(AgentState.TERMINATED, {"reason": "cancelled"})
                await self._send_feedback_to_lead(status="cancelled")
            except Exception as e:
                # Log exceptions that occur *during* the await self._main_task itself
                # or during the feedback sending after normal completion
                self.logger.error(f"[{self.agent_id}] Exception occurred in start() after awaiting main task or during feedback for task {self.current_task_id}: {e}", exc_info=True)
                self.last_error = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
                self._set_state(AgentState.ERROR, self.last_error)
                await self._send_feedback_to_lead(status="error", error=self.last_error)
            finally:
                # Add logging to see if cleanup is called and why
                completion_status = "Unknown"
                exception_info = "None"
                task_result = "Not available"
                if self._main_task:
                     if self._main_task.cancelled():
                         completion_status = "Cancelled"
                     elif self._main_task.done():
                         exc = self._main_task.exception()
                         if exc:
                             completion_status = f"Error"
                             # Log the full traceback of the underlying exception
                             exception_info = traceback.format_exception(type(exc), exc, exc.__traceback__)
                             exception_info = "".join(exception_info) # Join traceback list into string
                             # Try to get the result even if there was an error
                             try:
                                task_result = self._main_task.result()
                             except Exception as res_exc:
                                task_result = f"Result retrieval failed: {res_exc}"
                         else:
                             completion_status = "Completed Normally"
                             # Get the result if completed normally
                             try:
                                task_result = self._main_task.result()
                             except Exception as res_exc:
                                task_result = f"Result retrieval failed: {res_exc}"
                     else:
                         completion_status = "Still Running(?)"

                self.logger.debug(f"[{self.agent_id}] FINALLY: Entering finally block in start() for task {self.current_task_id}. Loop task status: {completion_status}. Current agent state: {self.state}. Exception info: {exception_info}. Task result: {task_result}")
                await self.cleanup()

        except Exception as e:
            # This outer except block catches errors in the start() method itself, 
            # *before* or *after* awaiting _main_task, or during the finally block.
            self.logger.error(f"Error in start(): {e}", exc_info=True)
            self.last_error = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            self._set_state(AgentState.ERROR, self.last_error)
            await self._send_feedback_to_lead(status="error", error=self.last_error)

    def _get_termination_context(self) -> Optional[Dict[str, Any]]:
         """ Helper to safely get termination context (implementation depends on _set_state details) """
         if self.state == AgentState.TERMINATED and isinstance(self._current_state_context, dict):
             return self._current_state_context
         return None

    async def cleanup(self) -> None:
        """Clean up resources used by the agent loop"""
        # Cancel the main task if it's still running
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                # Add a timeout to prevent hanging
                await asyncio.wait_for(asyncio.shield(self._main_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                # This is expected, we're cancelling the task
                pass
            
        # Set state to terminated if not already
        if self.state != AgentState.TERMINATED:
            self._set_state(AgentState.TERMINATED, {"reason": "cleanup"})
            
    async def _run_loop(self) -> None:
        """Run the main agent loop"""
        while self.state != AgentState.TERMINATED:
            # Safety check for max iterations
            self.iteration_count += 1
            if self.iteration_count > self.max_iterations:
                self.logger.warning(f"Reached maximum iterations ({self.max_iterations}), terminating")
                self._set_state(AgentState.TERMINATED, {"reason": "max_iterations_reached"})
                break
                
            # Execute the appropriate phase based on current state
            if self.state == AgentState.IDLE:
                await self._observe_phase()
            elif self.state == AgentState.OBSERVING:
                await self._reason_phase()
            elif self.state == AgentState.REASONING:
                await self._plan_phase()
            elif self.state == AgentState.PLANNING:
                # After planning, we should ACT on the plan
                await self._act_phase() 
            elif self.state == AgentState.ACTING:
                # After acting, we should OBSERVE the results
                await self._observe_phase()  # Back to observation
            elif self.state == AgentState.WAITING:
                # Check if what we're waiting for is ready
                if await self._check_waiting_condition():
                    # Resume from waiting state
                    self._set_state(AgentState.OBSERVING)
                else:
                    # Still waiting, sleep a bit
                    await asyncio.sleep(0.5)
            elif self.state == AgentState.HOLD:
                # Stay in HOLD until state is changed externally (e.g., by TeamLead)
                self.logger.debug(f"Agent {self.agent_id} is in HOLD state.")
                await asyncio.sleep(1.0)  # Sleep longer while holding
            elif self.state == AgentState.ERROR:
                # Try to recover from error
                if await self._attempt_recovery():
                    self._set_state(AgentState.OBSERVING)
                else:
                    self._set_state(AgentState.TERMINATED, {"reason": "unrecoverable_error"})
                    
            # Small delay to prevent tight loops if not holding/waiting
            if self.state not in [AgentState.HOLD, AgentState.WAITING]:
                await asyncio.sleep(0.1)
            
    async def _observe_phase(self) -> None:
        """Observation phase: gather inputs and context"""
        self._set_state(AgentState.OBSERVING)
        
        # Gather new observations (including those derived from previous results)
        new_observations = await self._gather_observations()
        
        # Add new observations to memory
        self.memory["observations"].extend(new_observations)
        
        # --- ADDED: Clear results memory after processing --- 
        if self.memory["results"]:
             self.logger.debug(f"Clearing {len(self.memory['results'])} results from memory after observation.")
             self.memory["results"].clear()
        
        # Move to reasoning phase
        self._set_state(AgentState.REASONING, {"new_observations": len(new_observations)})
        
    async def _reason_phase(self) -> None:
        """Reasoning phase: analyze observations and context"""
        self._set_state(AgentState.REASONING)
        
        # Generate thoughts based on observations and history
        thoughts = await self._generate_thoughts()
        
        # Add to memory
        self.memory["thoughts"].extend(thoughts)
        
        # Move to planning phase
        self._set_state(AgentState.PLANNING, {"new_thoughts": len(thoughts)})
        
    async def _plan_phase(self) -> None:
        """Planning phase: decide on actions to take"""
        self._set_state(AgentState.PLANNING)
        
        # Generate plan based on thoughts
        try:
            self.logger.debug(f"[{self.agent_id}] PLAN: Generating plan for task {self.current_task_id}...")
            plan = await self._generate_plan()
            self.logger.debug(f"[{self.agent_id}] PLAN: Plan generated successfully for task {self.current_task_id}. Plan details: {plan}") # Log the plan details
        except Exception as e:
            self.logger.error(f"[{self.agent_id}] PLAN: Error during plan generation for task {self.current_task_id}: {e}", exc_info=True)
            self.last_error = {"error": str(e), "traceback": traceback.format_exc(), "phase": "planning"}
            self._set_state(AgentState.ERROR, self.last_error)
            return # Exit phase on error

        # Add to memory
        self.memory["plans"].append({
            "id": str(uuid.uuid4()),
            "content": plan,
            "task_id": self.current_task_id,
            "timestamp": datetime.now().isoformat()
        })

        # Move to acting phase
        self._set_state(AgentState.ACTING, {"plan": plan})
        
    async def _act_phase(self) -> None:
        """Action phase: execute planned actions with adhesive context"""
        self._set_state(AgentState.ACTING)
        
        # Get the latest plan
        current_plan = self.memory["plans"][-1] if self.memory["plans"] else None

        # Get actions from the plan content (fix: the plan's "content" field contains the actual plan object from _generate_plan)
        if current_plan and "content" in current_plan:
            actions_to_execute = current_plan["content"].get("actions", [])
        else:
            actions_to_execute = None

        # --- DETAILED DEBUGGING ADDED ---
        self.logger.debug(f"[{self.agent_id}] ACT_PHASE_CHECK: current_plan from memory: {current_plan}")
        # Use .get() with a default empty list for safety before checking type/len
        safe_actions = actions_to_execute if actions_to_execute is not None else [] 
        self.logger.debug(f"[{self.agent_id}] ACT_PHASE_CHECK: actions_to_execute type={type(safe_actions)}, len={len(safe_actions) if isinstance(safe_actions, list) else 'N/A'}, content={safe_actions}")
        # Re-assign actions_to_execute using the safe access
        actions_to_execute = safe_actions 
        # --- END DETAILED DEBUGGING ---

        if not actions_to_execute:
            # Log if plan existed but had no actions, or if plan was missing
            log_reason = "'actions' key missing or empty in plan" if current_plan else "no current plan found in memory"
            self.logger.debug(f"No actions to execute: {log_reason}. Returning to observe.")
            self._set_state(AgentState.OBSERVING)
            return # Returns here if no actions

        # >>> ADDED DEBUG: Log the actions list just before the loop <<<
        self.logger.debug(f"[{self.agent_id}] ACT_PHASE: About to loop over actions_to_execute (type: {type(actions_to_execute)}): {actions_to_execute}")

        # Execute actions in the plan
        results = []
        # --- CORRECTED LOOP: Iterate over actions_to_execute --- 
        for action in actions_to_execute:
            self.logger.debug(f"[{self.agent_id}] ACT_PHASE: Processing action: {action}") # Log action being processed
            # Assume action dict now contains 'adhesive': AdhesiveType
            action_result = await self._execute_action(action) 
            self.logger.debug(f"[{self.agent_id}] ACT_PHASE: Action result received: {action_result}") # Log received result
            results.append(action_result)
            # --- MOVED --- Add result to memory immediately
            self.memory["results"].append(action_result) # Append single result
            # --- ADDED Log list ID after append ---\n            self.logger.debug(f"[{self.agent_id}] ACT_PHASE: Appended to memory['results']. Object ID: {id(self.memory['results'])}, Current results memory: {self.memory['results']}")

            # Check if we need to wait after this action
            if action_result.get("requires_waiting"):
                self.waiting_for = action_result.get("waiting_for")
                self._set_state(AgentState.WAITING, {"waiting_for": self.waiting_for})
                # --- MODIFIED: Add partial results to memory before returning ---
                # Note: We are now appending results within the loop, so extending here is redundant/incorrect.
                # self.memory[\"results\"].extend(results) # Remove this redundant extend
                self.logger.debug(f"[{self.agent_id}] ACT_PHASE: Entering WAITING state for {self.waiting_for}. Results added so far: {len(self.memory['results'])}")
                return # Exit _act_phase to handle waiting state
            
            # Check for errors in action execution
            if not action_result.get("success", False):
                 self.logger.error(f"Action failed: {action_result.get('error')}. Proceeding with next action if any.")
                 # Decide if we should stop or continue on action error? For now, continue.

        # Move back to observation phase
        self._set_state(AgentState.OBSERVING, {"action_results": len(results)})
        
    async def _gather_observations(self) -> List[Dict[str, Any]]:
        """Gather new observations from environment, team messages, and previous action results."""
        await asyncio.sleep(0.01) # Keep the sleep just in case
        # --- ADDED Log list ID before access ---
        self.logger.debug(f"[{self.agent_id}] GATHER_OBS START: Checking memory['results']. Object ID: {id(self.memory['results'])}")
        
        observations = []
        now_iso = datetime.now().isoformat()

        # --- 1. Check results from the last action phase ---
        self.logger.debug(f"Gathering observations. Current self.memory[\"results\"]: {self.memory['results']}")
        last_results = self.memory["results"]
        self.logger.debug(f"Gathering observations. Processing {len(last_results)} results from memory.")

        for result_item in reversed(last_results): # Check recent results first
            self.logger.debug(f"Gathering observations. Processing result item: {result_item}")

            # Check if the result is from a successful 'communicate' action
            # The result itself might now contain the V1 payload if the tool returns it
            # Or, if the tool just returns success/failure, the actual message might arrive via handle_message later.
            # Let's assume for now the communicate tool might return the received payload directly if synchronous,
            # or just success/failure if asynchronous (queued).
            # We also need to process messages received via handle_message.

            # --- Process Tool Results (including potential direct responses) ---
            if isinstance(result_item, dict) and result_item.get("success"):
                tool_name = result_item.get("tool")
                raw_result_content = result_item.get("result") # This might be the V1 payload dict or simple text

                # If it's a communicate result that potentially contains a direct V1 payload response:
                if tool_name == "communicate" and isinstance(raw_result_content, dict) and raw_result_content.get('message_type') == MessageType.DIRECT_MESSAGE.value:
                     # It seems the communicate tool might have directly returned the response payload
                     # Let's parse it as a V1 payload observation
                     v1_payload = raw_result_content
                     self.logger.info(f"Adding observation from direct 'communicate' V1 payload result.")
                     observations.append({
                         "id": str(uuid.uuid4()),
                         "type": "direct_message_response", # Specific type
                         "tool_name": tool_name,
                         "sender_agent_id": v1_payload.get('sender_agent_id'),
                         "sender_team_id": v1_payload.get('sender_team_id'),
                         "adhesive_type": v1_payload.get('adhesive_type'),
                         "content": v1_payload.get('content'),
                         "timestamp": v1_payload.get('timestamp', now_iso) # Use payload timestamp if available
                     })
                elif tool_name: # Handle results from other tools or simple communicate success messages
                     self.logger.info(f"Adding observation for successful tool result: {tool_name}")
                     observations.append({
                         "id": str(uuid.uuid4()),
                         "type": "tool_success",
                         "tool_name": tool_name,
                         "content": str(raw_result_content), # Convert result to string for observation
                         "adhesive_used": result_item.get("adhesive_used"),
                         "timestamp": result_item.get("timestamp", now_iso)
                     })
            elif isinstance(result_item, dict) and not result_item.get("success"):
                 # Log failed actions as observations
                 tool_name = result_item.get("tool", "unknown_action")
                 error_content = result_item.get("error", "Unknown error")
                 self.logger.info(f"Adding observation for failed action/tool: {tool_name}")
                 observations.append({
                     "id": str(uuid.uuid4()),
                     "type": "action_failure",
                     "tool_name": tool_name,
                     "error": error_content,
                     "timestamp": result_item.get("timestamp", now_iso)
                 })

        # --- 2. Check for new messages received via handle_message (from team queue) ---
        # We need a way to get these messages. Add a queue to AgentLoop?
        # For now, assume _get_team_messages fetches from a hypothetical internal queue populated by handle_message
        team_messages = await self._get_team_messages() # This needs implementation
        for msg in team_messages:
            observations.append({
                "id": str(uuid.uuid4()),
                "type": "team_message",
                "source": msg.get("source"),
                "content": msg.get("content"),
                "timestamp": now_iso
            })
            
        # --- 3. Check for environmental changes --- 
        env_changes = await self._get_environment_changes()
        for change in env_changes:
            observations.append({
                "id": str(uuid.uuid4()),
                "type": "environment_change",
                "entity": change.get("entity"),
                "change": change.get("change"),
                "timestamp": now_iso
            })
        
        # --- 4. NEW: Check agent's velcro memory if available ---
        if self.agent_ref and hasattr(self.agent_ref, 'velcro_memory') and self.agent_ref.velcro_memory:
            self.logger.info(f"Found {len(self.agent_ref.velcro_memory)} items in agent's velcro memory")
            
            # Process each item in velcro memory
            velcro_items = list(self.agent_ref.velcro_memory.items())
            for result_id, tool_result in velcro_items:
                if not isinstance(tool_result, ToolResult):
                    self.logger.warning(f"Skipping invalid velcro memory item (not ToolResult): {result_id}")
                    continue
                    
                # Create an observation from the velcro memory item
                observations.append({
                    "id": str(uuid.uuid4()),
                    "type": "velcro_memory",
                    "tool_name": tool_result.tool_name,
                    "content": str(tool_result.result),  # Convert result to string
                    "result_id": result_id,  # Include the original ID for reference
                    "timestamp": tool_result.timestamp.isoformat(),
                    "adhesive_type": "velcro"
                })
                
                # By default, once we've observed a velcro memory item, remove it 
                # (unless metadata explicitly says to preserve it)
                preserve = False
                if hasattr(tool_result, 'metadata') and isinstance(tool_result.metadata, dict):
                    preserve = tool_result.metadata.get('preserve_velcro', False)
                    
                if not preserve:
                    # Remove item from velcro memory after it's been observed
                    self.logger.debug(f"Removing used velcro memory item: {result_id}")
                    self.agent_ref.velcro_memory.pop(result_id, None)
                else:
                    self.logger.debug(f"Preserving velcro memory item for future use: {result_id}")
            
        # Reverse the collected observations so newest are last when added to memory
        observations.reverse()
        
        # --- ADDED: Clear results memory AFTER processing them --- 
        if self.memory["results"]:
             self.logger.debug(f"Clearing {len(self.memory['results'])} results from memory after gathering observations.")
             self.memory["results"].clear()
             
        return observations
        
    async def _get_team_messages(self) -> List[Dict[str, Any]]:
        """Get new messages from the team
        
        Returns:
            List of message objects
        """
        # This would connect to the team communication system
        # For this example, we'll return an empty list
        return []
        
    async def _get_environment_changes(self) -> List[Dict[str, Any]]:
        """Get changes in the environment
        
        Returns:
            List of change objects
        """
        # This would connect to the environment monitoring system
        # For this example, we'll return an empty list
        return []
        
    async def _generate_thoughts(self) -> List[Dict[str, Any]]:
        """Generate thoughts based on observations
        
        Returns:
            List of thought objects
        """
        # Prepare context for the model
        context = {
            "observations": self.memory["observations"][-10:],  # Last 10 observations
            "thoughts": self.memory["thoughts"][-5:],  # Last 5 thoughts
            "results": self.memory["results"][-5:],  # Last 5 results
        }
        
        # Create a prompt for the model
        prompt = f"""
        You are an agent with ID {self.agent_id} in team {self.team_id}.
        
        Recent observations:
        {self._format_observations(context['observations'])}
        
        Previous thoughts:
        {self._format_thoughts(context['thoughts'])}
        
        Recent action results:
        {self._format_results(context['results'])}
        
        Based on this information, generate new thoughts about what's happening and what to do next.
        **Consider any recent action failures or errors in your thoughts. If a communication attempt failed, think about why and potential alternative approaches.**
        """
        
        # Call the model
        try:
            response = await self.model.generate(prompt)
            
            # Parse the response into thoughts
            thought_text = response.strip()
            
            thought = {
                "id": str(uuid.uuid4()),
                "content": thought_text,
                "timestamp": datetime.now().isoformat()
            }
            
            return [thought]
            
        except Exception as e:
            self.logger.error(f"Error generating thoughts: {str(e)}")
            
            # Return a fallback thought
            return [{
                "id": str(uuid.uuid4()),
                "content": "I encountered an error while thinking. I should try a different approach.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }]
    
    def _format_observations(self, observations: List[Dict[str, Any]]) -> str:
        """Format observations for inclusion in prompts"""
        if not observations:
            return "No recent observations."
            
        formatted = []
        for obs in observations:
            if obs.get("type") == "team_message":
                formatted.append(f"- Message from {obs.get('source')}: {obs.get('content')}")
            elif obs.get("type") == "environment_change":
                formatted.append(f"- Environment change: {obs.get('entity')} {obs.get('change')}")
            elif obs.get("type") == "initial_input":
                formatted.append(f"- Initial input: {obs.get('content')}")
            else:
                formatted.append(f"- {obs.get('type', 'Observation')}: {obs.get('content', str(obs))}")
                
        return "\n".join(formatted)
        
    def _format_thoughts(self, thoughts: List[Dict[str, Any]]) -> str:
        """Format thoughts for inclusion in prompts"""
        if not thoughts:
            return "No previous thoughts."
            
        formatted = []
        for thought in thoughts:
            formatted.append(f"- {thought.get('content')}")
                
        return "\n".join(formatted)
        
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results for inclusion in prompts"""
        if not results:
            return "No recent action results."
            
        formatted = []
        for result in results:
            if result.get("success", False):
                formatted.append(f"- Action with {result.get('tool', 'unknown tool')} succeeded: {result.get('result', '')}")
            else:
                formatted.append(f"- Action with {result.get('tool', 'unknown tool')} failed: {result.get('error', 'Unknown error')}")
                
        return "\n".join(formatted)
        
    async def _generate_plan(self) -> Dict[str, Any]:
        """Generate a plan based on thoughts
        
        Returns:
            Plan object
        """
        self.logger.debug(f"[{self.agent_id}] Entered _generate_plan for task {self.current_task_id}")

        # Prepare context for the model
        context = {
            "observations": self.memory["observations"][-10:],
            "thoughts": self.memory["thoughts"][-5:],
            "results": self.memory["results"][-5:],
        }
        tool_descriptions = []
        for tool_name in self.tools:
            tool_func = self.tools[tool_name]
            doc = tool_func.__doc__ or f"Tool: {tool_name}"
            tool_descriptions.append(f"- {tool_name}: {doc}")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."

        # Create a prompt for the model
        prompt = f"""
        You are an agent with ID {self.agent_id} in team {self.team_id}.
        Your current task is to: {self.memory['observations'][0]['content'] if self.memory['observations'] else 'Unknown'}
        
        Recent observations (including tool responses):
        {self._format_observations(context['observations'])}
        
        Your thoughts:
        {self._format_thoughts(context['thoughts'])}
        
        Available tools:
        {tools_text}
        
        Based on this information, create a plan with specific actions using the available tools to achieve the task goal.
        
        **IMPORTANT:**
        - **Consider Failures:** If recent observations include action failures (especially communication failures), adapt your plan. Consider retrying the action, trying a different target, using a different tool (like web_search if communication fails), or determining if the required information can be obtained another way.
        - **Final Answer Check:** If your observations indicate you have successfully gathered all necessary information and are ready to provide the final answer, respond with ONLY the final text answer. Do not add explanations. Just provide the answer.
        
        Otherwise, if more actions are needed:
        For each action, specify:
        1. The tool to use.
        2. The parameters (JSON format).
        3. The adhesive ('glue', 'velcro', or 'tape'). **MUST be specified.**
        4. The reason for the action (and how it addresses previous failures, if applicable).
        
        Format your response as a list of actions OR as the final direct text answer.
        """
        
        # Call the model
        try:
            # Get the raw response from the model provider
            provider_response = await self.model.generate_response(
                messages=[Message(role="user", content=prompt)], 
                tools=self.tools # Pass tools for potential native handling
            )
            self.logger.debug(f"[{self.agent_id}] PLAN: LLM call completed. Raw Response Type: {type(provider_response)}, Length: {len(str(provider_response))}")

            parsed_actions = []
            action_count = 0

            # Handle potential native tool calls first (if provider returns them)
            if isinstance(provider_response, dict) and "tool_calls" in provider_response:
                self.logger.info(f"[{self.agent_id}] PLAN: Received native tool calls from provider.")
                # Process native tool calls (assuming a format like OpenAI's)
                for tool_call in provider_response["tool_calls"]:
                    # Example processing - adjust based on actual provider format
                    if "function" in tool_call:
                        tool_name = tool_call["function"].get("name")
                        try:
                            arguments = json.loads(tool_call["function"].get("arguments", "{}"))
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not decode arguments for native tool call {tool_name}: {tool_call['function'].get('arguments')}")
                            arguments = {}
                        
                        # Assume 'glue' adhesive for native calls unless specified differently
                        adhesive = AdhesiveType.GLUE

                        if tool_name and tool_name in self.tools:
                             # Replace Action(...) with dict literal
                             parsed_action = {
                                "type": "tool_use",
                                "tool": tool_name,
                                "parameters": arguments,
                                "adhesive": adhesive.value,
                                "reason": f"Execute native tool call {tool_name}",
                                "id": str(uuid.uuid4()) # Add unique ID here
                            }
                             parsed_actions.append(parsed_action)
                             action_count += 1
                             self.logger.info(f"[{self.agent_id}] PLAN: Successfully parsed native tool action: Tool={tool_name}, Adhesive={adhesive.value}, Params={arguments}")
                        else:
                             self.logger.warning(f"[{self.agent_id}] PLAN: Unknown tool name in native tool call: {tool_name}")
            
            # Handle string responses (potentially containing JSON for simulated tool calls)
            elif isinstance(provider_response, str):
                response_text = provider_response.strip()
                self.logger.debug(f"[{self.agent_id}] PLAN: Received string response. Attempting JSON extraction.")
                self.logger.debug(f"[{self.agent_id}] PLAN: Raw string response from LLM:\\n{provider_response}") # Log raw response

                # Attempt to extract JSON using the utility function
                json_action = extract_json(response_text)
                self.logger.debug(f"[{self.agent_id}] PLAN: Result from extract_json: {json_action}") # Log extracted result

                if json_action and isinstance(json_action, dict):
                    # Successfully extracted JSON, process it as a tool call
                    self.logger.info(f"[{self.agent_id}] PLAN: Successfully extracted JSON object.")

                    # --- Explicit Validation ---
                    if "tool_name" not in json_action:
                        self.logger.warning(f"[{self.agent_id}] PLAN: Extracted JSON is missing the required 'tool_name' key. Ignoring JSON. JSON content: {json_action}")
                        # Continue to the next part of the 'elif' to check for final answer
                        json_action = None # Nullify to prevent further processing as action
                    else:
                        tool_name = json_action.get("tool_name")
                        arguments = json_action.get("arguments", {}) # Default to empty dict if missing
                        reason = json_action.get("reason", "Reason not provided by model.") # Optional reason
                        adhesive_str = json_action.get("adhesive", "glue") # Default to glue

                        # Validate tool name
                        if not tool_name or tool_name not in self.tools:
                             self.logger.warning(f"[{self.agent_id}] PLAN: Extracted JSON has invalid or unknown 'tool_name': {tool_name}. Ignoring JSON. JSON content: {json_action}")
                             json_action = None # Nullify to prevent further processing as action
                        else:
                            # --- Adhesive Parsing ---
                            try:
                                adhesive = AdhesiveType(adhesive_str.lower())
                                if adhesive not in self.model.adhesives:
                                     self.logger.warning(f"[{self.agent_id}] PLAN: Model does not support adhesive '{adhesive_str}' specified in JSON. Defaulting to 'glue'.")
                                     adhesive = AdhesiveType.GLUE
                            except ValueError:
                                 self.logger.warning(f"[{self.agent_id}] PLAN: Invalid adhesive '{adhesive_str}' specified in JSON. Defaulting to 'glue'.")
                                 adhesive = AdhesiveType.GLUE
                            # --- End Adhesive Parsing ---

                            # Validate arguments format
                            if not isinstance(arguments, dict):
                                self.logger.warning(f"[{self.agent_id}] PLAN: Invalid 'arguments' format in JSON action. Expected dict, got {type(arguments)}. Ignoring JSON. JSON content: {json_action}")
                                json_action = None # Nullify to prevent further processing as action
                            else:
                                # If all validation passes, construct the action dictionary
                                parsed_action = {
                                    "type": "tool_use",
                                    "tool": tool_name,
                                    "parameters": arguments,
                                    "adhesive": adhesive.value, # Store the string value
                                    "reason": reason,
                                    "id": str(uuid.uuid4()) # Add unique ID here
                                }
                                parsed_actions.append(parsed_action)
                                action_count += 1
                                self.logger.info(f"[{self.agent_id}] PLAN: Successfully parsed validated JSON action: Tool={tool_name}, Adhesive={adhesive.value}, Params={arguments}")
                    # --- End Explicit Validation ---

                # If no JSON was extracted OR if extracted JSON failed validation
                if not json_action: # Check if json_action is None (either never extracted or nullified above)
                    # Check if the raw text looks like an attempt at a tool call (contains tool keywords)
                    # This helps differentiate between failed JSON and a direct answer.
                    likely_tool_attempt = any(tool_name in response_text for tool_name in self.tools)

                    if likely_tool_attempt:
                         self.logger.warning(f"[{self.agent_id}] PLAN: Failed to extract valid JSON action, but response text mentions tool names. Assuming failed tool call format.")
                         # Optionally: Could add a "failed_action" observation here?
                    else:
                        # Assume it's a final response if no JSON found/validated and no tool keywords detected
                        self.logger.info(f"[{self.agent_id}] PLAN: No valid JSON action found/validated, assuming direct final response.")
                        # Replace Action(...) with dict literal
                        parsed_action = {
                            "type": "final_response",
                            "content": response_text,
                            "reason": "Agent assuming direct final response after failing to parse valid JSON action.",
                            "id": str(uuid.uuid4()) # Add unique ID here
                        }
                        parsed_actions.append(parsed_action)
                        action_count = 1 # Treat final response as a single action step

            # Handle other unexpected response types from the provider
            else:
                 self.logger.warning(f"[{self.agent_id}] PLAN: Unexpected response type from model provider: {type(provider_response)}. Treating as final response.")
                 # Replace Action(...) with dict literal
                 parsed_action = {
                    "type": "final_response",
                    "content": str(provider_response),
                    "reason": "Unexpected provider response type.",
                    "id": str(uuid.uuid4()) # Add unique ID here
                 }
                 parsed_actions.append(parsed_action)
                 action_count = 1
                 
            self.logger.debug(f"[{self.agent_id}] PLAN: Finished processing. Found {action_count} actions.")
            plan = {
                "id": str(uuid.uuid4()),
                "actions": parsed_actions, # Use the newly parsed actions
                "timestamp": datetime.now().isoformat()
            }
            self.logger.debug(f"[{self.agent_id}] PLAN: Returning parsed plan for task {self.current_task_id}: {plan}")
            return plan

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] PLAN: Error generating or parsing plan: {e}", exc_info=True)
            fallback_plan = {
                "id": str(uuid.uuid4()),
                "actions": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.debug(f"[{self.agent_id}] PLAN: Returning fallback plan for task {self.current_task_id}")
            return fallback_plan
        
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action
        
        Args:
            action: Action object to execute
            
        Returns:
            Result object
        """
        action_type = action.get("type")
        action_id = action.get("id")
        now_iso = datetime.now().isoformat()
        
        if action_type == "tool_use":
            return await self._execute_tool_action(action)
            
        # --- ADDED: Handle final_response action type --- 
        elif action_type == "final_response":
            final_content = action.get("content")
            self.logger.info(f"[{self.agent_id}] EXECUTE_ACTION: Handling final_response action. Content: {final_content}")
            
            # Determine adhesive (Task default should be GLUE for final output)
            final_adhesive = self.current_task_adhesive
            if final_adhesive != AdhesiveType.GLUE:
                 self.logger.warning(f"Final response generated, but task adhesive is {final_adhesive.value}, not GLUE. Final result might not be persisted correctly.")
                 # Should we override to GLUE here? For now, respect task adhesive.
            
            # Create a result object similar to ToolResult for consistency?
            final_result_obj = ToolResult(
                tool_name="final_summary", # Use a placeholder name
                result=final_content,
                error=None,
                adhesive=final_adhesive,
                timestamp=datetime.fromisoformat(now_iso), # Use consistent timestamp
                metadata={"agent_id": self.agent_id, "action_id": action_id, "task_id": self.current_task_id}
            )
            
            # Share/Persist using the appropriate adhesive (should be GLUE)
            agent_name = self.agent_ref.name if self.agent_ref else self.team_ref.lead.name
            await self.team_ref.share_result("final_summary", final_result_obj, agent_name=agent_name)
            
            # Terminate the loop successfully
            self.terminate(reason="final_response_generated")
            
            # Return a success result for this action
            return {
                "id": str(uuid.uuid4()), 
                "action_id": action_id,
                "tool": "final_summary", # Match ToolResult
                "success": True,
                "result": final_content, 
                "error": None,
                "adhesive_used": final_adhesive.value,
                "timestamp": now_iso
            }
        # --- END Handle final_response --- 
            
        elif action_type == "message":
            # Keep existing message action handling (if any)
            return await self._execute_message_action(action)
        else:
            # Keep existing unknown action handling
            return {
                "id": str(uuid.uuid4()),
                "action_id": action_id,
                "success": False,
                "error": f"Unknown action type: {action_type}",
                "timestamp": now_iso
            }
            
    async def _execute_tool_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool action and handle result based on adhesive."""
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})
        # Decide which adhesive to use: Action specific > Task Default > TAPE
        action_adhesive = action.get("adhesive")
        final_adhesive_type = AdhesiveType.TAPE # Overall default
        
        if action_adhesive: # 1. Check action itself
             if isinstance(action_adhesive, AdhesiveType):
                 final_adhesive_type = action_adhesive
             else:
                 try:
                     final_adhesive_type = AdhesiveType(str(action_adhesive).lower())
                 except ValueError:
                     logger.warning(f"Invalid adhesive '{action_adhesive}' in action, using task default.")
                     final_adhesive_type = self.current_task_adhesive # Fallback to task adhesive
        else: # 2. Use task default if no action adhesive
             final_adhesive_type = self.current_task_adhesive

        # Check if tool exists in this loop's registered tools
        if tool_name not in self.tools:
            error_msg = f"Tool not found: {tool_name}"
            logger.error(error_msg)
            return {
                "id": str(uuid.uuid4()),
                "action_id": action.get("id"),
                "tool": tool_name,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            
        tool_result_obj: Optional[ToolResult] = None
        execution_success = False
        raw_result = None
        error_detail = None

        try:
            # Execute the tool
            tool_func = self.tools[tool_name]
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Found tool_func: {tool_func}, params: {parameters}") # Log before await
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Awaiting tool \'{tool_name}\'") # Log before await
            self.logger.debug(f"[{self.agent_id}] PRE_EXECUTE_TOOL_ACTION: Calling tool '{tool_name}' with parameters: {parameters}")
            raw_result = await tool_func(**parameters)
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Tool \'{tool_name}\' await completed.") # Log after await
            execution_success = True
            
        except asyncio.CancelledError as ce:
            # Explicitly catch CancelledError
            error_detail = f"Tool execution cancelled: {ce}"
            logger.warning(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Tool '{tool_name}' execution was cancelled.")
            execution_success = False
        except Exception as e:
            # Catch standard exceptions
            error_detail = str(e)
            logger.error(f"Error executing tool '{tool_name}' with params {parameters}: {e}", exc_info=True)
            execution_success = False
        except BaseException as be:
            # Catch other potential BaseExceptions (like SystemExit, KeyboardInterrupt if not handled higher)
            error_detail = f"Tool execution failed with BaseException: {type(be).__name__}: {be}"
            logger.error(f"BaseException during tool '{tool_name}' execution: {be}", exc_info=True)
            execution_success = False
            # Re-raising might be appropriate depending on the exception, but for now, treat as failure

        # --- Add outer try/except for result processing --- 
        # This block should now always be reached, even if the tool call failed/was cancelled
        try: 
            # Create ToolResult object regardless of success (to potentially log failure)
            tool_result_obj = ToolResult(
                tool_name=tool_name,
                result=raw_result if execution_success else None,
                error=error_detail if not execution_success else None,
                adhesive=final_adhesive_type,
                timestamp=datetime.now(),
                metadata={"agent_id": self.agent_id, "action_id": action.get("id")}
            )
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Created ToolResult object: success={execution_success}, error={error_detail}")
    
            # --- Handle result based on adhesive ---
            agent_name = self.agent_ref.name if self.agent_ref else self.team_ref.lead.name 
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Determined agent_name: {agent_name}, adhesive: {final_adhesive_type.value}")
    
            if final_adhesive_type == AdhesiveType.GLUE:
                if execution_success:
                    # Call team method to handle persistence/sharing
                    await self.team_ref.share_result(tool_name, tool_result_obj, agent_name=agent_name)
                    logger.info(f"GLUE adhesive: Initiated sharing/persistence for result from {tool_name} by {agent_name}.")
                else:
                     logger.warning(f"GLUE adhesive specified, but tool {tool_name} failed. Result not shared/persisted.")
    
            elif final_adhesive_type == AdhesiveType.VELCRO:
                if execution_success and self.agent_ref:
                    # Store result in the specific agent's velcro memory
                    result_id = str(uuid.uuid4()) # Use a unique ID for storage key
                    self.agent_ref.velcro_memory[result_id] = tool_result_obj
                    logger.info(f"VELCRO adhesive: Stored result {result_id} from {tool_name} locally for agent {agent_name}.")
                elif not self.agent_ref:
                     logger.warning(f"VELCRO adhesive specified, but cannot store locally for TeamLead ({agent_name}).")
                elif not execution_success:
                     logger.warning(f"VELCRO adhesive specified, but tool {tool_name} failed. Result not stored locally for agent {agent_name}.")
    
            elif final_adhesive_type == AdhesiveType.TAPE:
                # Transient - do nothing further with the result here
                logger.info(f"TAPE adhesive: Result from {tool_name} used transiently by {agent_name}.")
                
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Finished adhesive handling.")
                
            # Return a standardized result dictionary for the loop's memory
            result_dict_for_memory = {
                "id": str(uuid.uuid4()), 
                "action_id": action.get("id"),
                "tool": tool_name,
                "success": execution_success,
                "result": raw_result if execution_success else None, 
                "error": error_detail,
                "adhesive_used": final_adhesive_type.value,
                "timestamp": tool_result_obj.timestamp.isoformat() 
            }
            self.logger.debug(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Returning result for memory: {result_dict_for_memory}")
            return result_dict_for_memory
            
        except Exception as e_res_proc:
            # Catch errors specifically during result object creation or adhesive handling
            self.logger.error(f"[{self.agent_id}] EXECUTE_TOOL_ACTION: Error processing result for tool '{tool_name}': {e_res_proc}", exc_info=True)
            # Return a failure dictionary
            return {
                "id": str(uuid.uuid4()), 
                "action_id": action.get("id"),
                "tool": tool_name,
                "success": False,
                "result": None, 
                "error": f"Error processing tool result: {e_res_proc}",
                "adhesive_used": final_adhesive_type.value,
                "timestamp": datetime.now().isoformat() 
            }
        # --- End outer try/except ---
            
    async def _execute_message_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a message action
        
        Args:
            action: Message action to execute
            
        Returns:
            Result object
        """
        target = action.get("target")
        content = action.get("content")
        
        # This would send a message to the target
        # For this example, we'll just create a result
        return {
            "id": str(uuid.uuid4()),
            "action_id": action.get("id"),
            "target": target,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "requires_waiting": action.get("requires_response", False),
            "waiting_for": f"response_from_{target}" if action.get("requires_response", False) else None
        }
            
    async def _check_waiting_condition(self) -> bool:
        """Check if waiting condition is satisfied
        
        Returns:
            True if condition is satisfied, False otherwise
        """
        if not self.waiting_for:
            return True
            
        # This would check the specific waiting condition
        # For this example, we'll just return True to continue
        return True
        
    async def _attempt_recovery(self) -> bool:
        """Attempt to recover from an error
        
        Returns:
            True if recovery successful, False otherwise
        """
        # This would implement recovery strategies
        # For this example, we'll just return True
        return True
        
    def terminate(self, reason: str = "requested") -> None:
        """Terminate the agent loop
        
        Args:
            reason: Reason for termination
        """
        self._set_state(AgentState.TERMINATED, {"reason": reason})
        self.logger.info(f"Agent terminated: {reason}")
        
        # Cancel the main task if it exists and is running
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent
        
        Returns:
            Status object
        """
        return {
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "state": self.state,
            "iteration_count": self.iteration_count,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "last_activity": self.last_activity,
            "memory_size": {k: len(v) for k, v in self.memory.items()},
            "waiting_for": self.waiting_for,
            "error": self.last_error
        }

    async def _send_feedback_to_lead(self, status: str, result: Any = None, error: Optional[Dict[str, Any]] = None) -> None:
        """Send status feedback about the current task to the TeamLead."""
        if not self.current_task_id:
             logger.debug("No current task ID, skipping feedback.")
             return
             
        if not self.team_ref or not self.team_ref.lead:
             logger.warning(f"Cannot send feedback for task {self.current_task_id}: Team or TeamLead not found.")
             return

        # Construct the content part of the V1 payload
        feedback_content = {
            "status": status,
            "result": result, # Keep result/error separate within content for clarity
            "error": error
        }

        # Construct the full V1 payload
        try:
            # Import necessary types if not already at top level
            from .types import V1MessagePayload, MessageType, AdhesiveType # Keep imports

            payload = V1MessagePayload(
                task_id=self.current_task_id,
                sender_agent_id=self.agent_id,
                sender_team_id=self.team_id,
                timestamp=datetime.now().isoformat(),
                message_type=MessageType.AGENT_FEEDBACK,
                adhesive_type=AdhesiveType.TAPE, # Feedback is transient
                content=feedback_content,
                origin_tool_id=None # Feedback doesn't originate from a tool
            )
            # Use the new to_dict method for serialization
            payload_dict = payload.to_dict()

        except Exception as e:
             logger.error(f"Failed to construct V1 payload for feedback: {e}", exc_info=True)
             return # Cannot send feedback if payload construction fails

        lead_loop_id = f"{self.team_ref.name}-lead-{self.team_ref.lead.name}"
        
        # How does the agent send a message *specifically* to the lead's loop?
        # Option 1: Use Team message queue with routing info
        # Option 2: Direct access if loops know each other (complex)
        # Option 3: Coordinator handles routing
        
        # Using Option 1 (Team Queue) for now:
        try:
            # Package the V1 payload dict for the team queue
            routed_message = {
                 "source": "internal", # Indicate internal origin
                 "target_loop_id": lead_loop_id,
                 "content": payload_dict # The V1 payload dictionary
            }
            await self.team_ref.message_queue.put(routed_message)
            logger.info(f"Sent V1 feedback payload for task {self.current_task_id} (Status: {status}) to lead via team queue.")
        except Exception as e:
            logger.error(f"Failed to send feedback for task {self.current_task_id} to lead: {e}")

    # Method to allow external state changes (e.g., by TeamLead/Coordinator)
    def set_loop_state(self, new_state: AgentState, context: Optional[Dict] = None) -> None:
        """Externally set the agent loop's state."""
        # Potentially add checks here: e.g., don't allow setting to TERMINATED externally?
        if new_state in AgentState:
             self.logger.info(f"State explicitly set externally to {new_state.value} by external caller.")
             self._set_state(new_state, context)
        else:
             logger.warning(f"Attempted to set invalid state externally: {new_state}")
             
    # Method to handle messages routed specifically to this loop
    async def handle_message(self, message_content: Dict[str, Any]) -> None:
        """Process a message directed specifically to this agent loop."""
        # Extract and validate message_type
        raw_message_type = message_content.get("message_type")
        message_type = None

        # Handle both string values and enum values
        if isinstance(raw_message_type, str):
            try:
                # Ensure it's a valid MessageType value before assigning
                message_type = MessageType(raw_message_type).value 
            except ValueError:
                 self.logger.warning(f"Invalid message_type string value: {raw_message_type}")
                 return
        elif isinstance(raw_message_type, MessageType):
            message_type = raw_message_type.value
        else:
            self.logger.warning(f"Invalid message_type format: {raw_message_type}")
            return

        self.logger.debug(f"AgentLoop {self.agent_id} received message: type={message_type}")
        
        if message_type == MessageType.PAUSE_QUERY.value:
            # When agent receives a pause query, go into HOLD state
            query_content = message_content.get("content", {}).get("query_content", "No query specified")
            self.logger.info(f"Setting state to HOLD based on pause query: {query_content}")
            self._set_state(AgentState.HOLD, {
                "reason": "pause_query_received", 
                "query": query_content
            })
            
        elif message_type == MessageType.RESUME_TASK.value:
            # Exit HOLD state and update task goal if provided
            content = message_content.get("content", {})
            updated_goal = content.get("updated_goal")
            original_goal = content.get("original_goal")
            
            self.logger.info(f"Received RESUME_TASK message, exiting HOLD state.")
            
            # If we have an updated goal, create a new observation with it
            if updated_goal:
                self.logger.info(f"Task goal updated: {updated_goal}")
                
                # Add the updated goal as a new observation with special type
                self.memory["observations"].append({
                    "id": str(uuid.uuid4()),
                    "type": "task_goal_updated",
                    "content": updated_goal,
                    "previous_goal": original_goal,
                    "task_id": self.current_task_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Optionally clear some memory to ensure fresh approach with new goal
                # (This depends on implementation needs)
                # self.memory["thoughts"].clear()
                # self.memory["plans"].clear()
                
            # Move back to OBSERVING state to restart cycle with new/same goal
            # Only transition if currently in HOLD state
            if self.state == AgentState.HOLD:
                 self._set_state(AgentState.OBSERVING, {"reason": "task_resumed_after_refinement"})
            else:
                 # If not in HOLD, just log that we received the message but didn't need to resume
                 self.logger.debug(f"Received RESUME_TASK but was not in HOLD state. Current state: {self.state}")
                 
        elif message_type == "set_hold": # Legacy or direct hold command
            hold_flag = message_content.get("hold", True)
            if hold_flag:
                 self.logger.info(f"Setting state to HOLD based on external message.")
                 self.set_loop_state(AgentState.HOLD, {"reason": "external_request"})
            else:
                 self.logger.info(f"Exiting HOLD state based on external message. Returning to IDLE.")
                 self.set_loop_state(AgentState.IDLE, {"reason": "external_request"})
                 
        elif message_type == "new_task":
            # Used to assign a new task after hold or for initial assignment?
            task_data = message_content.get("task_data")
            if task_data:
                 self.logger.info(f"Received new task data externally: {task_data.get('task_id')}. Restarting loop.")
                 self.terminate(reason="new_task_assigned_externally")
            else:
                 logger.warning("Received new_task message without task_data.")

        elif message_type == MessageType.DIRECT_MESSAGE.value:
            # --- Enhanced Direct Message Handling --- 
            sender_agent_id = message_content.get("sender_agent_id", "unknown")
            sender_team_id = message_content.get("sender_team_id", "unknown")
            received_content = message_content.get("content")
            task_id = message_content.get("task_id")
            timestamp = message_content.get("timestamp", datetime.now().isoformat())
            
            self.logger.info(f"Received direct message from {sender_agent_id} (team {sender_team_id})")

            # Check if it's a simple query suitable for immediate response
            # (Simple heuristic: check if it's a string and ends with a question mark)
            is_simple_query = isinstance(received_content, str) and received_content.strip().endswith("?")

            if is_simple_query and self.state != AgentState.HOLD:
                 self.logger.info(f"Identified simple query from {sender_agent_id}. Attempting direct response.")
                 try:
                     # Directly prompt the model to answer the question
                     response_prompt = f"You received the following question from agent '{sender_agent_id}' in team '{sender_team_id}'. Please provide a concise answer.\n\nQuestion: {received_content}"
                     direct_response_content = await self.model.generate(response_prompt)
                     self.logger.info(f"Generated direct response: {direct_response_content[:100]}...")

                     # Construct the response payload (V1MessagePayload)
                     response_payload = V1MessagePayload(
                         task_id=task_id, # Use the same task_id if relevant
                         sender_agent_id=self.agent_id,
                         sender_team_id=self.team_id,
                         timestamp=datetime.now().isoformat(),
                         message_type=MessageType.DIRECT_MESSAGE,
                         adhesive_type=AdhesiveType.TAPE, # Response is transient
                         content=direct_response_content,
                         origin_tool_id=None
                     )
                     # Use the new to_dict method for serialization
                     payload_dict = response_payload.to_dict()

                     # Determine target loop ID for the sender
                     # Need to reconstruct sender's loop ID format
                     # Assuming format is "{team_id}-{lead/agent}-{agent_id}"
                     # This might need refinement based on exact ID structure
                     sender_loop_id = None
                     # Check if sender is lead or agent within their team (requires app access or better context)
                     # Simple guess based on ID structure (less robust):
                     if "-lead-" in sender_agent_id:
                          sender_loop_id = sender_agent_id
                     elif "-agent-" in sender_agent_id:
                          sender_loop_id = sender_agent_id
                     else: # Fallback/Guess - needs improvement
                          # Check if sender is the lead of *their* team
                          # This requires accessing team info, which AgentLoop might not have directly
                          # For now, assume agent format if not clearly lead
                          sender_loop_id = f"{sender_team_id}-agent-{sender_agent_id}" 
                          self.logger.warning(f"Could not definitively determine sender '{sender_agent_id}' type (lead/agent). Assuming agent loop ID: {sender_loop_id}")
                         
                     # Package for team queue
                     routed_message = {
                         "target_loop_id": sender_loop_id,
                         "content": payload_dict,
                         "source": "internal" # Indicate internal origin for routing
                     }

                     # Send response via team queue
                     await self.team_ref.message_queue.put(routed_message)
                     self.logger.info(f"Sent direct response back to loop {sender_loop_id}")

                     # Add the original query AND the response to observations for context
                     self.memory["observations"].append({
                         "id": str(uuid.uuid4()),
                         "type": "direct_message_received",
                         "sender_agent_id": sender_agent_id,
                         "sender_team_id": sender_team_id,
                         "content": received_content,
                         "task_id": task_id,
                         "timestamp": timestamp
                     })
                     self.memory["observations"].append({
                         "id": str(uuid.uuid4()),
                         "type": "direct_message_sent",
                         "recipient_agent_id": sender_agent_id,
                         "recipient_team_id": sender_team_id,
                         "content": direct_response_content,
                         "task_id": task_id,
                         "timestamp": datetime.now().isoformat()
                     })
                     # Remain in current state, no need to trigger full cycle just for this
                     # self._set_state(AgentState.OBSERVING, {"reason": "sent_direct_response"})

                 except Exception as e:
                     self.logger.error(f"Error generating or sending direct response: {e}", exc_info=True)
                     # Fallback: Add message as normal observation and proceed with cycle
                     self._add_direct_message_observation(message_content)
                     if self.state != AgentState.HOLD:
                         self._set_state(AgentState.OBSERVING, {"reason": "received_direct_message_failed_direct_response"})
            else:
                 # Not a simple query or agent is on hold, add as observation and continue standard loop
                 self._add_direct_message_observation(message_content)
                 if self.state != AgentState.HOLD:
                     self._set_state(AgentState.OBSERVING, {"reason": "received_direct_message"})

        elif message_type == "received_pause_query": # Should not happen for agent loop
             if self.agent_ref:
                  logger.warning(f"Agent {self.agent_id} received pause query message - should be handled by lead.")
             pass

        else:
            logger.warning(f"AgentLoop {self.agent_id} received unhandled message type: {message_type}")

    # Helper method to add direct message observation to avoid code duplication
    def _add_direct_message_observation(self, message_content: Dict[str, Any]):
        sender_agent_id = message_content.get("sender_agent_id", "unknown")
        sender_team_id = message_content.get("sender_team_id", "unknown")
        content = message_content.get("content")
        task_id = message_content.get("task_id")
        timestamp = message_content.get("timestamp", datetime.now().isoformat())

        self.memory["observations"].append({
            "id": str(uuid.uuid4()),
            "type": "direct_message", # Generic type for standard processing
            "sender_agent_id": sender_agent_id,
            "sender_team_id": sender_team_id,
            "content": content,
            "task_id": task_id,
            "timestamp": timestamp
        })


class TeamLoopCoordinator:
    """Coordinates agent loops within a team"""
    
    def __init__(self, team_id: str):
        """Initialize team loop coordinator
        
        Args:
            team_id: Team ID
        """
        self.team_id = team_id
        self.agents = {}  # agent_id -> AgentLoop
        self.logger = logging.getLogger(f"glue.team.{team_id}")
        self.message_queue = asyncio.Queue()
        self.running = False
        self._message_task = None  # Message processing task
        
    def add_agent(self, agent_loop: AgentLoop) -> None:
        """Add an agent to the team
        
        Args:
            agent_loop: Agent loop to add
        """
        self.agents[agent_loop.agent_id] = agent_loop
        
        # Register as observer for agent state changes
        agent_loop.register_observer(self._agent_state_changed)
        
        self.logger.info(f"Added agent {agent_loop.agent_id} to team {self.team_id}")
        
    async def start(self) -> None:
        """Start all agent loops in the team"""
        self.running = True
        
        # Start message processing
        self._message_task = asyncio.create_task(self._process_messages())
        
        # Start all agents
        start_tasks = []
        for agent_id, agent_loop in self.agents.items():
            task = asyncio.create_task(agent_loop.start())
            start_tasks.append(task)
            
        # Wait for all agents to start
        await asyncio.gather(*start_tasks)
        
    async def cleanup(self) -> None:
        """Clean up resources used by the coordinator"""
        # Cancel the message processing task
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                # Add a timeout to prevent hanging
                await asyncio.wait_for(asyncio.shield(self._message_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                # This is expected, we're cancelling the task
                pass
                
        # Clean up all agent loops
        cleanup_tasks = []
        for agent_id, agent_loop in self.agents.items():
            task = asyncio.create_task(agent_loop.cleanup())
            cleanup_tasks.append(task)
            
        # Wait for all cleanups to complete with a timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*cleanup_tasks), timeout=2.0)
            except asyncio.TimeoutError:
                # Log the timeout but continue
                self.logger.warning("Timeout waiting for agent cleanup")
            
        self.running = False
        
    async def _process_messages(self) -> None:
        """Process messages in the queue"""
        while self.running:
            try:
                message = await self.message_queue.get()
                await self._route_message(message)
                self.message_queue.task_done()
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                self.logger.info(f"Message processing cancelled for team {self.team_id}")
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)
                
    def _agent_state_changed(self, agent_id: str, new_state: AgentState, context: Any) -> None:
        """Handle agent state changes
        
        Args:
            agent_id: Agent ID
            new_state: New agent state
            context: State change context
        """
        self.logger.info(f"Agent {agent_id} state changed to {new_state}")
        
        # Handle specific state changes
        if new_state == AgentState.ERROR:
            self.logger.error(f"Agent {agent_id} entered error state: {context}")
            # This could implement team-level error handling
            
        elif new_state == AgentState.TERMINATED:
            self.logger.info(f"Agent {agent_id} terminated: {context}")
            # This could implement team-level agent replacement
            
    def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the team message queue
        
        Args:
            message: Message to send
        """
        self.message_queue.put_nowait(message)
        
    def terminate(self, reason: str = "requested") -> None:
        """Terminate all agent loops in the team
        
        Args:
            reason: Reason for termination
        """
        for agent_id, agent_loop in self.agents.items():
            agent_loop.terminate(reason)
            
        self.running = False
        
        # Cancel the message processing task
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the team
        
        Returns:
            Status object
        """
        return {
            "team_id": self.team_id,
            "agent_count": len(self.agents),
            "agents": {agent_id: agent_loop.get_status() for agent_id, agent_loop in self.agents.items()},
            "message_queue_size": self.message_queue.qsize() if hasattr(self.message_queue, "qsize") else "unknown",
            "running": self.running
        }
