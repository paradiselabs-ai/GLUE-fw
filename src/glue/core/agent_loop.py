"""
Agent Loop module for the GLUE framework.

This module provides the core agent loop implementation that drives the execution
cycle of agents within the GLUE framework, including observation, reasoning,
planning, and action phases with proper state management.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Set
import asyncio
import logging
import time
import uuid
import traceback
from datetime import datetime

from .types import AdhesiveType, Message, ToolResult
from .model import Model
from ..prompts import (
    format_reasoning_prompt, 
    format_planning_prompt, 
    format_observations,
    format_thoughts,
    format_results
)

# Set up logging
logger = logging.getLogger("glue.agent_loop")


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


class AgentLoop:
    """Core agent loop implementation with state management"""
    
    def __init__(self, agent_id: str, team_id: str, model: Model):
        """Initialize the agent loop
        
        Args:
            agent_id: Unique identifier for this agent
            team_id: ID of the team this agent belongs to
            model: Model instance to use for reasoning
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.model = model
        self.state = AgentState.IDLE
        self.memory = {
            "observations": [],
            "thoughts": [],
            "plans": [],
            "actions": [],
            "results": []
        }
        self.waiting_for = None  # What the agent is waiting for
        self.last_error = None  # Last error encountered
        self.tools = {}  # Available tools
        self.adhesives = {}  # Available adhesives
        self.logger = logging.getLogger(f"glue.agent.{agent_id}")
        self.max_iterations = 50  # Safety limit
        self.iteration_count = 0
        self.start_time = None
        self.last_activity = None
        self.observers = []  # Callbacks for state changes
        self._main_task = None  # Main loop task
        
    def register_tool(self, tool_name: str, tool_func: Callable) -> None:
        """Register a tool for the agent to use
        
        Args:
            tool_name: Name of the tool
            tool_func: Tool implementation function
        """
        self.tools[tool_name] = tool_func
        self.logger.info(f"Registered tool: {tool_name}")
        
    def register_adhesive(self, adhesive_name: str, adhesive_type: AdhesiveType) -> None:
        """Register an adhesive for the agent to use
        
        Args:
            adhesive_name: Name of the adhesive
            adhesive_type: Type of adhesive (GLUE, VELCRO, TAPE)
        """
        self.adhesives[adhesive_name] = adhesive_type
        self.logger.info(f"Registered adhesive: {adhesive_name} ({adhesive_type.name})")
        
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
        self.state = new_state
        self.last_activity = time.time()
        
        # Log state change
        self.logger.info(f"State change: {old_state} -> {new_state}")
        
        # Notify observers
        for observer in self.observers:
            try:
                observer(self.agent_id, new_state, context)
            except Exception as e:
                self.logger.error(f"Error in observer: {str(e)}")
                
    async def start(self, initial_input: Optional[str] = None) -> None:
        """Start the agent loop
        
        Args:
            initial_input: Optional initial input to the agent
        """
        self.start_time = time.time()
        self.iteration_count = 0
        self._set_state(AgentState.IDLE)
        
        try:
            # Add initial input if provided
            if initial_input:
                self.memory["observations"].append({
                    "type": "initial_input",
                    "content": initial_input,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Start the main loop as a task so we can cancel it if needed
            self._main_task = asyncio.create_task(self._run_loop())
            await self._main_task
            
        except asyncio.CancelledError:
            self.logger.info(f"Agent loop cancelled for {self.agent_id}")
            self._set_state(AgentState.TERMINATED, {"reason": "cancelled"})
        except Exception as e:
            self.last_error = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            self._set_state(AgentState.ERROR, self.last_error)
            self.logger.error(f"Agent loop error: {str(e)}", exc_info=True)
        finally:
            # Ensure we clean up properly
            await self.cleanup()
            
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
                await self._act_phase()
            elif self.state == AgentState.ACTING:
                await self._observe_phase()  # Back to observation
            elif self.state == AgentState.WAITING:
                # Check if what we're waiting for is ready
                if await self._check_waiting_condition():
                    # Resume from waiting state
                    self._set_state(AgentState.OBSERVING)
                else:
                    # Still waiting, sleep a bit
                    await asyncio.sleep(0.5)
            elif self.state == AgentState.ERROR:
                # Try to recover from error
                if await self._attempt_recovery():
                    self._set_state(AgentState.OBSERVING)
                else:
                    self._set_state(AgentState.TERMINATED, {"reason": "unrecoverable_error"})
                    
            # Small delay to prevent tight loops
            await asyncio.sleep(0.1)
            
    async def _observe_phase(self) -> None:
        """Observation phase: gather inputs and context"""
        self._set_state(AgentState.OBSERVING)
        
        # Gather new observations
        new_observations = await self._gather_observations()
        
        # Add to memory
        self.memory["observations"].extend(new_observations)
        
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
        plan = await self._generate_plan()
        
        # Add to memory
        self.memory["plans"].append(plan)
        
        # Move to action phase
        self._set_state(AgentState.ACTING, {"plan_id": plan["id"]})
        
    async def _act_phase(self) -> None:
        """Action phase: execute planned actions"""
        self._set_state(AgentState.ACTING)
        
        # Get the latest plan
        current_plan = self.memory["plans"][-1] if self.memory["plans"] else None
        
        if not current_plan:
            self._set_state(AgentState.OBSERVING)
            return
            
        # Execute actions in the plan
        results = []
        for action in current_plan["actions"]:
            action_result = await self._execute_action(action)
            results.append(action_result)
            
            # Check if we need to wait after this action
            if action_result.get("requires_waiting"):
                self.waiting_for = action_result.get("waiting_for")
                self._set_state(AgentState.WAITING, {"waiting_for": self.waiting_for})
                return
                
        # Add results to memory
        self.memory["results"].extend(results)
        
        # Move back to observation phase
        self._set_state(AgentState.OBSERVING, {"action_results": len(results)})
        
    async def _gather_observations(self) -> List[Dict[str, Any]]:
        """Gather new observations from environment and team
        
        Returns:
            List of observation objects
        """
        observations = []
        
        # Check for new messages from team
        team_messages = await self._get_team_messages()
        for msg in team_messages:
            observations.append({
                "id": str(uuid.uuid4()),
                "type": "team_message",
                "source": msg.get("source"),
                "content": msg.get("content"),
                "timestamp": datetime.now().isoformat()
            })
            
        # Check for environmental changes
        env_changes = await self._get_environment_changes()
        for change in env_changes:
            observations.append({
                "id": str(uuid.uuid4()),
                "type": "environment_change",
                "entity": change.get("entity"),
                "change": change.get("change"),
                "timestamp": datetime.now().isoformat()
            })
            
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
        
    def _format_observations(self, observations: List[Dict[str, Any]]) -> str:
        """Format observations for the model prompt.
        
        Args:
            observations: List of observation dictionaries
        
        Returns:
            Formatted observations string
        """
        if not observations:
            return "No observations recorded."
            
        formatted = []
        for i, obs in enumerate(observations[-10:]):  # Only include last 10 observations
            timestamp = obs.get("timestamp", "")
            content = obs.get("content", "No content")
            obs_type = obs.get("type", "general")
            
            formatted.append(f"{i+1}. [{timestamp}] Type: {obs_type}\n   {content}")
            
        return format_observations("\n".join(formatted))
    
    def _format_thoughts(self, thoughts: List[Dict[str, Any]]) -> str:
        """Format thoughts for the model prompt.
        
        Args:
            thoughts: List of thought dictionaries
            
        Returns:
            Formatted thoughts string
        """
        if not thoughts:
            return "No previous thoughts."
            
        formatted = []
        for i, thought in enumerate(thoughts[-5:]):  # Only include last 5 thoughts
            timestamp = thought.get("timestamp", "")
            content = thought.get("content", "No content")
            
            formatted.append(f"{i+1}. [{timestamp}]\n   {content}")
            
        return format_thoughts("\n".join(formatted))
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format action results for the model prompt.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Formatted results string
        """
        if not results:
            return "No action results yet."
            
        formatted = []
        for i, result in enumerate(results[-5:]):  # Only include last 5 results
            timestamp = result.get("timestamp", "")
            success = "Success" if result.get("success", False) else "Failure"
            content = result.get("content", "No content")
            
            formatted.append(f"{i+1}. [{timestamp}] {success}\n   {content}")
            
        return format_results("\n".join(formatted))
        
    async def _generate_thoughts(self) -> List[Dict[str, Any]]:
        """Generate thoughts based on observations and memory.
        
        This method formats observations and current memory state into a prompt,
        sends it to the model, and extracts structured thoughts.
        
        Returns:
            List of thought dictionaries
        """
        # Format the observations
        formatted_observations = self._format_observations(self.memory["observations"])
        
        # Format existing thoughts for context
        formatted_thoughts = self._format_thoughts(self.memory["thoughts"])
        
        # Get goal or context if available
        goal = "Complete the current task successfully."
        if hasattr(self, "goal") and self.goal:
            goal = self.goal
        
        # Create reasoning prompt
        reasoning_prompt = format_reasoning_prompt(
            observations=formatted_observations,
            thoughts=formatted_thoughts,
            goal=goal
        )
        
        # Generate reasoning response from model
        try:
            self.logger.debug("Generating thoughts using model...")
            response = await self.model.generate(reasoning_prompt)
            self.logger.debug(f"Generated response: {response[:100]}...")
            
            # Process the response into thoughts
            thoughts = []
            
            # Split into paragraphs
            paragraphs = response.strip().split("\n\n")
            
            # Remove any paragraph that seems to be a plan or action rather than a thought
            for i, paragraph in enumerate(paragraphs):
                if paragraph and not paragraph.lower().startswith(("i'll", "i should", "i will", "i need to")):
                    thought_id = str(uuid.uuid4())
                    thoughts.append({
                        "id": thought_id,
                        "content": paragraph,
                        "timestamp": datetime.now().isoformat(),
                        "sources": [obs.get("id") for obs in self.memory["observations"][-3:]]
                    })
            
            return thoughts
            
        except Exception as e:
            self.logger.error(f"Error generating thoughts: {str(e)}", exc_info=True)
            return [{
                "id": str(uuid.uuid4()),
                "content": f"I encountered an error while thinking: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }]
        
    async def _generate_plan(self) -> Dict[str, Any]:
        """Generate a plan based on thoughts and observations.
        
        This method formats thoughts and observations into a prompt,
        sends it to the model, and extracts a structured plan.
        
        Returns:
            Plan dictionary
        """
        # Format the thoughts
        formatted_thoughts = self._format_thoughts(self.memory["thoughts"])
        
        # Format any recent results
        formatted_results = self._format_results(self.memory["results"][-5:] if self.memory["results"] else [])
        
        # Get goal or context
        goal = "Complete the current task successfully."
        if hasattr(self, "goal") and self.goal:
            goal = self.goal
            
        # Current context
        context = f"Recent results:\n{formatted_results}"
        
        # Format available tools
        available_tools = ""
        if self.tools:
            tool_descriptions = []
            for name, tool in self.tools.items():
                description = getattr(tool, 'description', 'No description')
                tool_descriptions.append(f"- {name}: {description}")
            available_tools = "\n".join(tool_descriptions)
        
        # Create planning prompt
        planning_prompt = format_planning_prompt(
            goal=goal,
            context=context,
            formatted_thoughts=formatted_thoughts,
            available_tools=available_tools
        )
        
        # Generate plan from model
        try:
            self.logger.debug("Generating plan using model...")
            response = await self.model.generate(planning_prompt)
            self.logger.debug(f"Generated response: {response[:100]}...")
            
            # Process the response into a plan
            plan_id = str(uuid.uuid4())
            
            # Extract actions from the response
            actions = []
            
            # First, check if there's a tool call format in the response
            if "```" in response and ("json" in response.lower() or "tool" in response.lower()):
                # Try to extract JSON blocks
                import re
                import json
                
                # Find code blocks
                code_blocks = re.findall(r"```(?:json|tool(?:_call)?|)\n(.*?)```", response, re.DOTALL)
                
                # Process each block
                for block in code_blocks:
                    try:
                        # Parse the JSON
                        action_data = json.loads(block.strip())
                        
                        # Check if it's a properly formatted tool call
                        if isinstance(action_data, dict) and "name" in action_data:
                            tool_name = action_data.get("name")
                            parameters = action_data.get("parameters", {})
                            
                            # Add to actions
                            actions.append({
                                "id": str(uuid.uuid4()),
                                "type": "tool",
                                "tool": tool_name,
                                "parameters": parameters
                            })
                    except json.JSONDecodeError:
                        # Not valid JSON, skip
                        continue
            
            # If no tool calls found, check for plaintext action descriptions
            if not actions:
                # Split into paragraphs and look for action descriptions
                paragraphs = response.strip().split("\n\n")
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph.lower().startswith(("action:", "step:", "i'll ", "i will ", "let's ")):
                        # This looks like an action description
                        actions.append({
                        "id": str(uuid.uuid4()),
                                "type": "message",  # Default to message if no specific tool identified
                                "content": paragraph
                            })
            
            # If still no actions found, use the whole response as a message action
            if not actions:
                actions.append({
                "id": str(uuid.uuid4()),
                    "type": "message",
                    "content": response
                })
            
            # Create the plan
            plan = {
                "id": plan_id,
                "timestamp": datetime.now().isoformat(),
                "description": response,
                "actions": actions,
                "based_on": [thought.get("id") for thought in self.memory["thoughts"][-3:]]
            }
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}", exc_info=True)
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "description": f"Error generating plan: {str(e)}",
                "actions": [{
                    "id": str(uuid.uuid4()),
                    "type": "message",
                    "content": f"I encountered an error while planning: {str(e)}"
                }],
                "error": True
            }
        
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action
        
        Args:
            action: Action object to execute
            
        Returns:
            Result object
        """
        action_type = action.get("type")
        
        if action_type == "tool_use":
            return await self._execute_tool_action(action)
        elif action_type == "message":
            return await self._execute_message_action(action)
        else:
            return {
                "id": str(uuid.uuid4()),
                "action_id": action.get("id"),
                "success": False,
                "error": f"Unknown action type: {action_type}",
                "timestamp": datetime.now().isoformat()
            }
            
    async def _execute_tool_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool action
        
        Args:
            action: Tool action to execute
            
        Returns:
            Result object
        """
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})
        
        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "id": str(uuid.uuid4()),
                "action_id": action.get("id"),
                "success": False,
                "error": f"Tool not found: {tool_name}",
                "timestamp": datetime.now().isoformat()
            }
            
        try:
            # Execute the tool
            tool_func = self.tools[tool_name]
            result = await tool_func(**parameters)
            
            # Apply adhesive if specified
            adhesive_name = action.get("adhesive")
            if adhesive_name and adhesive_name in self.adhesives:
                adhesive_type = self.adhesives[adhesive_name]
                # Create a proper ToolResult object
                tool_result = ToolResult(
                    tool_name=tool_name,
                    result=result,
                    adhesive=adhesive_type,
                    timestamp=datetime.now(),
                    metadata={"agent_id": self.agent_id, "action_id": action.get("id")}
                )
                
                # Here we would apply the adhesive persistence logic
                # For now, we just return the result
                
                return {
                    "id": str(uuid.uuid4()),
                    "action_id": action.get("id"),
                    "tool": tool_name,
                    "success": True,
                    "result": result,
                    "adhesive": adhesive_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # No adhesive specified, just return the result
                return {
                    "id": str(uuid.uuid4()),
                    "action_id": action.get("id"),
                    "tool": tool_name,
                    "success": True,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            return {
                "id": str(uuid.uuid4()),
                "action_id": action.get("id"),
                "tool": tool_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
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
