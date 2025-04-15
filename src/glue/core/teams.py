# glue/core/team.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List, Union, TYPE_CHECKING
from datetime import datetime
import asyncio
import logging
from pydantic import BaseModel, Field
import uuid
import json
from dataclasses import dataclass, field, asdict
import os

from .types import AdhesiveType, TeamConfig, ToolResult, FlowType, Message, V1MessagePayload, MessageType # Added V1MessagePayload, MessageType
from .schemas import Message, ToolCall
from ..utils.json_utils import extract_json
from ..persistence.knowledge_base import KnowledgeBase

# Import Flow class conditionally to avoid circular imports
try:
    from .flow import Flow
except ImportError:
    Flow = Any  # Type hint for Flow

from .model import Model
# Import all required classes from agent_loop unconditionally for runtime use
from .agent_loop import AgentLoop, TeamLoopCoordinator, AgentState

# ==================== Constants ====================
logger = logging.getLogger("glue.team")
KB_BASE_PATH = os.environ.get("GLUE_KB_PATH", "./glue_kb_data") # Configurable KB path

# Parameter normalization mappings
# Maps common alternative parameter names to their expected parameter names
# Key: tool name
# Value: dictionary mapping alternative parameter names to expected parameter names
# Note: Do not map parameters to themselves, as this creates redundant operations
TOOL_PARAM_MAPPINGS = {
    "web_search": {
        "search_term": "query",   # Maps 'search_term' parameter to 'query'
        "q": "query",             # Maps 'q' parameter to 'query'
        "query_text": "query",    # Maps 'query_text' parameter to 'query'
        "search_query": "query"   # Maps 'search_query' parameter to 'query'
    },
    "communicate": {
        "recipient_type": "target_type",     # Maps 'recipient_type' parameter to 'target_type'
        "recipient": "target_name",          # Maps 'recipient' parameter to 'target_name'
        "content": "message",                # Maps 'content' parameter to 'message'
        "text": "message",                   # Maps 'text' parameter to 'message'
        "recipient_name": "target_name",     # Maps 'recipient_name' parameter to 'target_name'
        "target": "target_name",             # Maps 'target' parameter to 'target_name'
        "type": "target_type"                # Maps 'type' parameter to 'target_type'
    }
    # Add mappings for other tools as needed
}

# ==================== Helper Classes ====================
@dataclass
class Agent:
    """Represents an agent within a team."""
    name: str
    model: Model  # The underlying model instance for this agent
    role: str
    default_adhesives: Set[AdhesiveType] = field(default_factory=lambda: {AdhesiveType.GLUE}) # Default to Glue
    agent_loop: Optional['AgentLoop'] = None # Use string forward reference
    velcro_memory: Dict[str, Any] = field(default_factory=dict) # For Velcro adhesive results

# ==================== Class Definition ====================
class Team:
    """
    Team implementation for GLUE framework.
    Manages model collaboration, tool sharing, and result persistence.
    Now uses explicit TeamLead and Agent roles.
    """
    def __init__(
        self,
        name: str,
        mode: str = "non-interactive", # Added mode parameter
        config: Optional[TeamConfig] = None,
        # DEPRECATED: lead and members args are for backward compatibility only
        lead_model_instance: Optional[Model] = None,
        member_model_instances: Optional[List[Model]] = None
    ):
        self.name = name
        self.mode = mode # Store mode
        self.config = config or TeamConfig(name=name, lead="", members=[], tools=[])
        
        # Core components
        self.lead: Optional[Model] = None  # The Model instance acting as the lead orchestrator
        self.agents: Dict[str, Agent] = {} # Dictionary of Agent instances keyed by name
        self._tools: Dict[str, Any] = {}
        
        # State management
        self.shared_results: Dict[str, ToolResult] = {} # Corresponds to 'Glue' KB conceptually
        self.knowledge_base = KnowledgeBase(base_path=KB_BASE_PATH, team_id=self.name) # Initialize KB
        self.conversation_history: List[Message] = []
        self.relationships: Dict[str, str] = {}  # Team magnetic relationships
        self.repelled_by: Set[str] = set()      # Teams that repel this one
        
        # Flow management
        self.incoming_flows: List[Any] = []
        self.outgoing_flows: List[Any] = []
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.pending_broadcasts: Dict[str, asyncio.Future] = {} # For tracking broadcast responses
        self.response_handlers = {}
        self.running = False # Initialize running flag

        # Agent loop management (will need refactoring)
        self.agent_loops: Dict[str, AgentLoop] = {} # Might change: loops could be within Agent/Lead directly
        self.loop_coordinator: Optional[TeamLoopCoordinator] = None
        
        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Handle backward compatibility with tests using DEPRECATED args
        if lead_model_instance is not None:
            # Assume the lead model instance is the lead
            self.set_lead(lead_model_instance)
            # Add other members as agents with default roles/adhesives
            if member_model_instances:
                for member_model in member_model_instances:
                    if member_model.name != lead_model_instance.name: # Avoid adding lead twice
                        default_agent = Agent(
                            name=member_model.name,
                            model=member_model,
                            role="member" # Default role
                        )
                        self.add_agent(default_agent)

        self.logger = logging.getLogger(f"glue.team.{self.name}") # Logger init
        self.logger.info(f"Initializing team '{self.name}' in '{self.mode}' mode.") # Log mode

    # Property for test compatibility (points to lead model)
    @property
    def model(self):
        """Get the lead model for test compatibility."""
        return self.lead # Directly return the lead model instance
        
    # Override the tools property to return a list for test compatibility
    @property
    def tools(self):
        """Get tools as a list for test compatibility."""
        # The test is expecting a list of tools, not a dictionary
        return list(self._tools.values())
        
    @tools.setter
    def tools(self, value):
        """Set tools dictionary."""
        self._tools = value

    # List-like access to tools for test compatibility
    def __getitem__(self, key):
        """Support list-like access to tools for test compatibility."""
        if isinstance(key, int):
            # If key is an integer, return the tool at that index
            tool_values = list(self._tools.values())
            if 0 <= key < len(tool_values):
                return tool_values[key]
            raise KeyError(key)
        # Otherwise, delegate to the tools dictionary
        return self._tools[key]
        
    # Make tools iterable for test compatibility
    def __iter__(self):
        """Support iteration over tools for test compatibility."""
        return iter(self._tools.values())
        
    # Support len() for test compatibility
    def __len__(self):
        """Support len() for test compatibility."""
        return len(self._tools)

    # ==================== Properties ====================
    # Note: Duplicate @property tools removed, keeping the one with list conversion for tests

    # ==================== Core Methods ====================
    def set_lead(self, lead_model: Model) -> None:
        """Set the lead model for the team."""
        if self.lead and self.lead.name != lead_model.name:
            logger.warning(f"Replacing existing lead {self.lead.name} with {lead_model.name} in team {self.name}")
        
        self.lead = lead_model
        self.lead.team = self # Ensure team reference is set
        self.lead.role = "lead" # Assign lead role explicitly
        self.config.lead = lead_model.name # Update config name
        
        # Ensure lead model has all team tools
        self._register_tools_for_model(self.lead)
        
        self.updated_at = datetime.now()
        logger.info(f"Set model {lead_model.name} as lead for team {self.name}")

    def add_agent(self, agent: Agent) -> None:
        """Add an agent instance to the team."""
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already in team {self.name}. Replacing.")
        
        self.agents[agent.name] = agent
        agent.model.team = self # Set team reference on the agent's model
        
        # Ensure agent's model has all team tools
        self._register_tools_for_model(agent.model)
            
        # Update config member list (if not already present)
        if agent.name not in self.config.members:
            self.config.members.append(agent.name)
            
        self.updated_at = datetime.now()
        logger.info(f"Added agent {agent.name} (Role: {agent.role}) to team {self.name}")

    def _register_tools_for_model(self, model_instance: Model):
        """Helper to register all current team tools for a given model instance."""
        if not hasattr(model_instance, 'add_tool_sync') or not callable(model_instance.add_tool_sync):
            logger.warning(f"Model {model_instance.name} missing add_tool_sync method.")
            return
            
        for tool_name, tool in self._tools.items():
            # Check if tool exists to prevent errors if add_tool_sync doesn't handle duplicates
            if tool_name not in model_instance.tools: 
                try:
                    model_instance.add_tool_sync(tool_name, tool)
                except Exception as e:
                     logger.error(f"Error adding tool {tool_name} to model {model_instance.name}: {e}")

    # DEPRECATED: Use set_lead and add_agent instead
    async def add_member(
        self,
        model: Model,
        role: str = "member",
        tools: Optional[Set[str]] = None # Tools arg seems less relevant now with dynamic init plan
    ) -> None:
        """DEPRECATED: Add a model to the team. Use set_lead or add_agent."""
        logger.warning("add_member is deprecated. Use set_lead or add_agent.")
        if role == "lead":
            self.set_lead(model)
        else:
            # Create a default Agent wrapper
            agent = Agent(name=model.name, model=model, role=role)
            self.add_agent(agent)
            # Note: Specific tools for member ignored, relies on team tools now

    async def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to this team's toolbox."""
        # Add tool to team's internal registry
        self._tools[name] = tool
        
        # Initialize the tool if needed (keep existing logic)
        if hasattr(tool, "initialize") and callable(tool.initialize) and hasattr(tool, "_initialized"):
            if not tool._initialized:
                try:
                    await tool.initialize()
                    logger.info(f"Initialized tool {name} for team {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize tool {name}: {e}")
        
        # Register tool for the lead model
        if self.lead:
            self._register_tools_for_model(self.lead)
            
        # Register tool for all agent models
        for agent in self.agents.values():
            self._register_tools_for_model(agent.model)
        
        # Update config tool list (if not already present)
        if name not in self.config.tools:
             self.config.tools.append(name) # Assuming tools are stored by name in config

        logger.info(f"Added tool {name} to team {self.name} toolbox and registered for lead/agents")
        self.updated_at = datetime.now()

    async def share_result(
        self,
        tool_name: str,
        result: ToolResult,
        agent_name: Optional[str] = None # Changed from model_name to agent_name
    ) -> None:
        """Share a tool result within the team based on adhesive.
        Needs integration with adhesive logic implementation.
        """
        # Determine adhesive (using result.adhesive which should be set by AgentLoop)
        adhesive = result.adhesive
        
        if adhesive is None:
             logger.warning(f"Adhesive type missing in ToolResult for tool {tool_name}. Cannot process sharing.")
             return
        
        # Process based on adhesive type
        if adhesive == AdhesiveType.GLUE:
            # Persist to shared KB via KnowledgeBase instance
            # Package the data appropriately
            # TODO: Add verification_status based on result.metadata
            verification_status = result.metadata.get("verification_status", "unverified")
            entry_data = {
                # Add a unique ID for the KB entry itself
                "id": str(uuid.uuid4()),
                "tool_name": tool_name,
                # Serialize the actual result content if it's complex
                "result_content": result.result,
                "error": result.error, # Include error if tool failed
                "source_agent_or_lead": agent_name,
                "team_id": self.name,
                "result_timestamp": result.timestamp.isoformat(),
                "verification_status": verification_status, # Add status
                # Include other relevant metadata from ToolResult if needed
                "metadata": result.metadata
            }
            
            entry_id = self.knowledge_base.add_entry(entry_data)
            if entry_id:
                logger.info(f"Stored GLUE result {entry_id} (Status: {verification_status}) from tool {tool_name} in KB for team {self.name}")
                # Optionally, still keep in self.shared_results for quick access?
                # self.shared_results[entry_id] = result
        elif adhesive == AdhesiveType.VELCRO:
             # Logic handled in AgentLoop._execute_tool_action
             logger.debug(f"share_result called with VELCRO for tool {tool_name}, handling is done in AgentLoop.")
             pass
        elif adhesive == AdhesiveType.TAPE:
            # Logic handled in AgentLoop (effectively ignored after use)
            logger.debug(f"share_result called with TAPE for tool {tool_name}, handling is done in AgentLoop.")
            pass # Add pass to make the block valid
        
        self.updated_at = datetime.now()

    async def process_message(
        self,
        content: Any,
        source_model: Optional[str] = None,
        target_model: Optional[str] = None,
        from_model: Optional[str] = None
    ) -> str:
        """DEPRECATED? Process a message within the team. Use AgentLoop/Communicate instead."""
        logger.warning("Team.process_message is likely deprecated. Agent interactions should go through AgentLoop.")
        # Handle backward compatibility
        if from_model is not None and source_model is None:
            source_model = from_model
            
        # Handle dict-like messages
        message_content = content
        if isinstance(content, dict) and "content" in content:
            message_content = content["content"]
            
        # Get source model instance (either lead or agent's model)
        source_instance = None
        if source_model:
             if self.lead and self.lead.name == source_model:
                 source_instance = self.lead
             elif source_model in self.agents:
                 source_instance = self.agents[source_model].model
        
        if not source_instance and self.lead: # Default to lead if no source specified
             source_instance = self.lead

        if not source_instance:
            raise ValueError(f"Source model '{source_model or 'default lead'}' not found in team {self.name}")

        # --- Simplified processing ---
        # Add incoming message to history (assuming user role for simplicity)
        self.conversation_history.append(Message(role="user", content=str(message_content)))

        # Generate response using the source model instance
        response_content = await source_instance.generate(str(message_content))

        # Add response to history
        self.conversation_history.append(Message(role="assistant", content=response_content))

        # Basic tool call handling (might be needed for simple tests)
        tool_call_data = extract_json(response_content)
        if tool_call_data and isinstance(tool_call_data, dict) and "tool_name" in tool_call_data:
             # Log but don't execute here - let AgentLoop handle execution
             logger.info(f"[Deprecated process_message] Detected tool call: {tool_call_data['tool_name']}")

        return response_content # Return the direct response

    # ==================== Agent Loop Management (Needs Refactoring) ====================

    async def create_agent_loops(self) -> None:
        """Create AgentLoop instances for the lead and each agent."""
        self.agent_loops = {} # Clear existing loops
        self.loop_coordinator = TeamLoopCoordinator(self.name)
        
        # Create loop for the Lead
        if self.lead:
            lead_agent_id = f"{self.name}-lead-{self.lead.name}"
            # Pass team_ref=self, agent_ref=None for the lead
            lead_loop = AgentLoop(agent_id=lead_agent_id, team_id=self.name, model=self.lead, team_ref=self)
            # Register tools for the lead loop (using lead's model which should have them)
            for tool_name, tool_instance in self._tools.items(): # Use team's _tools
                 lead_loop.register_tool(tool_name, tool_instance) # Pass the instance
            self.agent_loops[lead_agent_id] = lead_loop
            self.loop_coordinator.add_agent(lead_loop)
            logger.info(f"Created agent loop for Lead: {lead_agent_id}")
        else:
             logger.warning(f"Team {self.name} has no lead defined. Cannot create lead agent loop.")

        # Create loops for Agents
        for agent_name, agent_instance in self.agents.items():
            agent_loop_id = f"{self.name}-agent-{agent_name}"
            # Pass team_ref=self and agent_ref=agent_instance for agents
            agent_loop = AgentLoop(agent_id=agent_loop_id, team_id=self.name, model=agent_instance.model, team_ref=self, agent_ref=agent_instance)
            # Register tools for the agent loop (using agent's model)
            for tool_name, tool_instance in self._tools.items(): # Use team's _tools
                agent_loop.register_tool(tool_name, tool_instance) # Pass the instance
            self.agent_loops[agent_loop_id] = agent_loop
            agent_instance.agent_loop = agent_loop # Link loop back to agent instance
            self.loop_coordinator.add_agent(agent_loop)
            logger.info(f"Created agent loop for Agent: {agent_loop_id}")
            
    async def start_agent_loops(self, initial_input: Optional[str] = None) -> None:
        """Start all agent loops in the team"""
        # Ensure agent loops are created
        await self.create_agent_loops()
        
        # Start the coordinator
        if self.loop_coordinator:
            await self.loop_coordinator.start()
            logger.info(f"Started agent loops for team {self.name}")
        else:
            logger.error(f"Cannot start agent loops: coordinator not initialized for team {self.name}")
            
    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of agent loops
        
        Args:
            agent_id: Optional specific agent ID to get status for
            
        Returns:
            Status information
        """
        if agent_id:
            if agent_id in self.agent_loops:
                return self.agent_loops[agent_id].get_status()
            else:
                return {"error": f"Agent {agent_id} not found"}
        else:
            # Get status of all agents
            return {
                "team": self.name,
                "agents": {agent_id: loop.get_status() for agent_id, loop in self.agent_loops.items()},
                "coordinator": self.loop_coordinator.get_status() if self.loop_coordinator else None
            }
            
    async def terminate_agent_loops(self, reason: str = "requested") -> None:
        """Terminate all agent loops in the team
        
        Args:
            reason: Reason for termination
        """
        if self.loop_coordinator:
            self.loop_coordinator.terminate(reason)
            logger.info(f"Terminated agent loops for team {self.name}: {reason}")
        
        # Reset agent loops
        self.agent_loops = {}
        self.loop_coordinator = None

    # ==================== Compatibility/Helper Methods ====================

    # DEPRECATED: Use add_agent instead.
    def add_member_sync(
        self,
        model: Model,
        role: str = "member",
        tools: Optional[Set[str]] = None
    ) -> None:
        """DEPRECATED synchronous version of add_member."""
        logger.warning("add_member_sync is deprecated. Use set_lead or add_agent.")
        if role == "lead":
            self.set_lead(model)
        else:
             agent = Agent(name=model.name, model=model, role=role)
             self.add_agent(agent)

    # ==================== Magnetic Field Methods ====================
    def set_relationship(self, team_name: str, relationship: str) -> None:
        """Set magnetic relationship with another team"""
        if team_name in self.repelled_by:
            raise ValueError(f"Cannot set relationship with {team_name} - repelled")
            
        self.relationships[team_name] = relationship
        self.updated_at = datetime.now()
        logger.info(f"Set {relationship} relationship with team {team_name}")

    def break_relationship(self, team_name: str) -> None:
        """Break relationship with another team"""
        if team_name in self.relationships:
            del self.relationships[team_name]
            logger.info(f"Broke relationship with team {team_name}")

    def repel(self, team_name: str) -> None:
        """Set repulsion with another team"""
        self.repelled_by.add(team_name)
        if team_name in self.relationships:
            del self.relationships[team_name]
        logger.info(f"Set repulsion with team {team_name}")

    async def get_relationships(self) -> Dict[str, str]:
        """Get all team relationships"""
        return self.relationships.copy()

    # ==================== Helper Methods ====================
    def get_model_tools(self, model_name: str) -> Dict[str, Any]:
        """Get tools available to a specific model"""
        target_model = None
        if self.lead and self.lead.name == model_name:
             target_model = self.lead
        elif model_name in self.agents:
             target_model = self.agents[model_name].model
        else:
             raise ValueError(f"Model {model_name} not found in team {self.name}")

        # Handle different model implementations for storing tools
        if hasattr(target_model, 'tools'):
            return target_model.tools
        elif hasattr(target_model, '_tools'): # Check for private attribute if public fails
            return target_model._tools
            
        return {} # Return empty if no tools attribute found

    async def try_establish_relationship(self, target_team: str) -> Dict[str, Any]:
        """Attempt to automatically establish a relationship with another team.
        
        Args:
            target_team: Name of the target team
            
        Returns:
            Dict with success status, relationship type if established, and error message if failed
        """
        result = {
            "success": False,
            "relationship_type": None,
            "error": None
        }
        
        # Check if we already have a relationship
        if target_team in self.relationships:
            result["success"] = True
            result["relationship_type"] = self.relationships[target_team]
            return result
        
        # Try to find a flow and establish relationship
        if hasattr(self, 'outgoing_flows') and self.outgoing_flows:
            # First check outgoing flows
            for flow in self.outgoing_flows:
                if hasattr(flow, 'target') and flow.target.name == target_team:
                    # Found outgoing flow, establish bidirectional relationship
                    self.relationships[target_team] = FlowType.BIDIRECTIONAL.value
                    result["success"] = True
                    result["relationship_type"] = FlowType.BIDIRECTIONAL.value
                    logger.info(f"Automatically established BIDIRECTIONAL relationship with team {target_team} based on existing flow")
                    return result
            
        # If we have incoming flows, check those too
        if hasattr(self, 'incoming_flows') and self.incoming_flows:
            for flow in self.incoming_flows:
                if hasattr(flow, 'source') and flow.source.name == target_team:
                    # Found incoming flow, establish bidirectional relationship
                    self.relationships[target_team] = FlowType.BIDIRECTIONAL.value
                    result["success"] = True
                    result["relationship_type"] = FlowType.BIDIRECTIONAL.value
                    logger.info(f"Automatically established BIDIRECTIONAL relationship with team {target_team} based on existing flow")
                    return result
        
        # No flows found, can't establish relationship
        result["error"] = f"No flows exist between team {self.name} and {target_team}"
        return result

    def get_shared_results(self) -> Dict[str, ToolResult]:
        """Get shared tool results"""
        return self.shared_results

    async def cleanup(self) -> None:
        """Clean up resources used by the team."""
        # Stop message processing task first
        self.running = False
        if self.processing_task and not self.processing_task.done():
             self.processing_task.cancel()
             try:
                 await asyncio.wait_for(self.processing_task, timeout=1.0)
             except (asyncio.CancelledError, asyncio.TimeoutError):
                 logger.info("Message processing task cancelled during cleanup.")
             except Exception as e:
                 logger.error(f"Error waiting for message processing task during cleanup: {e}")
        self.processing_task = None

        # Clean up agent loops
        if self.loop_coordinator:
            await self.loop_coordinator.cleanup()
            
        # Clean up tools
        for tool_name, tool in self._tools.items():
            if hasattr(tool, 'cleanup') and callable(tool.cleanup):
                try:
                    # Check if cleanup is async
                    if asyncio.iscoroutinefunction(tool.cleanup):
                        await tool.cleanup()
                    else:
                        tool.cleanup() # Call synchronously if not async
                except Exception as e:
                    logger.error(f"Error cleaning up tool {tool_name}: {e}")

        # Clean up models (handle potential missing cleanup)
        model_instances = [self.lead] + [agent.model for agent in self.agents.values()]
        for model in filter(None, model_instances): # Filter out None lead
            if hasattr(model, 'cleanup') and callable(model.cleanup):
                try:
                    if asyncio.iscoroutinefunction(model.cleanup):
                        await model.cleanup()
                    else:
                        model.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up model {model.name}: {e}")

        # Clear shared results and history
        self.shared_results.clear()
        self.conversation_history.clear()
        
        logger.info(f"Cleaned up team {self.name}")

    async def setup(self) -> None:
        """Set up the team by initializing any required resources.
        
        This method is called during application setup to initialize
        team resources, configure tools, and establish connections.
        """
        # Add tools from config if they exist
        if hasattr(self.config, "tools") and self.config.tools:
            for tool_name in self.config.tools:
                # Tools will be added during app setup
                pass
                
        # Start the internal message processing loop unconditionally
        self.running = True # Set running flag before starting task
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_messages())
            logger.debug(f"Started message processing task for team {self.name}")
        else:
             logger.debug(f"Message processing task already running for team {self.name}")
                
        logger.info(f"Team {self.name} setup complete")

    # ==================== Error Handling ====================
    async def _handle_error(self, error: Exception) -> None:
        """Handle team-level errors"""
        logger.error(f"Team error in {self.name}: {str(error)}")
        # Depending on severity, might trigger termination or recovery
        raise error # Re-raise for now

    # ==================== Flow Management Methods ====================
    def register_outgoing_flow(self, flow: Any) -> None:
        """Register an outgoing flow with this team.
        
        Args:
            flow: Flow to register
        """
        if flow not in self.outgoing_flows:
            self.outgoing_flows.append(flow)
            # Ensure target has a name attribute before logging
            target_name = getattr(getattr(flow, 'target', None), 'name', 'unknown')
            logger.info(f"Registered outgoing flow from {self.name} to {target_name}")
            
    def register_incoming_flow(self, flow: Any) -> None:
        """Register an incoming flow with this team.
        
        Args:
            flow: Flow to register
        """
        if flow not in self.incoming_flows:
            self.incoming_flows.append(flow)
            # Ensure source has a name attribute before logging
            source_name = getattr(getattr(flow, 'source', None), 'name', 'unknown')
            logger.info(f"Registered incoming flow to {self.name} from {source_name}")
            
            # Task is now started in setup

    def unregister_outgoing_flow(self, flow: Any) -> None:
        """Unregister an outgoing flow from this team.
        
        Args:
            flow: Flow to unregister
        """
        if flow in self.outgoing_flows:
            self.outgoing_flows.remove(flow)
            target_name = getattr(getattr(flow, 'target', None), 'name', 'unknown')
            logger.info(f"Unregistered outgoing flow from {self.name} to {target_name}")
            
    def unregister_incoming_flow(self, flow: Any) -> None:
        """Unregister an incoming flow from this team.
        
        Args:
            flow: Flow to unregister
        """
        if flow in self.incoming_flows:
            self.incoming_flows.remove(flow)
            source_name = getattr(getattr(flow, 'source', None), 'name', 'unknown')
            logger.info(f"Unregistered incoming flow to {self.name} from {source_name}")
            
    async def receive_message(self, message: Dict[str, Any], sender: Any) -> None:
        """DEPRECATED? Handle messages received from flows. Use receive_information."""
        # This method seems redundant if receive_information handles flow input
        logger.warning("Team.receive_message called, but receive_information should handle flow inputs.")
        # For backward compatibility, maybe try to parse and queue?
        source_team = message.get("metadata", {}).get("source_team", "unknown")
        await self.receive_information(source_team, message) # Pass it to the new handler

    async def _process_messages(self) -> None:
        """Internal loop to process messages from the team queue."""
        # Ensure self.running is defined, maybe in __init__? Assuming it is.
        if not hasattr(self, 'running'): self.running = True # Simple fix if missing

        while self.running:
            try:
                # Wait for a message package from the queue
                message_package = await self.message_queue.get()

                payload_dict = None
                target_loop_id = None
                source_info = message_package.get("source", "internal") # Default to internal

                # --- Extract Payload and Target ---
                if source_info == "flow": # Message from another team via receive_information
                    payload_dict = message_package.get("payload")
                    # TODO: V1 - Route external messages to Lead by default? Or check intended_recipients?
                    if self.lead:
                         target_loop_id = f"{self.name}-lead-{self.lead.name}"
                         logger.debug(f"Routing message from external flow to Lead: {target_loop_id}")
                    else:
                         logger.warning(f"Received message from external flow but no Lead defined in team {self.name}. Discarding.")
                         self.message_queue.task_done()
                         continue
                elif source_info == "internal": # Message from Communicate tool or Agent Feedback
                     target_loop_id = message_package.get("target_loop_id")
                     payload_dict = message_package.get("content") # Internal messages put payload in 'content'
                else: # Unknown source
                     logger.warning(f"Unknown message package source: {source_info}. Discarding.")
                     self.message_queue.task_done()
                     continue

                # --- Validate Payload ---
                if not payload_dict or not isinstance(payload_dict, dict):
                    logger.warning(f"Invalid or empty payload received in queue package: {message_package}. Discarding.")
                    self.message_queue.task_done()
                    continue

                # Basic validation (check for essential V1 fields)
                # More robust validation using Pydantic model could be added here
                required_fields = ['task_id', 'sender_agent_id', 'sender_team_id', 'timestamp', 'message_type', 'adhesive_type', 'content']
                if not all(field in payload_dict for field in required_fields):
                     logger.warning(f"Received payload missing required V1 fields: {payload_dict}. Discarding.")
                     self.message_queue.task_done()
                     continue

                # --- Route to Target Loop ---
                if target_loop_id and target_loop_id in self.agent_loops:
                    target_loop = self.agent_loops[target_loop_id]
                    logger.info(f"Routing message type '{payload_dict.get('message_type')}' to loop {target_loop_id}")

                    if hasattr(target_loop, 'handle_message') and callable(target_loop.handle_message):
                        # Pass the validated payload dictionary to the agent loop
                        await target_loop.handle_message(payload_dict)
                    else:
                        logger.warning(f"Loop {target_loop_id} does not have a callable 'handle_message' method. Cannot deliver message.")
                        # Consider alternative handling: requeue? error state?

                elif target_loop_id: # Target specified but not found
                    logger.warning(f"Target loop {target_loop_id} not found in team {self.name}. Discarding message.")

                else: # No specific target loop identified (e.g., external message routed to non-existent lead)
                     logger.warning(f"No valid target loop identified for payload: {payload_dict}. Discarding.")

                # --- Handle Team-Level Broadcasts (Example - Needs Refinement) ---
                # This logic might need to move or be triggered differently based on V1 payload types
                message_type = payload_dict.get("message_type")
                # Assuming broadcast info might be in content or metadata for specific types
                metadata = payload_dict.get("metadata", {}) # Check if metadata exists
                is_internal_broadcast = metadata.get("internal") is True
                broadcast_id = metadata.get("broadcast_id")

                if is_internal_broadcast and broadcast_id:
                     logger.debug(f"Processing internal broadcast ID: {broadcast_id}")
                     if broadcast_id in self.pending_broadcasts:
                         broadcast_future = self.pending_broadcasts.get(broadcast_id)
                         if broadcast_future and not broadcast_future.done():
                             # TODO: Implement actual broadcast logic here.
                             # This likely involves routing the payload to *all* other agent loops
                             # and potentially aggregating responses before setting the future result.
                             # Placeholder result for now:
                             broadcast_result = {"status": "acknowledged", "info": "Broadcast received by team processor."}
                             
                             logger.info(f"Completing future for internal broadcast ID: {broadcast_id}")
                             try:
                                 broadcast_future.set_result(broadcast_result)
                             except asyncio.InvalidStateError:
                                 logger.warning(f"Future for broadcast {broadcast_id} was already done when trying to set result.")
                             finally: # Ensure pop happens even if set_result fails
                                 self.pending_broadcasts.pop(broadcast_id, None)
                         else:
                             logger.warning(f"Future for broadcast {broadcast_id} not found or already done in pending_broadcasts.")
                     else:
                         logger.warning(f"Received internal broadcast message for ID {broadcast_id}, but no pending future found.")
                # Remove old untargeted message handling logic as routing is now explicit
                # elif content_type == "some_general_info":
                #     pass
                # else:
                #     logger.warning(f"Discarding untargeted message of type '{content_type}'")

                # Mark task as done
                self.message_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Message processing loop for team {self.name} cancelled.")
                break # Exit loop cleanly on cancellation
            except Exception as e:
                logger.error(f"Error processing message queue for team {self.name}: {e}", exc_info=True)
                # Avoid breaking the loop on processing errors, just log and continue
                # Ensure task_done is called even on error if possible
                try:
                    self.message_queue.task_done()
                except ValueError: # May happen if queue is empty/task already done
                    pass
                await asyncio.sleep(1) # Prevent fast error loops

    async def share_glue_entry(self, kb_entry_id: str, target_team_name: str) -> bool:
        """Share a specific persisted GLUE entry with another team."""
        kb_entry = self.knowledge_base.get_entry(kb_entry_id)
        if not kb_entry:
             logger.error(f"Cannot share GLUE entry: ID {kb_entry_id} not found in KB.")
             return False

        # Construct message data according to V1 standard for send_information
        message_data = {
            "type": MessageType.GLUE_DATA_SHARE.value, # Use enum value
            "content": {"kb_entry": kb_entry},
            "metadata": {
                 "task_id": kb_entry.get("metadata", {}).get("task_id", f"share_{kb_entry_id}"), # Try to get original task ID
                 # Add other relevant metadata if needed
            }
        }

        logger.info(f"Attempting to share GLUE entry {kb_entry_id} with team {target_team_name}")
        success = await self.send_information(target_team_name, message_data)
        if success:
             logger.info(f"Successfully initiated sending GLUE entry {kb_entry_id} to team {target_team_name}")
        else:
             logger.warning(f"Failed to initiate sending GLUE entry {kb_entry_id} to team {target_team_name}")
        return success # Return status from send_information

    async def broadcast_pause_query(self, query_content: str) -> None:
         """Broadcast a pause query to all connected team leads if in interactive mode."""
         if self.mode != 'interactive': # Check mode
             self.logger.info(f"Not broadcasting pause query: Team '{self.name}' is in '{self.mode}' mode.")
             return
             
         if not self.outgoing_flows:
              logger.info("No outgoing flows, cannot broadcast pause query.")
              return
              
         # Construct message data according to V1 standard
         message_data = {
              "type": MessageType.PAUSE_QUERY.value, # Use enum value
              "content": {"query_content": query_content},
              "metadata": {
                   # Add task_id if available/relevant for the pause context
                   # "task_id": current_task_id,
              }
         }
         
         logger.info(f"Broadcasting pause query (Interactive Mode) to connected teams: {query_content}")
         broadcast_tasks = []
         # Find unique target team names from flows
         target_teams = {flow.target.name for flow in self.outgoing_flows if hasattr(flow, 'target') and hasattr(flow.target, 'name')}
         
         for target_team_name in target_teams:
              if target_team_name != self.name: # Don't broadcast to self
                  # Pass the structured message_data to send_information
                  broadcast_tasks.append(self.send_information(target_team_name, message_data))
                  
         results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
         
         # Process results (logging remains similar)
         for target_team_name, result in zip(target_teams - {self.name}, results):
             if isinstance(result, Exception):
                 logger.error(f"Error broadcasting pause query to team {target_team_name}: {result}")
             elif not result: # Check for False or error dict from send_information
                 error_info = f" (send_information returned {result})" if isinstance(result, dict) else ""
                 logger.warning(f"Failed to broadcast pause query to team {target_team_name}{error_info}.")
             else:
                 logger.info(f"Successfully initiated broadcast of pause query to team {target_team_name}.")

    # --- send_information and receive_information need review for robustness ---
    async def send_information(self, target_team_name: str, message_data: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
        """
        Send information (structured as V1MessagePayload) to another team via flows.

        Args:
            target_team_name: Name of the target team.
            message_data: A dictionary representing the V1MessagePayload content.
                          Must include 'type', 'content', and 'metadata'.

        Returns:
            True if successfully sent via at least one flow, False otherwise.
            Can return a dict with error details in specific failure cases.
        """
        # --- Construct V1 Payload ---
        # Expect message_data to already contain the core 'content' and 'type'
        # We add the standard sender/timestamp metadata here.
        message_type_str = message_data.get("type")
        content_payload = message_data.get("content")
        metadata = message_data.get("metadata", {}) # Use provided metadata or default

        if not message_type_str or content_payload is None:
             logger.error(f"send_information requires 'type' and 'content' in message_data.")
             return {"error": "Invalid message_data format", "status": "failed"}

        try:
            message_type_enum = MessageType(message_type_str)
        except ValueError:
             logger.error(f"Invalid message type '{message_type_str}' in send_information.")
             return {"error": f"Invalid message type: {message_type_str}", "status": "failed"}

        # Determine sender ID (assume lead if not specified in metadata)
        sender_id = metadata.get("source_agent_or_lead", self.lead.name if self.lead else self.name)
        # Determine adhesive (default to TAPE for inter-team info unless specified)
        adhesive_type_enum = metadata.get("adhesive_type", AdhesiveType.TAPE)
        if isinstance(adhesive_type_enum, str):
             try:
                 adhesive_type_enum = AdhesiveType(adhesive_type_enum)
             except ValueError:
                 adhesive_type_enum = AdhesiveType.TAPE
        elif not isinstance(adhesive_type_enum, AdhesiveType):
             adhesive_type_enum = AdhesiveType.TAPE

        # TODO: Get task_id reliably, maybe from metadata?
        task_id = metadata.get("task_id", f"inter_team_{datetime.now().isoformat()}")

        try:
            payload = V1MessagePayload(
                task_id=task_id,
                sender_agent_id=sender_id,
                sender_team_id=self.name,
                timestamp=datetime.now().isoformat(),
                message_type=message_type_enum,
                adhesive_type=adhesive_type_enum,
                content=content_payload,
                origin_tool_id=metadata.get("origin_tool_id") # Pass if available
            )

            # Convert payload to dict using the new method
            payload_dict = payload.to_dict()

        except Exception as e:
            logger.error(f"Error constructing V1 payload for inter-team message: {e}", exc_info=True)
            return {"error": f"Internal error constructing message payload: {e}", "status": "failed"}

        # --- Find Flow and Send ---
        flow_found = False
        sent_successfully = False
        for flow in self.outgoing_flows:
            # Check if flow object has 'target' attribute and 'name' matches
            if hasattr(flow, 'target') and hasattr(flow.target, 'name') and flow.target.name == target_team_name:
                flow_found = True
                try:
                    # Send the structured payload dictionary
                    await flow.send(payload_dict)
                    logger.info(f"Sent message type '{payload.message_type.value}' to team {target_team_name} via flow {flow.name}")
                    sent_successfully = True
                    # Optionally break if only one flow needed, or continue to send via all matching flows
                    # For now, assume sending via the first matching flow is sufficient
                    break
                except Exception as e:
                    logger.error(f"Error sending message to team {target_team_name} via flow {flow.name}: {e}")
                    # Continue trying other flows if available

        if not flow_found:
            logger.warning(f"No outgoing flow found for target team {target_team_name}")
            return False # Indicate no flow found

        return sent_successfully # Return True if sent via at least one flow, False if all attempts failed

    async def receive_information(self, source_team_name: str, payload_dict: Dict[str, Any]) -> Union[bool, Dict[str, Any]]:
        """
        Callback invoked by a Flow when information (as V1 payload dict) is received.
        Puts the received payload onto the internal message queue for processing.
        """
        if not isinstance(payload_dict, dict):
             logger.error(f"Received non-dict payload from {source_team_name} via flow. Discarding: {payload_dict}")
             return False # Indicate error

        logger.info(f"Received V1 payload dict from {source_team_name}. Queueing for processing.")
        # Add source_team_name to the payload if not already present, for context
        # This assumes payload_dict is mutable, which should be fine
        if 'metadata' not in payload_dict:
             payload_dict['metadata'] = {} # Ensure metadata key exists
        if 'source_flow_team' not in payload_dict['metadata']:
             payload_dict['metadata']['source_flow_team'] = source_team_name

        # Put the entire received payload dictionary onto the queue, wrapped to indicate source
        # The wrapper helps _process_messages distinguish internal vs external origin if needed
        await self.message_queue.put({"source": "flow", "payload": payload_dict})
        return True # Acknowledge receipt to the flow

    # ... rest of the file ...
