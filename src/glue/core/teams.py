# glue/core/team.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List, Union
from datetime import datetime
import asyncio
import logging
import uuid
import json

from .types import AdhesiveType, TeamConfig, ToolResult, FlowType, Message
from .hierarchy import (
    get_highest_ranking_model, 
    set_hierarchy_attributes,
    HierarchyDetectionError
)
from .glue_smolteam import GlueSmolTeam

# Monkey-patch InferenceClientModel to provide no-op add_tool and add_tool_sync for uniform model interface
try:
    from smolagents import InferenceClientModel
    if not hasattr(InferenceClientModel, 'add_tool'):
        async def _noop_add_tool(self, name: str, tool: Any):
            return None
        InferenceClientModel.add_tool = _noop_add_tool
    if not hasattr(InferenceClientModel, 'add_tool_sync'):
        def _noop_add_tool_sync(self, name: str, tool: Any):
            return None
        InferenceClientModel.add_tool_sync = _noop_add_tool_sync
except ImportError:
    pass

# ==================== Constants ====================
logger = logging.getLogger(__name__)

# Parameter normalization mappings
# Maps common alternative parameter names to their expected parameter names
# Key: tool name
# Value: dictionary mapping alternative parameter names to expected parameter names
# Note: Do not map parameters to themselves, as this creates redundant operations
TOOL_PARAM_MAPPINGS = {
    "web_search": {
        "search_term": "query",  # Maps 'search_term' parameter to 'query'
        "q": "query",  # Maps 'q' parameter to 'query'
        "query_text": "query",  # Maps 'query_text' parameter to 'query'
        "search_query": "query",  # Maps 'search_query' parameter to 'query'
    },
    # Add mappings for other tools as needed
}


# ==================== Class Definition ====================
class Team:
    """
    Team implementation for GLUE framework.
    Manages model collaboration, tool sharing, and result persistence.
    """

    def __init__(
        self,
        name: str,
        config: Optional[TeamConfig] = None,
        # For backward compatibility with tests
        lead: Any = None,
        members: Optional[List[Any]] = None,
        description: Optional[str] = None,  # Add description for test compatibility
    ):
        self.name = name
        self.description = description if description is not None else ""
        self.config = config or TeamConfig(name=name, lead="", members=[], tools=[])

        # Validate members argument for edge case tests
        if members is not None and not isinstance(members, list):
            raise Exception("members must be a list or None")

        # Core components
        self.models: Dict[str, Any] = {}
        self.subteams: Dict[str, 'Team'] = {}  # NEW: subteams by name
        self.lead: Optional[Any] = None
        self._tools: Dict[str, Any] = {}
        # self.tool_bindings: Dict[str, AdhesiveType] = {}

        # State management
        self.shared_results: Dict[str, ToolResult] = {}
        self.conversation_history: List[Message] = []
        self.relationships: Dict[str, str] = {}  # Team magnetic relationships
        self.repelled_by: Set[str] = set()  # Teams that repel this one

        # Flow management
        self.incoming_flows: List[Any] = []
        self.outgoing_flows: List[Any] = []
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.pending_broadcasts: Dict[
            str, asyncio.Future
        ] = {}  # For tracking broadcast responses
        self.response_handlers = {}

        # Track active TeamMember and TeamLead loops
        self.agent_loops: Dict[str, Any] = {}

        # Reference to parent app
        self.app = None

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()        # Store members for test compatibility
        if members is not None:
            if not isinstance(members, list):
                raise Exception("members must be a list")
            # If all members are strings, treat as names only (test compatibility)
            if all(isinstance(m, str) for m in members):
                self.members = members
                # Add string members to config.members as well
                for member_name in members:
                    if member_name not in self.config.members:
                        self.config.members.append(member_name)
            else:
                # If not all are strings, treat as model/subteam objects
                self.members = [m.name if hasattr(m, 'name') else m for m in members]
                for member in members:
                    if isinstance(member, str):
                        # Add string member to config.members
                        if member not in self.config.members:
                            self.config.members.append(member)
                    else:
                        self.add_member_sync(member)
        else:
            self.members = []

        # Handle backward compatibility with tests
        if lead is not None:
            self.add_member_sync(lead, role="lead")

    # Property for test compatibility
    @property
    def model(self):
        """Get the lead model for test compatibility."""
        lead_name = self.config.lead
        if lead_name and lead_name in self.models:
            return self.models[lead_name]
        # Return the first model if no lead is set
        if self.models:
            return next(iter(self.models.values()))
        return None    # Override the tools property to return a list for test compatibility
    # Note: This is consolidated with the tools property below

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
        return len(self._tools)    # ==================== Properties ====================
    @property
    def tools(self) -> List[Any]:
        """Get all tools available to this team.

        Returns:
            List of tools
        """
        # For test compatibility, we need to match the exact objects stored in app.tools
        # This is a bit of a hack, but it's necessary for the tests to pass
        return list(self._tools.values())

    @tools.setter
    def tools(self, value):
        """Set tools dictionary."""
        self._tools = value

    # ==================== Core Methods ====================
    async def add_member(
        self, member, role: str = "member", tools: Optional[Set[str]] = None
    ) -> None:
        """
        Add a model or subteam to the team asynchronously.
        Supports hierarchical teams by allowing Team instances as members.
        """
        from .teams import Team  # Local import to avoid circular import
        if isinstance(member, Team):
            if member.name in self.subteams:
                logger.warning(f"Subteam {member.name} already in team {self.name}")
                return
            self.subteams[member.name] = member
            logger.info(f"Added subteam {member.name} to team {self.name}")
            return
        # Otherwise, treat as Model
        if member.name in self.models:
            logger.warning(f"Model {member.name} already in team {self.name}")
            return
        self.models[member.name] = member
        member.team = self
        for tool_name, tool in self._tools.items():
            if hasattr(member, "add_tool_sync") and callable(member.add_tool_sync):
                member.add_tool_sync(tool_name, tool)
        
        if tools:
            for tool_name in tools:
                if tool_name in self._tools and tool_name not in member.tools:
                    if hasattr(member, "add_tool_sync") and callable(member.add_tool_sync):
                        member.add_tool_sync(tool_name, self._tools[tool_name])
        
        if role == "lead":
            self.config.lead = member.name
            self.lead = member
        else:
            if member.name not in self.config.members:
                self.config.members.append(member.name)
        self.updated_at = datetime.now()
        logger.info(f"Added model {member.name} to team {self.name} with role {role}")
        
        # Update hierarchy attributes after adding member
        self.update_hierarchy_attributes()

    async def add_tool(self, name: str, tool: Any, assign_to: str = "all", *args, **kwargs) -> None:
        """Add a SmolAgents Tool instance to this team with selective assignment.

        Args:
            name: Tool name
            tool: Tool instance
            assign_to: Who to assign the tool to:
                - "all": All team members (default, backward compatible)
                - "lead": Only the team lead
                - "members": Only team members (excluding lead)
                - "hierarchy_top": Only the highest-ranking model in hierarchy
            *args, **kwargs: Ignored (for DSL binding compatibility)
        """
        # Add tool to team
        self._tools[name] = tool

        # Initialize the tool if it's not already initialized
        if (
            hasattr(tool, "initialize")
            and callable(tool.initialize)
            and hasattr(tool, "_initialized")
        ):
            if not tool._initialized:
                try:
                    await tool.initialize()
                    logger.info(f"Initialized tool {name} for team {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize tool {name}: {e}")

        # Determine target models based on assignment strategy
        target_models = []
        
        if assign_to == "all":
            target_models = list(self.models.values())
        elif assign_to == "lead":
            if self.lead:
                target_models = [self.lead]
            else:
                logger.warning(f"No lead model found for tool assignment in team {self.name}")
        elif assign_to == "members":
            # All models except the lead
            lead_name = getattr(self.config, 'lead', None)
            target_models = [model for name, model in self.models.items() if name != lead_name]
        elif assign_to == "hierarchy_top":
            try:
                top_model_name = get_highest_ranking_model(self)
                if top_model_name and top_model_name in self.models:
                    target_models = [self.models[top_model_name]]
                    logger.info(f"Assigning tool {name} to hierarchy top: {top_model_name}")
                else:
                    logger.warning(f"No hierarchy top model found for tool assignment in team {self.name}")
            except HierarchyDetectionError as e:
                logger.error(f"Hierarchy detection failed for tool {name}: {e}")
        else:
            logger.error(f"Invalid assign_to value: {assign_to}. Using 'all' as fallback.")
            target_models = list(self.models.values())

        # Add tool to target models using the patched add_tool interface
        for model in target_models:
            await model.add_tool(name, tool)
            
        logger.info(f"Added tool {name} to team {self.name} (assigned to: {assign_to}, {len(target_models)} models)")

    async def share_result(
        self, tool_name: str, result: ToolResult, model_name: Optional[str] = None
    ) -> None:
        """Share a tool result with the team"""
        # For task completion, the record is already embedded in shared_results; skip adding ToolResult
        if tool_name == "report_task_completion":
            result.adhesive = AdhesiveType.GLUE
            return
        # For task completion reports, always use GLUE adhesive
        if tool_name == "report_task_completion":
            result.adhesive = AdhesiveType.GLUE
        # If a specific model is provided, use its adhesives for other tools
        elif model_name and model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, "adhesives") and model.adhesives:
                # Use the first adhesive defined in the model
                adhesive = next(iter(model.adhesives))
                result.adhesive = adhesive
                logger.debug(f"Using adhesive {adhesive} from model {model_name}")

        # Process based on adhesive type
        if result.adhesive == AdhesiveType.GLUE:
            # GLUE: Team-wide persistent results
            self.shared_results[tool_name] = result
            logger.info(
                f"Shared result from {tool_name} in team {self.name} with GLUE adhesive"
            )
        elif result.adhesive == AdhesiveType.VELCRO:
            # VELCRO: Session-based persistence
            # Only store temporarily for the current session
            logger.info(
                f"Stored result from {tool_name} in team {self.name} with VELCRO adhesive (session only)"
            )
        else:
            # TAPE: One-time use, no persistence
            logger.info(
                f"Used result from {tool_name} in team {self.name} with TAPE adhesive (one-time use)"
            )

    async def process_message(
        self,
        content: Any,
        source_model: Optional[str] = None,
        target_model: Optional[str] = None,
        from_model: Optional[str] = None,
    ) -> str:
        """Process a message within the team"""
        # Handle backward compatibility
        if from_model is not None and source_model is None:
            source_model = from_model

        # Handle dict-like messages
        message_content = content
        if isinstance(content, dict) and "content" in content:
            message_content = content["content"]

        # Get source model
        source = None
        if source_model:
            source = self.models.get(source_model)
            if not source:
                raise ValueError(f"Model {source_model} not in team")

        # Get target model
        target = None
        if target_model:
            target = self.models.get(target_model)
            if not target:
                raise ValueError(f"Model {target_model} not in team")

        # Use lead model if no specific models given
        if not source and self.config.lead:
            source = self.models[self.config.lead]

        if not source:
            raise ValueError("No source model available")

        # Record the incoming user message in the team history
        self.conversation_history.append(
            Message(role="user", content=message_content)
        )

        # Generate response from model, handling sync (Resp) and async coroutines
        gen_result = source.generate(message_content)
        if asyncio.iscoroutine(gen_result):
            raw_response = await gen_result
        else:
            raw_response = gen_result
        # Extract string content if response has .content attribute (e.g., Openrouter Resp)
        if hasattr(raw_response, 'content'):
            response_content = raw_response.content
        else:
            response_content = raw_response

        # Debug logging to capture the output of the `generate` method
        logger.debug(f"Model generated response: {raw_response}")

        # Store the model's raw response in history first
        self.conversation_history.append(
            Message(
                role="assistant",  # Use 'assistant' role for model's response
                content=response_content,
            )
        )

        # Prepare for potential tool call: parse JSON if response is a string
        final_response = response_content  # Default to original response
        tool_call_data = None
        if isinstance(response_content, str):
            try:
                tool_call_data = json.loads(response_content)
            except json.JSONDecodeError:
                tool_call_data = None
        logger.debug(
            f"Attempted JSON extraction from response. Found: {tool_call_data is not None}"
        )

        if tool_call_data and isinstance(tool_call_data, dict):
            # Check for standard format: {"tool_name": "...", "arguments": {...}}
            if "tool_name" in tool_call_data and "arguments" in tool_call_data:
                tool_name = tool_call_data["tool_name"]
                arguments = tool_call_data["arguments"]
                tool_call_id = f"call_{uuid.uuid4()}"  # Generate an ID for the call
                logger.info(
                    f"Detected tool call via JSON: {tool_name} (ID: {tool_call_id}) with args: {arguments}"
                )

            # Check for alternative format: {"tool_name": {...}}
            elif len(tool_call_data) == 1:
                tool_name = next(iter(tool_call_data.keys()))
                arguments = tool_call_data[tool_name]
                tool_call_id = f"call_{uuid.uuid4()}"  # Generate an ID for the call

                # Verify this is a valid tool
                if tool_name in source.tools:
                    logger.info(
                        f"Detected alternative format tool call: {tool_name} (ID: {tool_call_id}) with args: {arguments}"
                    )
                else:
                    # Not a valid tool call, treat as regular response
                    return response_content
            else:
                # Not a valid tool call format, treat as regular response
                return response_content

            # Default tool lookup
            tool_instance = source.tools.get(tool_name)
            if tool_instance and (
                hasattr(tool_instance, "execute") or callable(tool_instance)
            ):  # Ensure tool is executable or callable
                try:
                    # Add calling context to arguments
                    arguments_with_context = arguments.copy()
                    arguments_with_context["calling_model"] = source.name
                    arguments_with_context["calling_team"] = self.name

                    # Debugging: Log the tool invocation
                    logger.debug(f"Invoking tool '{tool_name}' with arguments: {arguments_with_context}")

                    # Log the type and attributes of the tool instance
                    logger.debug(f"Tool instance type: {type(tool_instance)}, attributes: {dir(tool_instance)}")

                    # Determine how to call the tool: wrapper function or .execute method
                    if callable(tool_instance) and not hasattr(
                        tool_instance, "execute"
                    ):
                        tool_callable = tool_instance
                    else:
                        tool_callable = tool_instance.execute
                    logger.debug(f"Calling execute method of tool '{tool_name}' with arguments: {arguments_with_context}")
                    tool_result_content = await tool_callable(**arguments_with_context)
                    # If the tool returned an error payload, abort and return the error message
                    if (
                        isinstance(tool_result_content, dict)
                        and tool_result_content.get("success") is False
                    ):
                        error_msg = tool_result_content.get(
                            "error", "Unknown tool error."
                        )
                        logger.error(f"Tool '{tool_name}' reported error: {error_msg}")
                        # Log error in history
                        self.conversation_history.append(
                            Message(
                                role="tool",
                                name=tool_name,  # Use 'name' instead of 'tool_call_id'
                                content=error_msg,
                            )
                        )
                        return error_msg
                    logger.info(f"Executed tool '{tool_name}' successfully.")

                    # Record the tool invocation in history
                    self.conversation_history.append(
                        Message(
                            role="tool",
                            name=tool_name,
                            content=str(tool_result_content),
                        )
                    )

                    # Share the result with the team
                    tool_result = ToolResult(
                        tool_name=tool_name, result=tool_result_content
                    )
                    await self.share_result(tool_name, tool_result, model_name=source.name)

                    # Let the model process the tool result to produce a final answer
                    final_response = await source.process_tool_result(tool_result)

                    # Record the final response in history
                    self.conversation_history.append(
                        Message(
                            role="assistant",
                            content=final_response,
                        )
                    )

                    logger.info(
                        f"Tool-handled response from {source.name}: {final_response[:100]}..."
                    )
                    return final_response
                except Exception as e:
                    logger.error(
                        f"Error executing tool '{tool_name}': {e}", exc_info=True
                    )
                    # Log the error as a tool result message
                    error_message = f"Error executing tool '{tool_name}': {e}"
                    self.conversation_history.append(
                        Message(
                            role="tool",
                            name=tool_name,
                            content=error_message,
                        )
                    )
                    # Optionally, generate a response acknowledging the error
                    # final_response = await source.generate(f"I encountered an error trying to use the {tool_name} tool: {e}")
                    # For now, just return the error message as the final response for simplicity
                    final_response = error_message
                    # No need to append this response to history as the tool error message is already there

            else:
                logger.warning(
                    f"Tool '{tool_name}' requested by model {source.name} but not found or not executable."
                )
                # Log this attempt in history as a failed tool message
                fail_message = f"Tool '{tool_name}' not found or available."
                self.conversation_history.append(
                    Message(
                        role="tool",
                        name=tool_name,
                        content=fail_message,
                    )
                )
                # Generate a response indicating the tool is unavailable
                # final_response = await source.generate(f"I tried to use the {tool_name} tool, but it is not available to me.")
                # Return the failure message directly
                final_response = fail_message
                # No need to append this response to history as the tool message is already there

        # Return the final response (either original model response or response after tool execution/error)
        return final_response

    async def direct_communication(
        self, from_model: str, to_model: str, message: Any
    ) -> str:
        """Handle direct communication between two models within the team
        Ensures the message history reflects the direct exchange."""

        # Skip internal subtask assignment/completion messages to the lead to prevent unintended planning triggers
        if isinstance(message, dict) and (
            "task_assigned" in message or "task_completed" in message
        ):
            lead_name = getattr(self.config, "lead", None)
            if lead_name and to_model == lead_name:
                logger.debug(
                    f"direct_communication: suppressed subtask message to lead {to_model}: {message}"
                )
                return ""

        # Find source and target models
        source = self.models.get(from_model)
        target = self.models.get(to_model)
        if not source or not target:
            raise ValueError("Source or target model not found in team")
        # For non-lead recipients: record the notification, then invoke their generate() to initialize them and get a response
        lead_name = getattr(self.config, "lead", None)
        if lead_name and to_model != lead_name:
            # Prepare the content string
            content_str = message if isinstance(message, str) else json.dumps(message)
            # Record the incoming assignment or notification
            self.conversation_history.append(
                Message(role="assistant", name=from_model, content=content_str)
            )
            # Let the member generate a response (this triggers model initialization)
            raw_response = await target.generate(content_str)
            # Record the member's response
            self.conversation_history.append(
                Message(role="assistant", name=to_model, content=raw_response)
            )
            logger.info(
                f"Direct communication response from {to_model}: {raw_response[:100]}..."
            )
            return raw_response

        # For lead recipients, record the dict message and let the lead generate an acknowledgement
        if isinstance(message, dict):
            content_str = json.dumps(message)
            # Record incoming notification from member
            self.conversation_history.append(
                Message(role="assistant", name=from_model, content=content_str)
            )
            # Let the lead generate an acknowledgement or process
            raw_response = await target.generate(content_str)
            # Record lead's response
            self.conversation_history.append(
                Message(role="assistant", name=to_model, content=raw_response)
            )
            logger.info(
                f"Direct communication response from {to_model}: {raw_response[:100]}..."
            )
            return raw_response
        logger.info(f"Direct communication initiated: {from_model} -> {to_model}")

        # Prepare the message content
        message_content = message if isinstance(message, str) else json.dumps(message)        # --- Update history for the target model ---
        history_for_target = self.conversation_history.copy()
        # Treat incoming internal message as user input for the lead model
        user_message = Message(role="user", name=from_model, content=message_content)
        history_for_target.append(user_message)
        
        # Record the user message in the main conversation history as well
        self.conversation_history.append(user_message)

        # --- Target model generates a response, including tool calls ---
        # Use generate_response to enable tool call handling        # 1. Model generates an initial response, which may include a tool call
        raw_response = await target.generate_response(history_for_target)

        # 2. Check for an embedded tool call in the response
        tool_call_data = None
        if isinstance(raw_response, str):
            try:
                tool_call_data = json.loads(raw_response)
            except json.JSONDecodeError:
                tool_call_data = None

        if tool_call_data and isinstance(tool_call_data, dict):
            # Parse tool name and arguments
            if "tool_name" in tool_call_data and "arguments" in tool_call_data:
                tool_name = tool_call_data["tool_name"]
                arguments = tool_call_data["arguments"]
            elif len(tool_call_data) == 1:
                tool_name, arguments = next(iter(tool_call_data.items()))
            else:
                tool_name = None
                arguments = {}

            if tool_name:
                # Ensure the tool is registered on the model
                if tool_name not in target.tools and tool_name in self._tools:
                    try:
                        target.add_tool_sync(tool_name, self._tools[tool_name])
                        # After calling add_tool_sync, update the model's tools if it's a mock
                        # This handles test scenarios where add_tool_sync doesn't actually modify tools
                        if not hasattr(target.tools, 'get') or target.tools.get(tool_name) is None:
                            if hasattr(target.tools, '__setitem__'):  # It's a dict-like object
                                target.tools[tool_name] = self._tools[tool_name]
                    except Exception as e:
                        logger.warning(f"Failed to add tool {tool_name} to model {to_model}: {e}")
                
                # Look up the tool callable
                tool_inst = target.tools.get(tool_name)
                
                # Check if tool is available before attempting to use it
                if tool_inst is None:
                    error_msg = f"Tool '{tool_name}' not available on model '{to_model}'"
                    logger.error(error_msg)
                    self.conversation_history.append(
                        Message(role="tool", name=tool_name, content=error_msg)
                    )
                    return error_msg
                
                # Prepare arguments with calling context
                args_ctx = dict(arguments)
                args_ctx.update({"calling_model": to_model, "calling_team": self.name})
                # Choose how to call it
                if callable(tool_inst) and not hasattr(tool_inst, "execute"):
                    tool_func = tool_inst
                else:
                    tool_func = tool_inst.execute
                # Invoke the tool
                logger.debug(f"Calling execute method of tool '{tool_name}' with arguments: {args_ctx}")
                tool_result = await tool_func(**args_ctx)
                # Record the tool invocation in history
                self.conversation_history.append(
                    Message(role="tool", name=tool_name, content=str(tool_result))
                )
                # Share the result with the team
                tr = ToolResult(tool_name=tool_name, result=tool_result)
                await self.share_result(tool_name, tr, model_name=to_model)
                # Let the model process the tool result to produce a final answer
                final_response = await target.process_tool_result(tr)
                self.conversation_history.append(
                    Message(role="assistant", name=to_model, content=final_response)
                )
                logger.info(
                    f"Direct communication tool-handled response from {to_model}: {final_response[:100]}..."
                )
                return final_response
        
        # No tool call detected, log the raw assistant response
        self.conversation_history.append(
            Message(role="assistant", name=to_model, content=raw_response)
        )
        logger.info(
            f"Direct communication response from {to_model}: {raw_response[:100]}..."
        )
        return raw_response
    
    def add_member_sync(self, member, role: str = "member", tools: Optional[Set[str]] = None) -> None:
        """
        Add a model or subteam to the team synchronously.
        Supports hierarchical teams by allowing Team instances as members.
        """
        from .teams import Team  # Local import to avoid circular import
        if isinstance(member, Team):
            # Add as subteam
            if member.name in self.subteams:
                logger.warning(f"Subteam {member.name} already in team {self.name}")
                return
            self.subteams[member.name] = member
            logger.info(f"Added subteam {member.name} to team {self.name}")
            return
        # Otherwise, treat as Model
        if member.name in self.models:
            logger.warning(f"Model {member.name} already in team {self.name}")
            return
        
        self.models[member.name] = member
        member.team = self
        
        # Add all current tools to the new member, but only assign 'user_input' to the top model
        for tool_name, tool in self._tools.items():
            if tool_name == 'user_input':
                # Only assign to the top model
                top_model_name = None
                try:
                    from .hierarchy import get_highest_ranking_model
                    top_model_name = get_highest_ranking_model(self)
                except Exception:
                    pass
                if member.name != top_model_name:
                    continue
            if hasattr(member, "add_tool_sync") and callable(member.add_tool_sync):
                member.add_tool_sync(tool_name, tool)
        
        # Add specific tools if provided
        if tools:
            for tool_name in tools:
                if tool_name in self._tools and tool_name not in member.tools:
                    if hasattr(member, "add_tool_sync") and callable(member.add_tool_sync):
                        member.add_tool_sync(tool_name, self._tools[tool_name])
        
        if role == "lead":
            self.config.lead = member.name
            self.lead = member
        else:
            if member.name not in self.config.members:
                self.config.members.append(member.name)
        self.updated_at = datetime.now()
        logger.info(f"Added model {member.name} to team {self.name} with role {role}")
          # Update hierarchy attributes after adding member
        self.update_hierarchy_attributes()

    async def start_agent_loops(self, initial_input: Optional[str] = None) -> None:
        """Start core agent loops using GlueSmolTeam instead of legacy loops."""
        logger.info(f"Starting core agent loops for team {self.name}")
        # Orchestrate entire team via GlueSmolTeam
        if self.config.lead and initial_input:
            smol_team = GlueSmolTeam(
                team=self,
                model_clients=self.models,
                glue_config=None,
            )
            smol_team.setup()
            self.agent_loops[self.config.lead] = smol_team
            # Run in background to avoid blocking
            asyncio.create_task(asyncio.to_thread(smol_team.run, initial_input))
            logger.info(f"Started GlueSmolTeam for {self.config.lead}")

    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of agent loops

        Args:
            agent_id: Optional specific agent ID to get status for

        Returns:
            Status information
        """
        if agent_id:
            if agent_id in self.agent_loops:
                return self.agent_loops[agent_id].state
            return {"error": f"Agent {agent_id} not found"}
        # Get status of all agents
        return {
            "team": self.name,
            "agents": {aid: loop.state for aid, loop in self.agent_loops.items()},
        }

    async def terminate_agent_loops(self, reason: str = "requested") -> None:
        """Terminate all agent loops in the team

        Args:
            reason: Reason for termination
        """
        # Terminate all tracked loops
        for loop in self.agent_loops.values():
            try:
                loop.terminate(reason)
            except Exception:
                pass
        logger.info(f"Terminated agent loops for team {self.name}: {reason}")
        # Reset tracking
        self.agent_loops = {}

    # ==================== Magnetic Field Methods ====================
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
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not in team")

        model = self.models[model_name]

        # Handle different model implementations
        if hasattr(model, "tools"):
            return model.tools
        elif hasattr(model, "_tools"):
            return model._tools

        return {}

    async def try_establish_relationship(self, target_team: str) -> Dict[str, Any]:
        """Attempt to automatically establish a relationship with another team.

        Args:
            target_team: Name of the target team

        Returns:
            Dict with success status, relationship type if established, and error message if failed
        """
        result = {"success": False, "relationship_type": None, "error": None}

        # Check if we already have a relationship
        if target_team in self.relationships:
            result["success"] = True
            result["relationship_type"] = self.relationships[target_team]
            return result

        # Try to find a flow and establish relationship
        if hasattr(self, "outgoing_flows") and self.outgoing_flows:
            # First check outgoing flows
            for flow in self.outgoing_flows:
                if hasattr(flow, "target") and flow.target.name == target_team:
                    # Found outgoing flow, establish bidirectional relationship
                    self.relationships[target_team] = FlowType.BIDIRECTIONAL.value
                    result["success"] = True
                    result["relationship_type"] = FlowType.BIDIRECTIONAL.value
                    logger.info(
                        f"Automatically established BIDIRECTIONAL relationship with team {target_team} based on existing flow"
                    )
                    return result

        # If we have incoming flows, check those too
        if hasattr(self, "incoming_flows") and self.incoming_flows:
            for flow in self.incoming_flows:
                if hasattr(flow, "source") and flow.source.name == target_team:
                    # Found incoming flow, establish bidirectional relationship
                    self.relationships[target_team] = FlowType.BIDIRECTIONAL.value
                    result["success"] = True
                    result["relationship_type"] = FlowType.BIDIRECTIONAL.value
                    logger.info(
                        f"Automatically established BIDIRECTIONAL relationship with team {target_team} based on existing flow"
                    )
                    return result

        # No flows found, can't establish relationship
        result["error"] = f"No flows exist between team {self.name} and {target_team}"
        return result

    def get_shared_results(self) -> Dict[str, ToolResult]:
        """Get shared tool results"""
        return self.shared_results.copy()

    async def cleanup(self) -> None:
        """Clean up team resources"""
        # Clean up tools
        for tool_name, tool in self._tools.items():
            if hasattr(tool, "cleanup") and callable(tool.cleanup):
                await tool.cleanup()

        # Clear shared results
        self.shared_results.clear()

        # Clear conversation history
        self.conversation_history.clear()

        logger.info(f"Cleaned up team {self.name}")

    def inject_managed_agents(self):
        # Ensure the lead's interpreter is initialized
        if self.lead and not hasattr(self.lead, "interpreter"):
            try:
                self.lead.run("__interpreter_init__")
            except Exception:
                pass
        if self.lead and hasattr(self.lead, "interpreter") and hasattr(self.lead.interpreter, "globals"):
            for member_name, member_model in self.models.items():
                if member_name == self.lead.name:
                    continue
                def make_delegate_func(target_model):
                    def delegate_func(task):
                        return target_model.generate(task)
                    delegate_func.__name__ = target_model.name
                    return delegate_func
                self.lead.interpreter.globals[member_name] = make_delegate_func(member_model)

    async def setup(self) -> None:
        """Set up the team by initializing members, tools, and managed agent callables."""
        # (Assume any other setup logic here, e.g., initializing tools, etc.)
        # ... existing setup logic ...
        self.inject_managed_agents()
        logger.info(f"Team {self.name} setup complete")

    # ==================== Error Handling ====================
    async def _handle_error(self, error: Exception) -> None:
        """Handle team-level errors"""
        logger.error(f"Team error in {self.name}: {str(error)}")
        raise

    # ==================== Flow Management Methods ====================
    def register_outgoing_flow(self, flow: Any) -> None:
        """Register an outgoing flow with this team.

        Args:
            flow: Flow to register
        """
        if flow not in self.outgoing_flows:
            self.outgoing_flows.append(flow)
            logger.info(
                f"Registered outgoing flow from {self.name} to {flow.target.name}"
            )

    def register_incoming_flow(self, flow: Any) -> None:
        """Register an incoming flow with this team.

        Args:
            flow: Flow to register
        """
        if flow not in self.incoming_flows:
            self.incoming_flows.append(flow)
            logger.info(
                f"Registered incoming flow to {self.name} from {flow.source.name}"
            )

            # Task is now started in setup
            # # Start message processing if not already running
            # if self.processing_task is None:
            #     self.processing_task = asyncio.create_task(self._process_messages())

    def unregister_outgoing_flow(self, flow: Any) -> None:
        """Unregister an outgoing flow from this team.

        Args:
            flow: Flow to unregister
        """
        if flow in self.outgoing_flows:
            self.outgoing_flows.remove(flow)
            logger.info(
                f"Unregistered outgoing flow from {self.name} to {flow.target.name}"
            )

    def unregister_incoming_flow(self, flow: Any) -> None:
        """Unregister an incoming flow from this team.

        Args:
            flow: Flow to unregister
        """
        if flow in self.incoming_flows:
            self.incoming_flows.remove(flow)
            logger.info(
                f"Unregistered incoming flow to {self.name} from {flow.source.name}"
            )

    async def receive_message(self, message: Dict[str, Any], sender: Any) -> None:
        """Receive a message from another team.

        Args:
            message: Message to receive
            sender: Team that sent the message
        """
        logger.info(f"Team {self.name} received message from {sender.name}")

        # Check if this is a response to a previous message
        if message.get("metadata", {}).get("is_response", False):
            # This is a response to a previous message
            message.get("metadata", {}).get("source_model")
            target_model = message.get("metadata", {}).get("target_model")
            message_id = message.get("metadata", {}).get("message_id")

            # Check if we have pending responses for this message ID
            if (
                message_id
                and hasattr(self, "pending_responses")
                and message_id in self.pending_responses
            ):
                # Store the response content
                self.pending_responses[message_id] = message.get(
                    "content", "No content in response"
                )
                logger.info(f"Stored response for message ID {message_id}")
                return  # Skip adding to message queue since we've handled it

            # If we have response handlers, try to find a matching one (legacy support)
            if hasattr(self, "response_handlers") and self.response_handlers:
                # First try to find a handler by message ID
                if message_id and message_id in self.response_handlers:
                    try:
                        await self.response_handlers[message_id](message)
                        # Remove the handler after it's been called
                        del self.response_handlers[message_id]
                        logger.debug(
                            f"Called response handler for message ID {message_id}"
                        )
                        return  # Skip adding to message queue since we've handled it
                    except Exception as e:
                        logger.error(
                            f"Error calling response handler for message ID {message_id}: {e}"
                        )

                # If no message ID or handler not found, try to find by target model
                if target_model:
                    for key, handler in list(self.response_handlers.items()):
                        # Check if this response matches the handler
                        if target_model in key:
                            # Call the handler with the message
                            try:
                                await handler(message)
                                # Remove the handler after it's been called
                                del self.response_handlers[key]
                                logger.debug(f"Called response handler for {key}")
                                return  # Skip adding to message queue since we've handled it
                            except Exception as e:
                                logger.error(f"Error calling response handler: {e}")

        # Add to message queue for processing
        await self.message_queue.put((message, sender))

    async def _process_messages(self) -> None:
        """Process messages from the message queue."""
        logger.debug(f"_process_messages task started/running for team {self.name}")
        while True:
            logger.debug(f"Team {self.name} message queue polling...")
            try:
                # Get message from queue
                message, sender = await self.message_queue.get()
                # Route internal messages through direct_communication if they target a model
                metadata = (
                    message.get("metadata", {}) if isinstance(message, dict) else {}
                )
                internal = metadata.get("internal", False)
                target_model_name = metadata.get("target_model")
                source_model_name = metadata.get("source_model")
                if internal and target_model_name:
                    # Skip all internal messages targeted at models to prevent unintended LLM invocations
                    logger.debug(
                        f"_process_messages: suppressed internal message for model {target_model_name}: {message}"
                    )
                    self.message_queue.task_done()
                    continue

                sender_name = sender.name if sender else "internal"

                try:
                    # Process the message
                    logger.debug(
                        f"Processing message in team {self.name} from {sender_name}: {message}"
                    )

                    # Extract content from message
                    content = message.get("content", "")
                    if isinstance(content, dict) and "content" in content:
                        content = content["content"]

                    # Get metadata
                    metadata = message.get("metadata", {})
                    source_model_name = metadata.get("source_model")
                    target_model_name = metadata.get(
                        "target_model"
                    )  # Check if a specific model is targeted
                    is_internal = metadata.get("internal", False)
                    broadcast_id = metadata.get(
                        "broadcast_id"
                    )  # Get broadcast ID if present
                    is_internal_broadcast = (
                        is_internal
                        and target_model_name is None
                        and broadcast_id is not None
                    )

                    # --- Message Routing Logic ---
                    recipients = []
                    if is_internal_broadcast:
                        # Internal broadcast to all team members
                        logger.info(
                            f"Internal team message detected. Broadcasting to all members of team {self.name}."
                        )
                        recipients = list(self.models.values())
                    elif target_model_name and target_model_name in self.models:
                        # Message targeted at a specific model
                        logger.debug(
                            f"Message targeted at specific model: {target_model_name}"
                        )
                        # Route only to the specified model
                        recipients = [self.models[target_model_name]]
                    elif self.config.lead and self.config.lead in self.models:
                        # Default: External message or internal without specific target -> Route to lead
                        logger.debug(
                            f"Routing message from {sender_name} to lead model: {self.config.lead}"
                        )
                        recipients.append(self.models[self.config.lead])
                    else:
                        logger.warning(
                            f"No valid recipient (lead or target) found for message in team {self.name}. Discarding."
                        )
                        recipients = []  # No one to send to

                    # --- Process message for each recipient ---
                    broadcast_responses = []  # List to store responses for this broadcast
                    for recipient_model in recipients:
                        try:
                            # Bypass LLM for internal dict notifications (e.g. task_assigned/task_completed)
                            if metadata.get("internal") and isinstance(
                                message.get("content"), dict
                            ):
                                # Record the original notification
                                self.conversation_history.append(
                                    Message(
                                        role="system",
                                        content=json.dumps(message.get("content")),
                                    )
                                )
                                logger.debug(
                                    f"Skipped LLM generation for internal dict message to {recipient_model.name}"
                                )
                                continue

                            # Create a formatted message for the recipient model
                            model_message_content = f"Message from {source_model_name or sender_name}: {content}"
                            model_message = Message(
                                role="system",  # Use system role for inter-model/team messages
                                content=model_message_content,
                            )

                            logger.debug(
                                f"Generating response from recipient model {recipient_model.name}"
                            )
                            # Generate the model's response
                            raw_response = await recipient_model.generate_response(
                                [model_message]
                            )
                            # Invoke any JSON-encoded tool calls and refine the response via helper
                            response = await self._invoke_tool_and_refine_response(
                                recipient_model, raw_response
                            )

                            # Store interaction in conversation history (adjust roles as needed)
                            if is_internal_broadcast:
                                broadcast_responses.append(
                                    {
                                        "model": recipient_model.name,
                                        "response": response,
                                    }
                                )

                            # TODO: Handle response logic? Should the lead respond on behalf of the team?
                            # Should responses be aggregated? For now, individual models process internally.

                            # If the original sender was external and requires a response,
                            # maybe only the lead's response should be sent back?
                            # This part needs careful consideration based on desired team behavior.
                            if (
                                sender
                                and metadata.get("requires_response", False)
                                and recipient_model.name == self.config.lead
                            ):
                                logger.info(
                                    f"Lead model {self.config.lead} generated response to external sender {sender_name}"
                                )
                                # Simplified response sending - assumes lead response is the one to send back
                                response_message = {
                                    "content": response,
                                    "metadata": {
                                        "source_team": self.name,
                                        "target_team": sender.name,  # Use the original sender's name
                                        "source_model": self.config.lead,
                                        "is_response": True,
                                        "message_id": metadata.get("message_id"),
                                        "target_model": metadata.get("source_model"),
                                    },
                                }
                                # Find and use the correct flow to send back
                                sent_back = False
                                for flow in self.outgoing_flows:
                                    if (
                                        hasattr(flow, "target")
                                        and flow.target == sender
                                    ):
                                        await flow.send_from_source(response_message)
                                        sent_back = True
                                        break
                                if not sent_back:
                                    for flow in self.incoming_flows:
                                        if (
                                            hasattr(flow, "source")
                                            and flow.source == sender
                                        ):
                                            await flow.send_from_target(
                                                response_message
                                            )
                                            sent_back = True
                                            break
                                if not sent_back:
                                    logger.warning(
                                        f"Could not find flow to send response back to sender {sender_name}"
                                    )

                        except Exception as model_e:
                            logger.error(
                                f"Error processing message for recipient model {recipient_model.name} in team {self.name}: {model_e}"
                            )

                    # --- Finalize Broadcast Processing (if applicable) ---
                    if is_internal_broadcast:
                        logger.debug(
                            f"Finished processing broadcast {broadcast_id}. Aggregated {len(broadcast_responses)} responses."
                        )
                        if (
                            hasattr(self, "pending_broadcasts")
                            and broadcast_id in self.pending_broadcasts
                        ):
                            future = self.pending_broadcasts.pop(
                                broadcast_id
                            )  # Get and remove future
                            if not future.done():
                                future.set_result(broadcast_responses)
                                logger.info(
                                    f"Set result for broadcast future {broadcast_id}"
                                )
                            else:
                                logger.warning(
                                    f"Future for broadcast {broadcast_id} was already done."
                                )
                        else:
                            logger.warning(
                                f"No pending future found for broadcast ID {broadcast_id}. Cannot set result."
                            )

                except Exception as e:
                    logger.error(
                        f"Error in message processing loop for team {self.name}: {e}"
                    )

                finally:
                    # Mark task as done
                    self.message_queue.task_done()

            except asyncio.CancelledError:
                # Handle task cancellation
                logger.debug(f"Message processing task cancelled for team {self.name}")
                break

            except Exception as e:
                # Log any errors but keep processing
                logger.error(
                    f"Error in message processing loop for team {self.name}: {e}"
                )

    async def send_information(
        self, target_team: str, content: Any
    ) -> Union[bool, Dict[str, Any]]:
        """Send information to another team.

        Args:
            target_team: Name of the target team
            content: Content to send

        Returns:
            True if the information was sent successfully,
            or dict with error information if failed
        """
        logger.info(
            f"Team {self.name} attempting to send information to team {target_team}"
        )

        # Handle intra-team communication directly
        if target_team == self.name:
            logger.debug(f"Handling intra-team communication within {self.name}")
            # Create message similar to inter-team format
            message = {
                "content": content,
                "metadata": {
                    "source_team": self.name,
                    "target_team": self.name,
                    # Determine source model if possible (e.g., from content metadata or default to lead)
                    "source_model": (
                        content.get("metadata", {}).get("source_model")
                        if isinstance(content, dict)
                        else self.config.lead
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "internal": True,  # Flag for internal message
                },
            }
            # Merge metadata if provided in content
            if isinstance(content, dict) and "metadata" in content:
                message["metadata"].update(content["metadata"])
                if "content" in content:  # Use content's content field if available
                    message["content"] = content["content"]

            # Put the message into the team's own queue for processing
            try:
                await self.message_queue.put(
                    (message, None)
                )  # Use None for sender as it's internal
                logger.info(f"Queued internal message within team {self.name}")
                return True
            except Exception as e:
                error_msg = f"Error queueing internal message in team {self.name}: {e}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

        # --- Inter-Team Communication Logic ---
        # Check if we have a relationships attribute
        if not hasattr(self, "relationships"):
            error_msg = f"Team {self.name} has no relationships attribute configured"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "suggestion": "Make sure the team is properly initialized with relationships support",
            }

        logger.debug(f"Team {self.name} has relationships: {self.relationships}")

        # Try to establish a relationship if one doesn't exist
        if target_team not in self.relationships:
            logger.warning(f"No relationship with team {target_team}")

            # Try to establish a relationship automatically
            relationship_result = await self.try_establish_relationship(target_team)

            if not relationship_result["success"]:
                error_msg = f"No relationship with team {target_team} and could not establish one automatically: {relationship_result['error']}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "suggestion": f"Add a flow between {self.name} and {target_team} in your GLUE configuration",
                }

        # Check if the relationship allows sending
        relationship = self.relationships.get(target_team)
        if relationship not in [FlowType.PUSH.value, FlowType.BIDIRECTIONAL.value]:
            error_msg = f"Relationship {relationship} with team {target_team} doesn't allow sending"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "suggestion": "Change the relationship type to PUSH or BIDIRECTIONAL in your GLUE configuration",
            }

        # Find the appropriate flow
        if not hasattr(self, "outgoing_flows"):
            error_msg = f"Team {self.name} has no outgoing_flows attribute"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "suggestion": "Make sure the team is properly initialized with flow support",
            }

        logger.debug(
            f"Team {self.name} has outgoing flows to: {[flow.target.name for flow in self.outgoing_flows if hasattr(flow, 'target')]}"
        )

        for flow in self.outgoing_flows:
            if hasattr(flow, "target") and flow.target.name == target_team:
                # Create message
                message = {
                    "content": content,
                    "metadata": {
                        "source_team": self.name,
                        "target_team": target_team,
                        "source_model": self.config.lead if self.config.lead else None,
                        "timestamp": datetime.now().isoformat(),
                    },
                }

                # If content is a dict with metadata, merge it
                if isinstance(content, dict) and "metadata" in content:
                    message["metadata"].update(content["metadata"])
                    if "content" in content:
                        message["content"] = content["content"]

                # Send message
                try:
                    logger.debug(f"Sending message via flow: {message}")
                    await flow.send_from_source(message)
                    logger.info(
                        f"Sent information from team {self.name} to {target_team}"
                    )
                    return True
                except Exception as e:
                    error_msg = f"Error sending message from team {self.name} to {target_team}: {e}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}

        error_msg = f"No outgoing flow found from team {self.name} to {target_team}"
        logger.warning(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "suggestion": f"Define a flow from {self.name} to {target_team} in your GLUE configuration",
        }

    async def receive_information(
        self, source_team: str, content: Any
    ) -> Union[bool, Dict[str, Any]]:
        """Receive information from another team.

        Args:
            source_team: Name of the source team
            content: Content to receive

        Returns:
            True if the information was received successfully,
            or dict with error information if failed
        """
        # Check if we have a relationship with the source team
        if source_team not in self.relationships:
            error_msg = f"No relationship with team {source_team}"
            logger.warning(error_msg)

            # Try to establish a relationship automatically
            relationship_result = await self.try_establish_relationship(source_team)

            if not relationship_result["success"]:
                error_msg = f"No relationship with team {source_team} and could not establish one automatically: {relationship_result['error']}"
                return {
                    "success": False,
                    "error": error_msg,
                    "suggestion": f"Add a flow between {self.name} and {source_team} in your GLUE configuration",
                }

        # Check if the relationship allows receiving
        relationship = self.relationships[source_team]
        if relationship not in [FlowType.PULL.value, FlowType.BIDIRECTIONAL.value]:
            error_msg = f"Relationship {relationship} with team {source_team} doesn't allow receiving"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "suggestion": "Change the relationship type to PULL or BIDIRECTIONAL in your GLUE configuration",
            }

        # Create message
        message = {
            "content": content,
            "metadata": {
                "source_team": source_team,
                "target_team": self.name,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Process the message
        try:
            await self.message_queue.put((message, None))
            logger.info(f"Received information in team {self.name} from {source_team}")
            return True
        except Exception as e:
            error_msg = f"Error processing received message from {source_team}: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    async def assign_user_input_tool_to_hierarchy_top(self, tool: Any) -> bool:
        """
        Assign user input tool exclusively to the highest-ranking model in the team hierarchy.
        
        Args:
            tool: UserInputTool instance to assign
            
        Returns:
            True if assignment was successful, False otherwise
        """
        logger.debug("Starting assign_user_input_tool_to_hierarchy_top")
        logger.debug(f"Team config: {self.config}")
        logger.debug(f"Team models: {self.models}")

        try:
            if not hasattr(self, 'config') or not hasattr(self, 'models'):
                logger.error(f"Team {self.name} is missing required attributes (config or models) for tool assignment")
                return False

            if not self.config or not isinstance(self.models, dict) or not self.models:
                logger.error(f"Team {self.name} has improperly initialized config or models")
                return False

            top_model_name = get_highest_ranking_model(self)
            if not top_model_name:
                logger.error(f"No hierarchy top model found in team {self.name} for user input tool assignment")
                return False

            if top_model_name not in self.models:
                logger.error(f"Hierarchy top model {top_model_name} not found in team models")
                return False

            # Add tool to team registry
            self._tools['user_input'] = tool
            
            # Initialize tool if needed
            if (
                hasattr(tool, "initialize")
                and callable(tool.initialize)
                and hasattr(tool, "_initialized")
                and not tool._initialized
            ):
                try:
                    await tool.initialize()
                    logger.info(f"Initialized user input tool for team {self.name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize user input tool: {e}")
            
            # Assign tool only to the hierarchy top model
            top_model = self.models[top_model_name]
            await top_model.add_tool('user_input', tool)
            
            # Update hierarchy attributes to reflect user input access
            self.update_hierarchy_attributes()
            
            logger.info(f"Successfully assigned user input tool to hierarchy top model: {top_model_name} in team {self.name}")
            return True
            
        except HierarchyDetectionError as e:
            logger.error(f"Failed to assign user input tool due to hierarchy detection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in user input tool assignment: {e}")
            return False

    def update_hierarchy_attributes(self) -> None:
        """Update hierarchy-related attributes on all models in the team."""
        try:
            set_hierarchy_attributes(self)
            logger.debug(f"Updated hierarchy attributes for team {self.name}")
        except Exception as e:
            logger.warning(f"Failed to update hierarchy attributes for team {self.name}: {e}")

