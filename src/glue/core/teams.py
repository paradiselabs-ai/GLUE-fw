# glue/core/team.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List, Union
from datetime import datetime
import asyncio
import logging
import uuid
import json

from .schemas import AdhesiveType, TeamConfig, ToolResult, FlowType
from .schemas import Message
from ..utils.json_utils import extract_json

from .model import Model

# ==================== Constants ====================
logger = logging.getLogger("glue.team")

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
        lead: Optional[Model] = None,
        members: Optional[List[Model]] = None,
    ):
        self.name = name
        self.config = config or TeamConfig(name=name, lead="", members=[], tools=[])

        # Core components
        self.models: Dict[str, Model] = {}
        self.lead: Optional[Model] = None
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
        self.updated_at = datetime.now()

        # Handle backward compatibility with tests
        if lead is not None:
            self.add_member_sync(lead, role="lead")

        if members is not None:
            for member in members:
                self.add_member_sync(member)

        # Note: Member models from config.members are expected to be added
        # by the app during setup, not here in the constructor

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
        return None

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
    @property
    def tools(self) -> List[Any]:
        """Get all tools available to this team.

        Returns:
            List of tools
        """
        # For test compatibility, we need to match the exact objects stored in app.tools
        # This is a bit of a hack, but it's necessary for the tests to pass
        return list(self._tools.values())

    # ==================== Core Methods ====================
    async def add_member(
        self, model: Model, role: str = "member", tools: Optional[Set[str]] = None
    ) -> None:
        """Add a model to the team"""
        if model.name in self.models:
            logger.warning(f"Model {model.name} already in team {self.name}")
            return

        # Add model
        self.models[model.name] = model
        model.team = self  # Set team reference

        # Set up tools - use add_tool_sync to avoid async issues
        # Add all team tools to the model
        for tool_name, tool in self._tools.items():
            if hasattr(model, "add_tool_sync") and callable(model.add_tool_sync):
                model.add_tool_sync(tool_name, tool)

        # Add specific tools if provided
        if tools:
            for tool_name in tools:
                if tool_name in self._tools and tool_name not in model.tools:
                    if hasattr(model, "add_tool_sync") and callable(
                        model.add_tool_sync
                    ):
                        model.add_tool_sync(tool_name, self._tools[tool_name])

        # Update config
        if role == "lead":
            self.config.lead = model.name
            self.lead = model
        else:
            if model.name not in self.config.members:
                self.config.members.append(model.name)

        self.updated_at = datetime.now()
        logger.info(f"Added model {model.name} to team {self.name} with role {role}")

    async def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to this team.

        Args:
            name: Tool name
            tool: Tool instance
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

        # Add tool to all models in the team
        for model in self.models.values():
            # Use await here since model.add_tool is async
            await model.add_tool(name, tool)

        logger.info(f"Added tool {name} to team {self.name}")

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

        # Generate response
        # --- Add history for the incoming message ---
        # This assumes process_message is typically called with user input or a simple string
        # We might need more context if it's called differently
        self.conversation_history.append(
            Message(
                role="user",  # Assuming incoming message is like user input
                content=message_content,
            )
        )

        response_content = await source.generate(message_content)

        # Store the model's raw response in history first
        self.conversation_history.append(
            Message(
                role="assistant",  # Use 'assistant' role for model's response
                content=response_content,
            )
        )

        # --- TOOL CALL HANDLING ---
        final_response = response_content  # Default to original response
        tool_call_data = None

        if isinstance(response_content, str):
            tool_call_data = extract_json(response_content)
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

            # Fallback: if the model hasn't had this tool added, wrap the team-level tool
            if tool_name not in source.tools and tool_name in self._tools:
                try:
                    source.add_tool_sync(tool_name, self._tools[tool_name])
                    logger.debug(
                        f"Added fallback wrapper for '{tool_name}' to model {source.name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add fallback wrapper for '{tool_name}' to model {source.name}: {e}"
                    )

            # Check if the tool exists in the source model's context
            tool_instance = source.tools.get(tool_name)
            if tool_instance and (
                hasattr(tool_instance, "execute") or callable(tool_instance)
            ):  # Ensure tool is executable or callable
                try:
                    # Parameter normalization
                    # This section normalizes parameter names by mapping common alternative names
                    # to their expected names. For example, it maps 'search_term' to 'query' for
                    # the web_search tool. Original parameters are removed after normalization
                    # to avoid duplication.
                    normalized_args = arguments.copy()
                    if tool_name in TOOL_PARAM_MAPPINGS:
                        for arg_name, arg_value in list(normalized_args.items()):
                            if arg_name in TOOL_PARAM_MAPPINGS[tool_name]:
                                # Map to the expected parameter name
                                expected_name = TOOL_PARAM_MAPPINGS[tool_name][arg_name]
                                # Skip if the mapping doesn't change the name
                                if expected_name != arg_name:
                                    normalized_args[expected_name] = arg_value
                                    # Remove the original parameter to avoid duplication
                                    del normalized_args[arg_name]
                                    logger.debug(
                                        f"Normalized parameter '{arg_name}' to '{expected_name}' for tool '{tool_name}'"
                                    )

                    # Add calling context to arguments
                    arguments_with_context = normalized_args.copy()
                    arguments_with_context["calling_model"] = source.name
                    arguments_with_context["calling_team"] = self.name

                    # Determine how to call the tool: wrapper function or .execute method
                    if callable(tool_instance) and not hasattr(
                        tool_instance, "execute"
                    ):
                        tool_callable = tool_instance
                    else:
                        tool_callable = tool_instance.execute
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
                                tool_call_id=tool_call_id,
                                name=tool_name,
                                content=error_msg,
                            )
                        )
                        return error_msg
                    logger.info(f"Executed tool '{tool_name}' successfully.")

                    # Create ToolResult object using backward-compatible fields
                    # The validator should handle converting this to the new format
                    tool_result_object = ToolResult(
                        tool_name=tool_name,
                        result=str(tool_result_content),  # Ensure content is string
                        # Omit adhesive; validator should handle default?
                        # Omit model_name as it caused previous TypeError
                    )

                    # Store the tool result in history
                    # Use the validated content from the object
                    self.conversation_history.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_id,  # Use the ID from the parsed JSON call
                            name=tool_name,
                            content=tool_result_object.result,  # Get content from the backward-compatible result field
                        )
                    )

                    # Share result (important for VELCRO/GLUE)
                    await self.share_result(
                        tool_name, tool_result_object, model_name=source.name
                    )

                    # Generate a final response *after* the tool execution
                    # The history now contains the original call and the result
                    logger.info("Generating final response after tool execution...")
                    final_response = await source.generate(content=None)

                    # Append this final assistant response to history
                    self.conversation_history.append(
                        Message(role="assistant", content=final_response)
                    )

                except Exception as e:
                    logger.error(
                        f"Error executing tool '{tool_name}': {e}", exc_info=True
                    )
                    # Log the error as a tool result message
                    error_message = f"Error executing tool '{tool_name}': {e}"
                    self.conversation_history.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_id,
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
                        tool_call_id=tool_call_id,
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
        message_content = message if isinstance(message, str) else json.dumps(message)

        # --- Update history for the target model ---
        history_for_target = self.conversation_history.copy()
        # Treat incoming internal message as user input for the lead model
        history_for_target.append(
            Message(role="user", name=from_model, content=message_content)
        )

        # --- Target model generates a response, including tool calls ---
        # Use generate_response to enable tool call handling
        # 1. Model generates an initial response, which may include a tool call
        raw_response = await target.generate_response(history_for_target)

        # 2. Check for an embedded tool call in the response
        tool_call_data = None
        if isinstance(raw_response, str):
            tool_call_data = extract_json(raw_response)

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
                    except Exception:
                        pass
                # Look up the tool callable
                tool_inst = target.tools.get(tool_name)
                # Prepare arguments with calling context
                args_ctx = dict(arguments)
                args_ctx.update({"calling_model": to_model, "calling_team": self.name})
                # Choose how to call it
                if callable(tool_inst) and not hasattr(tool_inst, "execute"):
                    tool_func = tool_inst
                else:
                    tool_func = tool_inst.execute
                # Invoke the tool
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

    def add_member_sync(
        self, model: Model, role: str = "member", tools: Optional[Set[str]] = None
    ) -> None:
        """Add a model to the team synchronously.

        This is a synchronous version of add_member for use during setup.

        Args:
            model: Model to add
            role: Role of the model in the team (lead or member)
            tools: Optional set of tool names to add to the model
        """
        # If model is already in team
        if model.name in self.models:
            # Don't log warning if it's already the lead and role is "lead"
            if role == "lead" and self.lead and self.lead.name == model.name:
                return
            # Otherwise log warning for duplicate attempts
            logger.warning(f"Model {model.name} already in team {self.name}")
            return

        # Add model
        self.models[model.name] = model
        model.team = self  # Set team reference

        # Set up tools
        if tools:
            for tool_name in tools:
                if tool_name in self._tools:
                    if hasattr(model, "add_tool_sync") and callable(
                        model.add_tool_sync
                    ):
                        model.add_tool_sync(tool_name, self._tools[tool_name])

        # Update config
        if role == "lead":
            self.config.lead = model.name
            self.lead = model
        else:
            self.config.members.append(model.name)

        self.updated_at = datetime.now()
        logger.info(f"Added model {model.name} to team {self.name} with role {role}")

    async def start_agent_loops(self, initial_input: Optional[str] = None) -> None:
        """Start core agent loops using simplified stubs for Team Lead and Team Member agents."""
        from .agent_loop import TeamMemberAgentLoop, TeamLeadAgentLoop

        logger.info(f"Starting core agent loops for team {self.name}")
        # Start Team Lead loop if a goal is provided
        if self.config.lead and initial_input:
            # Pass the execute coroutine of the delegate_task tool to the TeamLeadAgentLoop
            delegate_tool_exec = self._tools.get("delegate_task")
            if hasattr(delegate_tool_exec, "execute"):
                delegate_tool_exec = delegate_tool_exec.execute
            # Instantiate TeamLeadAgentLoop with team object and LLM
            agent_llm = self.models.get(self.config.lead)
            lead_loop = TeamLeadAgentLoop(
                team=self, delegate_tool=delegate_tool_exec, agent_llm=agent_llm
            )
            # Track and start TeamLead loop
            self.agent_loops[self.config.lead] = lead_loop
            asyncio.create_task(
                lead_loop.start(
                    parent_task_id=self.name, goal_description=initial_input
                )
            )
            logger.info(f"Started Team Lead loop for {self.config.lead}")
        # Start Team Member loops
        for member_id in self.config.members:
            member_loop = TeamMemberAgentLoop(
                member_id,
                self.name,
                self._tools.get("report_task_completion").execute,
                agent_llm=self.models.get(member_id),
            )
            # Register all available tools in the member loop
            for tool_name, tool in self._tools.items():
                tool_callable = getattr(tool, "execute", tool)
                member_loop.register_tool(tool_name, tool_callable)
            # Track and start TeamMember loop
            self.agent_loops[member_id] = member_loop
            asyncio.create_task(member_loop.start(self.fetch_task_for_member))
            logger.info(f"Started Team Member loop for {member_id}")

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
        return self.shared_results

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

    async def setup(self) -> None:
        """Set up the team by initializing any required resources.

        This method is called during application setup to initialize
        team resources, configure tools, and establish connections.
        """
        # Add any missing members from config
        if hasattr(self.config, "members") and self.config.members:
            for member_name in self.config.members:
                if member_name not in self.models and hasattr(self, "_app"):
                    # If we have a reference to the app, try to get the model
                    if hasattr(self._app, "models") and member_name in self._app.models:
                        logger.debug(
                            f"Adding missing member {member_name} to team {self.name} from config"
                        )
                        await self.add_member(self._app.models[member_name])

        # Add tools from config if they exist
        if hasattr(self.config, "tools") and self.config.tools:
            for tool_name in self.config.tools:
                # Tools will be added during app setup
                pass

        # Start the internal message processing loop unconditionally
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_messages())
            logger.debug(f"Started message processing task for team {self.name}")
        else:
            logger.debug(
                f"Message processing task already running for team {self.name}"
            )

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

    async def _invoke_tool_and_refine_response(
        self, model: Any, raw_response: str
    ) -> str:
        """
        Check a model's raw_response for JSON tool calls, execute if found,
        share result, and let the model process the tool result to produce final response.
        Otherwise return raw_response.
        """
        from ..utils.json_utils import extract_json

        # Only parse string responses
        if not isinstance(raw_response, str):
            return raw_response
        tool_call_data = extract_json(raw_response)
        if isinstance(tool_call_data, dict):
            # Parse tool name and arguments
            if "tool_name" in tool_call_data and "arguments" in tool_call_data:
                tool_name = tool_call_data["tool_name"]
                arguments = tool_call_data["arguments"]
            elif len(tool_call_data) == 1:
                tool_name, arguments = next(iter(tool_call_data.items()))
            else:
                return raw_response
            # Ensure tool on model
            if tool_name not in model.tools and tool_name in self._tools:
                try:
                    model.add_tool_sync(tool_name, self._tools[tool_name])
                except Exception:
                    pass
            tool_inst = model.tools.get(tool_name)
            # Prepare context for call
            ctx = dict(arguments)
            ctx.update({"calling_model": model.name, "calling_team": self.name})
            # Select callable
            if callable(tool_inst) and not hasattr(tool_inst, "execute"):
                tool_func = tool_inst
            else:
                tool_func = tool_inst.execute
            # Execute tool
            tool_result = await tool_func(**ctx)
            # Log and share result
            self.conversation_history.append(
                Message(role="tool", name=tool_name, content=str(tool_result))
            )
            tr = ToolResult(tool_name=tool_name, result=tool_result)
            await self.share_result(tool_name, tr, model_name=model.name)
            # Let model process tool result
            final = await model.process_tool_result(tr)
            self.conversation_history.append(
                Message(role="assistant", name=model.name, content=final)
            )
            logger.info(
                f"Tool {tool_name} applied for {model.name}, refined response generated."
            )
            return final
        return raw_response

    async def fetch_task_for_member(self, agent_id: str) -> dict:
        """Fetch next uncompleted task for the given team member."""
        while True:
            # Look for tasks assigned to this member without completion
            for task in list(self.shared_results.values()):
                if (
                    isinstance(task, dict)
                    and task.get("assigned_to") == agent_id
                    and not task.get("completion")
                ):
                    return task
            await asyncio.sleep(0.5)
