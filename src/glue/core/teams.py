# glue/core/team.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List, Union
from datetime import datetime
import asyncio
import logging
from pydantic import BaseModel

from .types import AdhesiveType, TeamConfig, ToolResult, FlowType
from .schemas import Message

# Import Flow class conditionally to avoid circular imports
try:
    from .flow import Flow
except ImportError:
    Flow = Any  # Type hint for Flow

from .model import Model
from .agent_loop import AgentLoop, TeamLoopCoordinator, AgentState

# ==================== Constants ====================
logger = logging.getLogger("glue.team")

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
        members: Optional[List[Model]] = None
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
        self.repelled_by: Set[str] = set()      # Teams that repel this one
        
        # Flow management
        self.incoming_flows: List[Any] = []
        self.outgoing_flows: List[Any] = []
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.response_handlers = {}

        # Agent loop management
        self.agent_loops: Dict[str, AgentLoop] = {}
        self.loop_coordinator: Optional[TeamLoopCoordinator] = None
        
        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Handle backward compatibility with tests
        if lead is not None:
            self.add_member_sync(lead, role="lead")
            
        if members is not None:
            for member in members:
                self.add_member_sync(member)

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
        self,
        model: Model,
        role: str = "member",
        tools: Optional[Set[str]] = None
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
            if hasattr(model, 'add_tool_sync') and callable(model.add_tool_sync):
                model.add_tool_sync(tool_name, tool)
            
        # Add specific tools if provided
        if tools:
            for tool_name in tools:
                if tool_name in self._tools and tool_name not in model.tools:
                    if hasattr(model, 'add_tool_sync') and callable(model.add_tool_sync):
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
        if hasattr(tool, "initialize") and callable(tool.initialize) and hasattr(tool, "_initialized"):
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
        self,
        tool_name: str,
        result: ToolResult,
        model_name: Optional[str] = None
    ) -> None:
        """Share a tool result with the team
        
        Args:
            tool_name: Name of the tool that generated the result
            result: The tool result to share
            model_name: Optional name of the model that used the tool
        """
        # If a specific model is provided, use its adhesives
        if model_name and model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'adhesives') and model.adhesives:
                # Use the first adhesive defined in the model
                adhesive = next(iter(model.adhesives))
                result.adhesive = adhesive
                logger.debug(f"Using adhesive {adhesive} from model {model_name}")
        
        # Process based on adhesive type
        if result.adhesive == AdhesiveType.GLUE:
            # GLUE: Team-wide persistent results
            self.shared_results[tool_name] = result
            logger.info(f"Shared result from {tool_name} in team {self.name} with GLUE adhesive")
        elif result.adhesive == AdhesiveType.VELCRO:
            # VELCRO: Session-based persistence
            # Only store temporarily for the current session
            logger.info(f"Stored result from {tool_name} in team {self.name} with VELCRO adhesive (session only)")
        else:
            # TAPE: One-time use, no persistence
            logger.info(f"Used result from {tool_name} in team {self.name} with TAPE adhesive (one-time use)")

    async def process_message(
        self,
        content: Any,
        source_model: Optional[str] = None,
        target_model: Optional[str] = None,
        from_model: Optional[str] = None
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
        response_content = await source.generate(message_content)

        # Store the model's raw response in history first
        self.conversation_history.append(Message(
            role="assistant", # Use 'assistant' role for model's response
            content=response_content
        ))

        # --- BEGIN SIMULATED TOOL CALL HANDLING ---
        simulated_tool_executed = False
        final_response = response_content # Default to original response

        if isinstance(response_content, str):
            # Attempt to extract and parse a JSON block for simulated tool call
            import json
            import re # Keep re for potential future fixes if needed

            # Check if the entire response string looks like a JSON object
            if response_content.strip().startswith("{") and response_content.strip().endswith("}"):
                json_string = response_content.strip()
                logger.debug(f"Detected potential JSON tool call string: {json_string}")
                try:
                    processed_json_string = json_string # Keep original for logging
                    try:
                        # Attempt to fix missing comma between top-level keys like "key": value\n "next_key": ...
                        # More robust: looks for closing quote/bracket/brace, whitespace/newline, opening quote
                        processed_json_string = re.sub(r'(["\}\]]\s*\n\s*)(")', r'\1,\n\2', json_string)
                        if processed_json_string != json_string:
                             logger.info(f"Attempting parsing with fixed JSON (added comma): {processed_json_string}")

                        tool_call_data = json.loads(processed_json_string)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed even after attempting fixes: {e}. Processed JSON attempt: {processed_json_string[:200]}...")
                        raise # Re-raise to fall through to printing original response

                    if isinstance(tool_call_data, dict) and "tool_name" in tool_call_data and "arguments" in tool_call_data:
                        tool_name = tool_call_data["tool_name"]
                        arguments = tool_call_data["arguments"]
                        logger.info(f"Detected SIMULATED tool call via JSON: {tool_name} with args: {arguments}")

                        # Execute the local tool
                        if tool_name in self._tools:
                            tool = self._tools[tool_name]
                            tool_result_content = None
                            tool_error = False
                            try:
                                # --- DEBUGGING ---
                                logger.debug(f"Checking tool object for execution: name='{tool_name}', object={tool}, type={type(tool)}")
                                # --- END DEBUGGING ---
                                if hasattr(tool, "execute") and callable(tool.execute):
                                    # Set context if possible
                                    if hasattr(tool, "set_context") and callable(tool.set_context):
                                        tool.set_context({"model": source, "team": self, "message": message_content})
                                    
                                    # Execute
                                    tool_result_content = await tool.execute(**arguments)
                                    logger.info(f"Simulated tool {tool_name} executed successfully.")
                                else:
                                    tool_result_content = {"error": f"Tool {tool_name} has no execute method"}
                                    tool_error = True
                                    logger.error(tool_result_content["error"])
                            except Exception as e:
                                tool_result_content = {"error": f"Error executing tool {tool_name}: {str(e)}"}
                                tool_error = True
                                logger.error(tool_result_content["error"], exc_info=True)

                            # Create ToolResult message
                            tool_result_msg = Message(
                                role="tool",
                                content=json.dumps(tool_result_content) if isinstance(tool_result_content, dict) else str(tool_result_content),
                                metadata={"tool_name": tool_name, "is_error": tool_error}
                            )
                            self.conversation_history.append(tool_result_msg)

                            # Generate follow-up response from the model including the tool result
                            logger.info(f"Generating follow-up response after simulated tool call {tool_name}")
                            # We need to send the *entire* history up to this point
                            final_response = await source.generate_response(self.conversation_history)

                            # Add the final response to history
                            self.conversation_history.append(Message(role="assistant", content=final_response))
                            simulated_tool_executed = True

                        else:
                            logger.warning(f"Simulated tool call for unknown tool: {tool_name}")
                            # Optionally, inform the model the tool wasn't found
                            error_msg = Message(role="tool", content=json.dumps({"error": f"Tool '{tool_name}' not found."}), metadata={"tool_name": tool_name, "is_error": True})
                            self.conversation_history.append(error_msg)
                            final_response = await source.generate_response(self.conversation_history)
                            self.conversation_history.append(Message(role="assistant", content=final_response))
                            simulated_tool_executed = True # Mark as handled even if tool not found
                except json.JSONDecodeError:
                     # Not a valid JSON tool call, treat as regular text response
                     logger.debug(f"Response content is not a valid JSON object: {json_string[:100]}...") # Log beginning of string
                except Exception as e:
                     logger.error(f"Error processing potential simulated tool call JSON: {e}", exc_info=True)
            else:
                 # No JSON block found in the response
                 logger.debug("No JSON block found for simulated tool call.")

        # --- END SIMULATED TOOL CALL HANDLING ---


        # --- BEGIN NATIVE TOOL CALL HANDLING ---
        # Only process native tool calls if a simulated one wasn't executed
        if not simulated_tool_executed and isinstance(response_content, dict) and "tool_calls" in response_content:
            logger.info(f"Detected NATIVE tool calls in response from model {source.name}")
            # Use response_content here
            tool_calls = response_content.get("tool_calls", []) 
            
            # Process each tool call
            tool_results = []
            for tool_call in tool_calls:
                try:
                    # Extract tool information
                    function_info = tool_call.get("function", {})
                    tool_name = function_info.get("name", "")
                    arguments = function_info.get("arguments", {})
                    
                    logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
                    
                    # Check if the tool exists
                    if tool_name in self._tools:
                        tool = self._tools[tool_name]
                        
                        # Try to execute the tool
                        try:
                            if hasattr(tool, "execute") and callable(tool.execute):
                                # Convert arguments to the expected format if needed
                                if isinstance(arguments, str):
                                    try:
                                        import json
                                        arguments = json.loads(arguments)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse arguments as JSON: {arguments}")
                                
                                # Set context for the tool if it supports it
                                if hasattr(tool, "set_context") and callable(tool.set_context):
                                    tool.set_context({
                                        "model": source,
                                        "team": self,
                                        "message": message_content
                                    })
                                
                                # Execute the tool
                                result = await tool.execute(**arguments)
                                
                                # Create a tool result
                                adhesive_type = next(iter(source.adhesives)) if hasattr(source, 'adhesives') and source.adhesives else AdhesiveType.TAPE
                                tool_result = ToolResult(
                                    tool_name=tool_name,
                                    result=result,
                                    adhesive=adhesive_type
                                )
                                
                                # Share the result with the team
                                await self.share_result(tool_name, tool_result, source.name)
                                
                                # Add to results
                                tool_results.append(tool_result)
                                
                                logger.info(f"Tool {tool_name} executed successfully")
                            else:
                                error_msg = f"Tool {tool_name} does not have an execute method"
                                logger.error(error_msg)
                                tool_results.append(ToolResult(
                                    tool_name=tool_name,
                                    result={"error": error_msg},
                                    adhesive=AdhesiveType.TAPE,
                                    is_error=True
                                ))
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            logger.error(error_msg)
                            tool_results.append(ToolResult(
                                tool_name=tool_name,
                                result={"error": error_msg},
                                adhesive=AdhesiveType.TAPE,
                                is_error=True
                            ))
                    else:
                        # Handle case where tool doesn't exist - create a new tool if possible
                        logger.info(f"Tool {tool_name} not found, attempting to create it")
                        
                        # Check if there's a tool creation capability
                        if "create_tool" in self._tools:
                            try:
                                # Create a new tool configuration
                                tool_config = {
                                    "name": tool_name,
                                    "description": f"Dynamically created tool: {tool_name}",
                                    "parameters": arguments.get("parameters", {})
                                }
                                
                                # Use the create_tool tool to create a new tool
                                create_tool = self._tools["create_tool"]
                                new_tool_result = await create_tool.execute(config=tool_config)
                                
                                # Add the new tool to the team
                                if new_tool_result and "tool" in new_tool_result:
                                    new_tool = new_tool_result["tool"]
                                    await self.add_tool(tool_name, new_tool)
                                    
                                    # Now execute the newly created tool
                                    if callable(getattr(new_tool, "execute", None)):
                                        result = await new_tool.execute(**arguments)
                                        
                                        # Create a tool result
                                        adhesive_type = next(iter(source.adhesives)) if hasattr(source, 'adhesives') and source.adhesives else AdhesiveType.TAPE
                                        tool_result = ToolResult(
                                            tool_name=tool_name,
                                            result=result,
                                            adhesive=adhesive_type
                                        )
                                        
                                        # Share the result with the team
                                        await self.share_result(tool_name, tool_result, source.name)
                                        
                                        # Add to results
                                        tool_results.append(tool_result)
                                        
                                        logger.info(f"Newly created tool {tool_name} executed successfully")
                                else:
                                    error_msg = f"Failed to create tool {tool_name}"
                                    logger.error(error_msg)
                                    tool_results.append(ToolResult(
                                        tool_name=tool_name,
                                        result={"error": error_msg},
                                        adhesive=AdhesiveType.TAPE,
                                        is_error=True
                                    ))
                            except Exception as e:
                                error_msg = f"Error creating tool {tool_name}: {str(e)}"
                                logger.error(error_msg)
                                tool_results.append(ToolResult(
                                    tool_name=tool_name,
                                    result={"error": error_msg},
                                    adhesive=AdhesiveType.TAPE,
                                    is_error=True
                                ))
                        else:
                            error_msg = f"Tool {tool_name} not found and create_tool capability not available"
                            logger.error(error_msg)
                            tool_results.append(ToolResult(
                                tool_name=tool_name,
                                result={"error": error_msg},
                                adhesive=AdhesiveType.TAPE,
                                is_error=True
                            ))
                except Exception as e:
                    error_msg = f"Error executing tool {tool_call.get('function', {}).get('name', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    tool_results.append(ToolResult(
                        tool_name=tool_call.get("function", {}).get("name", "unknown"),
                        result={"error": error_msg},
                        adhesive=AdhesiveType.TAPE,
                        is_error=True
                    ))
            
            # Format tool results for the response
            tool_results_text = "\n\n".join([
                f"Tool: {result.tool_name}\n" +
                f"Result: {result.result}\n" +
                (f"Error: {result.result}" if result.is_error else "")
                for result in tool_results
            ])
            
            # Generate a new response with the tool results
            if tool_results:
                # Send tool results back to the model for further processing
                tool_results_message = f"I executed the following tools:\n\n{tool_results_text}"
                follow_up_response = await source.generate(tool_results_message)
                
                # Store the follow-up in history
                self.conversation_history.append(Message(
                    role="system",
                    content=tool_results_message
                ))
                
                self.conversation_history.append(Message(
                    role="model",
                    content=follow_up_response
                ))
                
                # Return the combined response
                return follow_up_response
        
        # Return the final response (which might be the original or the follow-up after tool execution)
        # The final_response variable holds the correct response to return,
        # whether it's the original model response or the follow-up after a tool call.
        # History is updated within the respective blocks.
        return final_response if isinstance(final_response, str) else str(final_response)

    async def direct_communication(
        self,
        from_model: str,
        to_model: str,
        message: Any
    ) -> str:
        """Direct communication between team members"""
        if from_model not in self.models:
            raise ValueError(f"Source model {from_model} not in team")
            
        if to_model not in self.models:
            raise ValueError(f"Target model {to_model} not in team")
            
        # Extract message content if dict
        message_content = message
        if isinstance(message, dict) and "content" in message:
            message_content = message["content"]
            
        # Generate response from target model
        target_model = self.models[to_model]
        response = await target_model.generate(message_content)
        
        # Add to conversation history
        history_message = Message(
            role="model",
            content=message_content,
            metadata={"from": from_model, "to": to_model}
        )
        self.conversation_history.append(history_message)
        
        return response

    def add_member_sync(
        self,
        model: Model,
        role: str = "member",
        tools: Optional[Set[str]] = None
    ) -> None:
        """Synchronous version of add_member for use in tests"""
        # Register the model with this team
        model.set_team(self)
        
        # Add to models dictionary
        self.models[model.name] = model
        
        # Assign tools if specified
        if tools:
            for tool_name in tools:
                if tool_name in self._tools:
                    model.add_tool(tool_name, self._tools[tool_name])
        
        # Set as lead if role is lead
        if role == "lead":
            self.config.lead = model.name
            self.lead = model
        
        logger.info(f"Added model {model.name} to team {self.name} with role {role}")

    async def create_agent_loops(self) -> None:
        """Create agent loops for all team members and set up the coordinator"""
        if not self.loop_coordinator:
            self.loop_coordinator = TeamLoopCoordinator(self.name)
            
        # Create agent loops for each model
        for agent_id, model in self.models.items():
            # Skip if agent loop already exists
            if agent_id in self.agent_loops:
                continue
                
            # Create new agent loop
            agent_loop = AgentLoop(agent_id, self.name, model)
            
            # Register tools
            for tool_name, tool_func in self._tools.items():
                agent_loop.register_tool(tool_name, tool_func)
                
            # Add to our tracking and the coordinator
            self.agent_loops[agent_id] = agent_loop
            self.loop_coordinator.add_agent(agent_loop)
            
            logger.info(f"Created agent loop for {agent_id} in team {self.name}")
            
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
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not in team")
            
        model = self.models[model_name]
        
        # Handle different model implementations
        if hasattr(model, 'tools'):
            return model.tools
        elif hasattr(model, '_tools'):
            return model._tools
            
        return {}

    def get_shared_results(self) -> Dict[str, ToolResult]:
        """Get shared tool results"""
        return self.shared_results

    async def cleanup(self) -> None:
        """Clean up team resources"""
        # Clean up tools
        for tool_name, tool in self._tools.items():
            if hasattr(tool, 'cleanup') and callable(tool.cleanup):
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
        # Add tools from config if they exist
        if hasattr(self.config, "tools") and self.config.tools:
            for tool_name in self.config.tools:
                # Tools will be added during app setup
                pass
                
        # Nothing else to do in the base implementation
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
            logger.info(f"Registered outgoing flow from {self.name} to {flow.target.name}")
            
    def register_incoming_flow(self, flow: Any) -> None:
        """Register an incoming flow with this team.
        
        Args:
            flow: Flow to register
        """
        if flow not in self.incoming_flows:
            self.incoming_flows.append(flow)
            logger.info(f"Registered incoming flow to {self.name} from {flow.source.name}")
            
            # Start message processing if not already running
            if self.processing_task is None:
                self.processing_task = asyncio.create_task(self._process_messages())
                
    def unregister_outgoing_flow(self, flow: Any) -> None:
        """Unregister an outgoing flow from this team.
        
        Args:
            flow: Flow to unregister
        """
        if flow in self.outgoing_flows:
            self.outgoing_flows.remove(flow)
            logger.info(f"Unregistered outgoing flow from {self.name} to {flow.target.name}")
            
    def unregister_incoming_flow(self, flow: Any) -> None:
        """Unregister an incoming flow from this team.
        
        Args:
            flow: Flow to unregister
        """
        if flow in self.incoming_flows:
            self.incoming_flows.remove(flow)
            logger.info(f"Unregistered incoming flow to {self.name} from {flow.source.name}")
            
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
            source_model = message.get("metadata", {}).get("source_model")
            target_model = message.get("metadata", {}).get("target_model")
            message_id = message.get("metadata", {}).get("message_id")
            
            # Check if we have pending responses for this message ID
            if message_id and hasattr(self, 'pending_responses') and message_id in self.pending_responses:
                # Store the response content
                self.pending_responses[message_id] = message.get("content", "No content in response")
                logger.info(f"Stored response for message ID {message_id}")
                return  # Skip adding to message queue since we've handled it
            
            # If we have response handlers, try to find a matching one (legacy support)
            if hasattr(self, 'response_handlers') and self.response_handlers:
                # First try to find a handler by message ID
                if message_id and message_id in self.response_handlers:
                    try:
                        await self.response_handlers[message_id](message)
                        # Remove the handler after it's been called
                        del self.response_handlers[message_id]
                        logger.debug(f"Called response handler for message ID {message_id}")
                        return  # Skip adding to message queue since we've handled it
                    except Exception as e:
                        logger.error(f"Error calling response handler for message ID {message_id}: {e}")
                
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
        while True:
            try:
                # Get message from queue
                message, sender = await self.message_queue.get()
                sender_name = sender.name if sender else "internal"
                
                try:
                    # Process the message
                    logger.debug(f"Processing message in team {self.name} from {sender_name}: {message}")
                    
                    # Extract content from message
                    content = message.get("content", "")
                    if isinstance(content, dict) and "content" in content:
                        content = content["content"]
                        
                    # Get metadata
                    metadata = message.get("metadata", {})
                    source_model_name = metadata.get("source_model")
                    target_model_name = metadata.get("target_model") # Check if a specific model is targeted
                    is_internal = metadata.get("internal", False)

                    # --- Message Routing Logic ---
                    recipients = []
                    if is_internal and target_model_name is None:
                        # Internal broadcast to all team members
                        logger.info(f"Internal team message detected. Broadcasting to all members of team {self.name}.")
                        recipients = list(self.models.values())
                    elif target_model_name and target_model_name in self.models:
                        # Message targeted at a specific model (internal or external)
                        logger.debug(f"Message targeted at specific model: {target_model_name}")
                        recipients.append(self.models[target_model_name])
                    elif self.config.lead and self.config.lead in self.models:
                        # Default: External message or internal without specific target -> Route to lead
                        logger.debug(f"Routing message from {sender_name} to lead model: {self.config.lead}")
                        recipients.append(self.models[self.config.lead])
                    else:
                        logger.warning(f"No valid recipient (lead or target) found for message in team {self.name}. Discarding.")
                        recipients = [] # No one to send to

                    # --- Process message for each recipient ---
                    for recipient_model in recipients:
                        try:
                            # Create a formatted message for the recipient model
                            model_message_content = f"Message from {source_model_name or sender_name}: {content}"
                            model_message = Message(
                                role="system", # Use system role for inter-model/team messages
                                content=model_message_content
                            )
                            
                            logger.debug(f"Generating response from recipient model {recipient_model.name}")
                            # Generate response (fire and forget for broadcasts, or handle response later)
                            # For simplicity now, we just let the model process it. Response handling might need refinement.
                            response = await recipient_model.generate_response([model_message]) 
                            
                            # Store interaction in conversation history (adjust roles as needed)
                            self.conversation_history.append(Message(
                                role="system", 
                                content=f"(To {recipient_model.name}) {model_message_content}"
                            ))
                            if response: # Log response if any
                                self.conversation_history.append(Message(
                                    role="assistant", 
                                    content=f"(From {recipient_model.name}) {response}"
                                ))

                            # TODO: Handle response logic? Should the lead respond on behalf of the team?
                            # Should responses be aggregated? For now, individual models process internally.
                            
                            # If the original sender was external and requires a response,
                            # maybe only the lead's response should be sent back?
                            # This part needs careful consideration based on desired team behavior.
                            if sender and metadata.get("requires_response", False) and recipient_model.name == self.config.lead:
                                 logger.info(f"Lead model {self.config.lead} generated response to external sender {sender_name}")
                                 # Simplified response sending - assumes lead response is the one to send back
                                 response_message = {
                                     "content": response,
                                     "metadata": {
                                         "source_team": self.name,
                                         "target_team": sender.name, # Use the original sender's name
                                         "source_model": self.config.lead,
                                         "is_response": True,
                                         "message_id": metadata.get("message_id"),
                                         "target_model": metadata.get("source_model")
                                     }
                                 }
                                 # Find and use the correct flow to send back
                                 sent_back = False
                                 for flow in self.outgoing_flows:
                                     if hasattr(flow, 'target') and flow.target == sender:
                                         await flow.send_from_source(response_message)
                                         sent_back = True
                                         break
                                 if not sent_back:
                                      for flow in self.incoming_flows:
                                         if hasattr(flow, 'source') and flow.source == sender:
                                             await flow.send_from_target(response_message)
                                             sent_back = True
                                             break
                                 if not sent_back:
                                     logger.warning(f"Could not find flow to send response back to sender {sender_name}")

                        except Exception as model_e:
                            logger.error(f"Error processing message for recipient model {recipient_model.name} in team {self.name}: {model_e}")

                except Exception as e:
                    logger.error(f"Error in message processing loop for team {self.name}: {e}")
                
                finally:
                    # Mark task as done
                    self.message_queue.task_done()
                    
            except asyncio.CancelledError:
                # Handle task cancellation
                logger.debug(f"Message processing task cancelled for team {self.name}")
                break
                
            except Exception as e:
                # Log any errors but keep processing
                logger.error(f"Error in message processing loop for team {self.name}: {e}")
                
    async def send_information(self, target_team: str, content: Any) -> bool:
        """Send information to another team.
        
        Args:
            target_team: Name of the target team
            content: Content to send
            
        Returns:
            True if the information was sent successfully, False otherwise
        """
        logger.info(f"Team {self.name} attempting to send information to team {target_team}")

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
                    "source_model": content.get("metadata", {}).get("source_model") if isinstance(content, dict) else self.config.lead,
                    "timestamp": datetime.now().isoformat(),
                    "internal": True # Flag for internal message
                }
            }
            # Merge metadata if provided in content
            if isinstance(content, dict) and "metadata" in content:
                message["metadata"].update(content["metadata"])
                if "content" in content: # Use content's content field if available
                    message["content"] = content["content"]

            # Put the message into the team's own queue for processing
            try:
                await self.message_queue.put((message, None)) # Use None for sender as it's internal
                logger.info(f"Queued internal message within team {self.name}")
                return True
            except Exception as e:
                logger.error(f"Error queueing internal message in team {self.name}: {e}")
                return False

        # --- Existing Inter-Team Communication Logic ---
        # Check if we have a relationship with the target team
        if not hasattr(self, 'relationships'):
            logger.warning(f"Team {self.name} has no relationships attribute")
            return False
            
        logger.debug(f"Team {self.name} has relationships: {self.relationships}")
        
        if target_team not in self.relationships:
            logger.warning(f"No relationship with team {target_team}")
            
            # If no relationship exists but we have outgoing flows, try to establish one
            if hasattr(self, 'outgoing_flows') and self.outgoing_flows:
                for flow in self.outgoing_flows:
                    if flow.target.name == target_team:
                        logger.info(f"Found flow to team {target_team}, establishing relationship")
                        self.relationships[target_team] = FlowType.BIDIRECTIONAL.value
                        break
            else:
                return False
            
        # Check if the relationship allows sending
        relationship = self.relationships.get(target_team)
        if relationship not in [FlowType.PUSH.value, FlowType.BIDIRECTIONAL.value]:
            logger.warning(f"Relationship {relationship} with team {target_team} doesn't allow sending")
            return False
            
        # Find the appropriate flow
        if not hasattr(self, 'outgoing_flows'):
            logger.warning(f"Team {self.name} has no outgoing_flows attribute")
            return False
            
        logger.debug(f"Team {self.name} has outgoing flows to: {[flow.target.name for flow in self.outgoing_flows if hasattr(flow, 'target')]}")
        
        for flow in self.outgoing_flows:
            if hasattr(flow, 'target') and flow.target.name == target_team:
                # Create message
                message = {
                    "content": content,
                    "metadata": {
                        "source_team": self.name,
                        "target_team": target_team,
                        "source_model": self.config.lead if self.config.lead else None,
                        "timestamp": datetime.now().isoformat()
                    }
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
                    logger.info(f"Sent information from team {self.name} to {target_team}")
                    return True
                except Exception as e:
                    logger.error(f"Error sending message from team {self.name} to {target_team}: {e}")
                    return False
                
        logger.warning(f"No outgoing flow found from team {self.name} to {target_team}")
        return False
        
    async def receive_information(self, source_team: str, content: Any) -> bool:
        """Receive information from another team.
        
        Args:
            source_team: Name of the source team
            content: Content to receive
            
        Returns:
            True if the information was received successfully, False otherwise
        """
        # Check if we have a relationship with the source team
        if source_team not in self.relationships:
            logger.warning(f"No relationship with team {source_team}")
            return False
            
        # Check if the relationship allows receiving
        relationship = self.relationships[source_team]
        if relationship not in [FlowType.PULL.value, FlowType.BIDIRECTIONAL.value]:
            logger.warning(f"Relationship {relationship} with team {source_team} doesn't allow receiving")
            return False
            
        # Create message
        message = {
            "content": content,
            "metadata": {
                "source_team": source_team,
                "target_team": self.name,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Process the message
        await self.message_queue.put((message, None))
        logger.info(f"Received information in team {self.name} from {source_team}")
        return True
        
    async def set_relationship(self, team_name: str, relationship: str) -> None:
        """Set relationship with another team.
        
        Args:
            team_name: Name of the team to set relationship with
            relationship: Type of relationship to set
        """
        if team_name in self.repelled_by:
            raise ValueError(f"Cannot set relationship with {team_name} - repelled")
            
        self.relationships[team_name] = relationship
        self.updated_at = datetime.now()
        logger.info(f"Set {relationship} relationship with team {team_name}")
        
    def add_member_sync(self, model: Model, role: str = "member", tools: Optional[Set[str]] = None) -> None:
        """Add a model to the team synchronously.
        
        This is a synchronous version of add_member for use during setup.
        
        Args:
            model: Model to add
            role: Role of the model in the team (lead or member)
            tools: Optional set of tool names to add to the model
        """
        if model.name in self.models:
            logger.warning(f"Model {model.name} already in team {self.name}")
            return
            
        # Add model
        self.models[model.name] = model
        model.team = self  # Set team reference
        
        # Set up tools
        if tools:
            for tool_name in tools:
                if tool_name in self._tools:
                    if hasattr(model, 'add_tool_sync') and callable(model.add_tool_sync):
                        model.add_tool_sync(tool_name, self._tools[tool_name])
                    
        # Update config
        if role == "lead":
            self.config.lead = model.name
            self.lead = model
        else:
            self.config.members.append(model.name)
            
        self.updated_at = datetime.now()
        logger.info(f"Added model {model.name} to team {self.name} with role {role}")
        
    async def initiate_communication(self, target_team: str, content: str, source_model: Optional[str] = None) -> bool:
        """Initiate communication with another team.
        
        This method allows a model to proactively communicate with another team.
        
        Args:
            target_team: Name of the target team
            content: Content to send
            source_model: Optional name of the source model
            
        Returns:
            True if communication was initiated successfully, False otherwise
        """
        # Use lead model if no source model specified
        if not source_model:
            source_model = self.config.lead
            
        if not source_model or source_model not in self.models:
            logger.warning(f"No valid source model for communication in team {self.name}")
            return False
            
        # Create message
        message = {
            "content": content,
            "metadata": {
                "source_team": self.name,
                "target_team": target_team,
                "source_model": source_model,
                "timestamp": datetime.now().isoformat(),
                "initiated": True
            }
        }
        
        # Send information to target team
        return await self.send_information(target_team, message)
