"""
Base model implementation for the GLUE framework.

This module provides a base model implementation that can be used with any provider.
It includes prompt engineering capabilities and standardized interfaces for all models.
"""

import os
import importlib
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Type
import asyncio
from enum import Enum

from .types import Message, ToolResult, AdhesiveType

# Set up logging
logger = logging.getLogger("glue.core.base_model")

class ModelProvider(str, Enum):
    """Enumeration of supported model providers."""
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    CUSTOM = "custom"

class BaseModel:
    """
    Base model implementation for the GLUE framework.
    
    This class provides a standardized interface for all models, regardless of provider.
    It includes prompt engineering capabilities and handles the conversion between
    different message formats.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize a new base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.name = self.config.get('name', 'unnamed_model')
        self.provider_name = self.config.get('provider', 'gemini')
        
        # Check for model name in nested config first, then at top level
        nested_config = self.config.get('config', {})
        self.model_name = nested_config.get('model') if nested_config and 'model' in nested_config else self.config.get('model', 'gpt-3.5-turbo')
        
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.api_key = self.config.get('api_key')
        self.api_params = self.config.get('api_params', {})
        self.description = self.config.get('description', '')
        self.provider_class = self.config.get('provider_class')
        
        # Provider instance
        self.provider = None
        self.client = None
        
        # Team reference
        self.team = None
        self.role = self.config.get('role', 'assistant')
        
        # Tools
        self.tools = {}
        
        # Development mode flag
        self.development = self.config.get('development', False)
        
        # Trace ID for tracking
        self._trace_id = str(uuid.uuid4())
        
        # Load the provider
        self._load_provider()
    
    def _load_provider(self):
        """Load the provider for this model."""
        provider_name = self.provider_name.lower()
        
        try:
            # First try to load from the providers directory
            provider_module_path = f"glue.core.providers.{provider_name}"
            
            try:
                # Try to import the provider module
                module = importlib.import_module(provider_module_path)
                
                # Get the provider class name
                provider_class_name = f"{provider_name.capitalize()}Provider"
                
                # Check if the provider class exists in the module
                if hasattr(module, provider_class_name):
                    # Create an instance of the provider
                    provider_class = getattr(module, provider_class_name)
                    self.provider = provider_class(self)
                    
                    # Set the client if available
                    if hasattr(self.provider, 'client'):
                        self.client = self.provider.client
                        
                    logger.info(f"Loaded provider {provider_name} for model {self.name}")
                else:
                    logger.warning(f"Provider module {provider_module_path} found, but {provider_class_name} class not found")
            except ImportError:
                logger.warning(f"Provider module {provider_module_path} not found")
            
            # If we get here and don't have a provider, try custom provider path if specified
            if self.provider is None and self.provider_class:
                try:
                    # Parse the provider class path
                    provider_path = self.provider_class
                    module_path, class_name = provider_path.rsplit('.', 1)
                    
                    # Import the module
                    module = importlib.import_module(module_path)
                    
                    # Get the provider class
                    provider_class = getattr(module, class_name)
                    
                    # Create an instance of the provider
                    self.provider = provider_class(self)
                    
                    # Set the client if available
                    if hasattr(self.provider, 'client'):
                        self.client = self.provider.client
                        
                    logger.info(f"Loaded custom provider {self.provider_class} for model {self.name}")
                except (ImportError, AttributeError) as e:
                    logger.error(f"Error loading custom provider {self.provider_class}: {e}")
            
            # If we still don't have a provider, raise an error
            if self.provider is None:
                raise ValueError(f"Could not load provider for {provider_name}")
                
            # Check if Portkey integration is enabled
            portkey_enabled = os.environ.get("PORTKEY_ENABLED", "").lower() in ("true", "1", "yes")
            
            if portkey_enabled and not self.development and self.provider is not None:
                try:
                    # Import the Portkey wrapper
                    from glue.core.providers.portkey_wrapper import wrap_provider
                    
                    # Wrap the provider with Portkey
                    self.provider = wrap_provider(self.provider, self.model_name, self._trace_id)
                    logger.info(f"Provider {provider_name} wrapped with Portkey (trace_id: {self._trace_id})")
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to wrap provider with Portkey: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading provider {provider_name}: {e}")
            raise
    
    def set_team(self, team):
        """Set the team this model belongs to."""
        self.team = team
    
    def add_tool_sync(self, name: str, tool: Any):
        """Add a tool to this model (synchronous version).
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        # Make sure the tool is callable
        if hasattr(tool, "execute") and callable(tool.execute):
            # Create a wrapper function that calls the tool's execute method
            async def tool_wrapper(**kwargs):
                # Initialize the tool if it's not already initialized
                if hasattr(tool, "_initialized") and not tool._initialized and hasattr(tool, "initialize"):
                    await tool.initialize()
                
                # Call the execute method
                if asyncio.iscoroutinefunction(tool.execute):
                    return await tool.execute(**kwargs)
                else:
                    return tool.execute(**kwargs)
            
            # Store the wrapper function
            self.tools[name] = tool_wrapper
            
            # Store the original tool object as an attribute of the wrapper
            tool_wrapper.tool_obj = tool
            
            # Copy important attributes from the tool to the wrapper
            if hasattr(tool, "description"):
                tool_wrapper.description = tool.description
            
            logger.info(f"Added callable tool wrapper for {name}")
        else:
            # Just store the tool as is (for backward compatibility)
            self.tools[name] = tool
            logger.warning(f"Added tool {name} without execute method")

    async def add_tool(self, name: str, tool: Any):
        """Add a tool to this model.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        # Just call the sync version
        self.add_tool_sync(name, tool)

    async def generate(self, content: str) -> str:
        """Generate a response from the model, with tool call detection and execution loop.
        
        Args:
            content: The content to generate a response for
            
        Returns:
            The generated response
        """
        # Initialize conversation history
        messages = [Message(role="user", content=content)]
        
        # Prepare tool schemas if available
        provider_tools = []
        if hasattr(self, "_provider_tools") and self._provider_tools:
            provider_tools = list(self._provider_tools.values())
        elif hasattr(self, "tools") and self.tools:
            # Fallback: format tools dict
            for tool_name, tool in self.tools.items():
                formatted = self._format_tool_for_provider(tool_name, tool)
                provider_tools.append(formatted)
        
        # Tool call + execution loop
        max_loops = 3  # prevent infinite loops
        for _ in range(max_loops):
            try:
                response = await self.generate_response(messages, tools=provider_tools)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return "I'm sorry, I encountered an error while generating a response."
            
            # If response is dict with tool calls (Gemini style)
            if isinstance(response, dict) and "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if not tool_calls:
                    # No tool calls, return empty content or fallback
                    return response.get("content", "")
                
                # Execute each tool call
                tool_results = []
                for call in tool_calls:
                    fn = call.get("function", {})
                    tool_name = fn.get("name")
                    arguments = fn.get("arguments", {})
                    # If arguments is stringified JSON, parse it
                    if isinstance(arguments, str):
                        try:
                            import json as _json
                            arguments = _json.loads(arguments)
                        except Exception:
                            # If it's not valid JSON, wrap it in a dictionary with a default key
                            arguments = {"query": arguments}
                    elif not isinstance(arguments, dict):
                        # If it's not a string or dict, convert to a dict with a default key
                        arguments = {"query": str(arguments)}
                    
                    # Find the tool
                    tool_obj = None
                    if hasattr(self, "tools") and self.tools:
                        tool_obj = self.tools.get(tool_name)
                    if not tool_obj:
                        logger.warning(f"Tool '{tool_name}' not found for call, skipping execution")
                        continue
                    
                    # Execute the tool
                    try:
                        if callable(tool_obj):
                            if asyncio.iscoroutinefunction(tool_obj):
                                result = await tool_obj(**arguments)
                            else:
                                result = tool_obj(**arguments)
                        elif hasattr(tool_obj, "execute") and callable(tool_obj.execute):
                            if asyncio.iscoroutinefunction(tool_obj.execute):
                                result = await tool_obj.execute(**arguments)
                            else:
                                result = tool_obj.execute(**arguments)
                        else:
                            logger.warning(f"Tool '{tool_name}' is not callable")
                            continue
                        # Append tool result message
                        tool_result_msg = Message(
                            role="function",
                            content=str(result),
                            name=tool_name
                        )
                        messages.append(tool_result_msg)
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_name}': {e}")
                        error_msg = Message(
                            role="function",
                            content=f"Error executing tool '{tool_name}': {e}",
                            name=tool_name
                        )
                        messages.append(error_msg)
                # Continue loop: re-call LLM with appended tool results
                continue
            
            # If response is string, check for ```tool_code``` blocks (Gemini style)
            if isinstance(response, str):
                import re
                import json
                
                # Check for ```tool_code``` blocks (Gemini style)
                tool_code_pattern = r"```tool_code\s*(.*?)\s*```"
                matches = re.findall(tool_code_pattern, response, re.DOTALL)
                if matches:
                    for code in matches:
                        # Parse tool_name(args)
                        m = re.match(r"(\w+)\s*\((.*)\)", code.strip())
                        if not m:
                            continue
                        tool_name = m.group(1)
                        args_str = m.group(2)
                        # Parse args into dict
                        args_dict = {}
                        try:
                            parts = [p.strip() for p in args_str.split(",") if "=" in p]
                            for p in parts:
                                k, v = p.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                args_dict[k] = v
                        except Exception:
                            pass
                        # Find tool
                        tool_obj = None
                        if hasattr(self, "tools") and self.tools:
                            tool_obj = self.tools.get(tool_name)
                        if not tool_obj:
                            logger.warning(f"Tool '{tool_name}' not found for tool_code, skipping execution")
                            continue
                        # Execute tool
                        try:
                            if callable(tool_obj):
                                if asyncio.iscoroutinefunction(tool_obj):
                                    result = await tool_obj(**args_dict)
                                else:
                                    result = tool_obj(**args_dict)
                            elif hasattr(tool_obj, "execute") and callable(tool_obj.execute):
                                if asyncio.iscoroutinefunction(tool_obj.execute):
                                    result = await tool_obj.execute(**args_dict)
                                else:
                                    result = tool_obj.execute(**args_dict)
                            else:
                                logger.warning(f"Tool '{tool_name}' is not callable")
                                continue
                            # Append tool result message
                            tool_result_msg = Message(
                                role="function",
                                content=str(result),
                                name=tool_name
                            )
                            messages.append(tool_result_msg)
                        except Exception as e:
                            logger.error(f"Error executing tool '{tool_name}': {e}")
                            error_msg = Message(
                                role="function",
                                content=f"Error executing tool '{tool_name}': {e}",
                                name=tool_name
                            )
                            messages.append(error_msg)
                    # Continue loop: re-call LLM with appended tool results
                    continue
                
                # Also check for ```tool_call``` blocks (for models without native tool support)
                tool_call_pattern = r"```tool_call\s*(.*?)\s*```"
                tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
                
                if tool_call_matches:
                    logger.debug(f"Found {len(tool_call_matches)} tool_call blocks in response")
                    
                    for tool_call_content in tool_call_matches:
                        # Parse the tool call content
                        tool_name = None
                        parameters = {}
                        
                        # Extract tool name and parameters
                        for line in tool_call_content.strip().split('\n'):
                            if line.startswith('tool_name:'):
                                tool_name = line.replace('tool_name:', '').strip()
                            elif line.startswith('parameters:'):
                                param_str = line.replace('parameters:', '').strip()
                                try:
                                    # Try to parse as JSON
                                    parameters = json.loads(param_str)
                                except json.JSONDecodeError:
                                    # If not valid JSON, try to parse as Python dict
                                    try:
                                        # Simple parsing for key-value pairs
                                        param_str = param_str.strip('{}')
                                        for pair in param_str.split(','):
                                            if ':' in pair:
                                                k, v = pair.split(':', 1)
                                                parameters[k.strip().strip('"').strip("'")] = v.strip().strip('"').strip("'")
                                    except Exception:
                                        logger.warning(f"Failed to parse parameters: {param_str}")
                        
                        if not tool_name:
                            logger.warning("Tool call block missing tool_name")
                            continue
                        
                        # Find the tool
                        tool_obj = None
                        if hasattr(self, "tools") and self.tools:
                            tool_obj = self.tools.get(tool_name)
                        if not tool_obj:
                            logger.warning(f"Tool '{tool_name}' not found for simulated tool call, skipping execution")
                            continue
                        
                        # Execute the tool
                        try:
                            if callable(tool_obj):
                                if asyncio.iscoroutinefunction(tool_obj):
                                    result = await tool_obj(**parameters)
                                else:
                                    result = tool_obj(**parameters)
                            elif hasattr(tool_obj, "execute") and callable(tool_obj.execute):
                                if asyncio.iscoroutinefunction(tool_obj.execute):
                                    result = await tool_obj.execute(**parameters)
                                else:
                                    result = tool_obj.execute(**parameters)
                            else:
                                logger.warning(f"Tool '{tool_name}' is not callable")
                                continue
                            
                            # Append tool result message
                            tool_result_msg = Message(
                                role="function",
                                content=str(result),
                                name=tool_name
                            )
                            messages.append(tool_result_msg)
                        except Exception as e:
                            logger.error(f"Error executing tool '{tool_name}': {e}")
                            error_msg = Message(
                                role="function",
                                content=f"Error executing tool '{tool_name}': {e}",
                                name=tool_name
                            )
                            messages.append(error_msg)
                    
                    # Continue the loop to process the tool results
                    continue
            
            # If response is plain string and no tool calls detected, return it
            if isinstance(response, str):
                return response
            # If response dict with 'content', return it
            if isinstance(response, dict) and "content" in response:
                return response["content"]
            # Else, return stringified response
            return str(response)
        
        # If max loops exceeded, return last response as string
        return str(response)
    
    def _format_messages_for_provider(self, messages: List[Message]) -> List[Message]:
        """Format messages for the provider, adding system prompt if needed.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted messages
        """
        formatted_messages = []
        
        # Check if there's already a system message
        has_system_message = any(
            (isinstance(msg, dict) and msg.get('role') == 'system') or
            (hasattr(msg, 'role') and msg.role == 'system')
            for msg in messages
        )
        
        # If no system message, add one
        if not has_system_message:
            system_prompt = self._generate_system_prompt()
            system_message = Message(role="system", content=system_prompt)
            formatted_messages.append(system_message)
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        # Log the formatted messages
        logger.debug(f"Formatted {len(formatted_messages)} messages for provider")
        
        return formatted_messages
    
    def _generate_system_prompt(self) -> str:
        """Generate a system prompt for the model based on its configuration and context.
        
        Returns:
            System prompt string
        """
        # Build the prompt
        prompt_parts = []
        
        # Add the model identity
        prompt_parts.append(f"# Model: {self.name}")
        
        # Add the model role if defined
        if self.role:
            prompt_parts.append(f"\n## Your Role\nYou are an AI assistant with the role: {self.role}")
        
        # Add team context if available
        if hasattr(self, 'team') and self.team is not None:
            prompt_parts.append(f"\n## Your Collaborators\nYou are part of the '{self.team.name}' group.")
            
            # Include team members if available
            if hasattr(self.team, 'models') and isinstance(self.team.models, dict):
                team_members = [name for name in self.team.models.keys() if name != self.name]
                if team_members:
                    members_str = ", ".join(team_members)
                    prompt_parts.append(f"Your collaborators: {members_str}")
                else:
                    prompt_parts.append("You are the only member of this group.")
        
        # Add tool result behavior descriptions
        adhesives = getattr(self, 'adhesives', set())
        if adhesives:
            prompt_parts.append("\n## Tool Result Behavior")
            behavior_descriptions = []
            
            if AdhesiveType.GLUE in adhesives:
                behavior_descriptions.append("**Shared Results**: Some tool results are automatically shared and persisted within your group.")
            if AdhesiveType.VELCRO in adhesives:
                behavior_descriptions.append("**Private Results**: Some tool results are kept private to you and persist only for the current session.")
            
            if behavior_descriptions:
                 prompt_parts.append("\n".join(behavior_descriptions))

        # Add response guidelines
        prompt_parts.append("""
## Response Guidelines
1. Stay focused on your role and the current task.
2. **Goal Adherence:** After using tools or communicating with collaborators, always review the original request and ensure your final response directly addresses the primary objective.
3. Consider how tool results are shared or persisted when choosing tools.
4. Collaborate effectively with your collaborators.
5. Follow established communication patterns.
6. Maintain a professional and helpful tone.
""")

        # Add generic tool usage instructions
        prompt_parts.append("""
## Tool Usage Instructions
To use the available tools:
- If you support native tool calling (e.g., function calling), use that method. Provide all required parameters as specified in the tool description.
- In some situations, you might be instructed to use a specific format (e.g., a JSON object or a specific code block) to trigger a tool call. Follow those instructions precisely if provided.
- Always refer to the "Available Tools" section (added when tools are available) for names, descriptions, and parameters.
""")

        # Join all parts
        return "\n".join(prompt_parts)
    
    def _prepare_tools_description(self, tools: List[Dict[str, Any]]) -> str:
        """Prepare a detailed description of available tools for the model.
        
        Args:
            tools: List of tools to describe
            
        Returns:
            Tool description string
        """
        if not tools:
            return ""
        
        tool_descriptions = ["## Available Tools\n"]
        
        for i, tool in enumerate(tools):
            name = tool.get("name", f"tool_{i}")
            description = tool.get("description", "No description available")
            
            # Get parameters info
            parameters = tool.get("parameters", {})
            required_params = parameters.get("required", [])
            properties = parameters.get("properties", {})
            
            # Format tool description
            tool_desc = [f"### {name}"]
            tool_desc.append(description)
            
            if properties:
                tool_desc.append("\nParameters:")
                for param_name, param_info in properties.items():
                    is_required = param_name in required_params
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if is_required else ""
                    
                    tool_desc.append(f"- `{param_name}`{req_marker}: {param_type} - {param_desc}")
            
            tool_descriptions.append("\n".join(tool_desc))

        return "\n\n".join(tool_descriptions)
    
    def _add_prompt_engineering(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> List[Message]:
        """Add prompt engineering to the messages before sending to the provider.
        
        Args:
            messages: List of messages to enhance
            tools: Optional list of tools to describe
            
        Returns:
            Enhanced messages
        """
        # Format messages for the provider
        formatted_messages = self._format_messages_for_provider(messages)
        
        # If tools are provided and there's no explicit tool description, add one
        if tools:
            # Check if there's already a tool description in the messages
            has_tool_description = any(
                "Available Tools" in (msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', ''))
                for msg in formatted_messages
            )
            
            if not has_tool_description:
                # Check if there's a communicate tool
                communicate_tool = None
                for tool in tools:
                    if tool.get("name") == "communicate":
                        communicate_tool = tool
                        break
                
                # Append tool descriptions to the system message
                for i, msg in enumerate(formatted_messages):
                    if (isinstance(msg, dict) and msg.get('role') == 'system') or (hasattr(msg, 'role') and msg.role == 'system'):
                        tool_description = self._prepare_tools_description(tools)
                        
                        # Add special instructions for the communicate tool if it exists
                        if communicate_tool:
                            tool_description += "\n\n## Special Instructions for the Communicate Tool\n\n"
                            tool_description += "The `communicate` tool allows you to send messages to other models or teams. "
                            tool_description += "You can use it to collaborate with other models in your team or in other teams.\n\n"
                            tool_description += "Example usage:\n\n"
                            
                            # Native tool call example
                            tool_description += "Native tool call:\n"
                            tool_description += "```\n"
                            tool_description += "communicate(target_type=\"model\", target_name=\"assistant\", message=\"Hello, can you help me with this task?\")\n"
                            tool_description += "```\n\n"
                            
                            # Simulated tool call example
                            tool_description += "Simulated tool call (for models without native tool support):\n"
                            tool_description += "```tool_call\n"
                            tool_description += "tool_name: communicate\n"
                            tool_description += "parameters: {\"target_type\": \"model\", \"target_name\": \"assistant\", \"message\": \"Hello, can you help me with this task?\"}\n"
                            tool_description += "```\n\n"
                            
                            # Add information about available models and teams
                            if hasattr(self, 'team') and self.team:
                                # List models in the same team
                                tool_description += "Models in your team:\n"
                                for model_name in self.team.models.keys():
                                    if model_name != self.name:  # Don't include self
                                        tool_description += f"- {model_name}\n"
                                
                                # List other teams if there are outgoing flows
                                if hasattr(self.team, 'outgoing_flows') and self.team.outgoing_flows:
                                    tool_description += "\nTeams you can communicate with:\n"
                                    for flow in self.team.outgoing_flows:
                                        tool_description += f"- {flow.target.name}\n"
                        
                        if isinstance(msg, dict):
                            msg['content'] += "\n\n" + tool_description
                        else:
                            msg.content += "\n\n" + tool_description
                        break
        
        return formatted_messages
    
    def _format_tool_for_provider(self, name: str, tool: Any) -> Dict[str, Any]:
        """Format a tool for the provider.
        
        Args:
            name: Tool name
            tool: Tool instance
            
        Returns:
            Formatted tool dictionary
        """
        # Get tool description
        description = getattr(tool, 'description', 'No description available')
        
        # Get parameters info
        parameters = getattr(tool, 'parameters', {})
        required_params = parameters.get("required", [])
        properties = parameters.get("properties", {})
        
        # Format tool description
        formatted_tool = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",  # Required by Gemini API
                "required": required_params,
                "properties": properties
            }
        }
        
        return formatted_tool
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a response from the model.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model
            
        Returns:
            The generated response
        """
        # Use the provider to generate a response if available
        if self.provider:
            try:
                # Add prompt engineering to the messages
                enhanced_messages = self._add_prompt_engineering(messages, tools)
                
                # Convert messages to the format expected by the provider
                provider_messages = []
                for msg in enhanced_messages:
                    if isinstance(msg, dict):
                        provider_messages.append(msg)
                    elif hasattr(msg, '__dict__'):
                        provider_messages.append(msg.__dict__)
                    else:
                        # Handle case where message might be a string or other type
                        provider_messages.append({"role": "user", "content": str(msg)})
                
                # Get available tools if any and format them for the provider
                provider_tools = None
                tools_source = None
                if tools is not None:
                    # Prioritize tools passed as argument (e.g., from cli.py)
                    tools_source = tools
                    logger.debug(f"Formatting {len(tools_source)} tools passed as argument.")
                elif hasattr(self, 'tools') and self.tools:
                    # Fallback to tools attached to the model instance
                    tools_source = self.tools
                    logger.debug(f"Formatting {len(tools_source)} tools attached to model.")
                
                if tools_source:
                    provider_tools = []
                    # Ensure tools_source is iterable (e.g., dict.items() or list)
                    items_to_iterate = tools_source.items() if isinstance(tools_source, dict) else tools_source
                    
                    for item in items_to_iterate:
                        tool_name = None
                        tool_instance = None
                        
                        if isinstance(tools_source, dict):
                            tool_name, tool_instance = item # item is (key, value) from .items()
                        elif isinstance(item, dict) and "name" in item:
                            # Handle case where a list of formatted tools might be passed
                            provider_tools.append(item)
                            continue
                        else:
                            # Attempt to handle list of tool instances (less common)
                            if hasattr(item, 'name'):
                                tool_name = item.name
                                tool_instance = item
                            else:
                                logger.warning(f"Skipping tool formatting for unrecognized item type: {type(item)}")
                                continue

                        # Format the tool instance if we have one
                        if tool_name and tool_instance:
                            if isinstance(tool_instance, dict) and "name" in tool_instance:
                                # Already formatted?
                                provider_tools.append(tool_instance)
                            else:
                                formatted_tool = self._format_tool_for_provider(tool_name, tool_instance)
                                provider_tools.append(formatted_tool)
                        elif tool_name and not tool_instance:
                             logger.warning(f"Tool '{tool_name}' found in source but instance is missing.")

                # Generate response using the provider
                if hasattr(self.provider, 'generate_response') and callable(self.provider.generate_response):
                    logger.debug("Calling provider's generate_response method")
                    logger.debug(f"Passing {len(provider_tools) if provider_tools else 0} tools to provider")
                    response = await self.provider.generate_response(provider_messages, provider_tools)
                    logger.debug(f"Provider response: {response}")
                    return response
                else:
                    logger.warning(f"Provider {self.provider.__class__.__name__} does not have a generate_response method")
                    raise ValueError(f"Provider {self.provider.__class__.__name__} does not have a generate_response method")
            except Exception as e:
                logger.error(f"Error generating response with provider: {e}")
                logger.exception("Exception details:")
                raise
        else:
            logger.error("No provider available for generating response")
            raise ValueError("No provider available for generating response")
    
    async def process_tool_result(self, tool_result: ToolResult) -> str:
        """Process a tool result and generate a response.
        
        Args:
            tool_result: The tool result to process
            
        Returns:
            The generated response
        """
        # Create a message from the tool result
        message = Message(
            role="function",
            content=tool_result.result,
            name=tool_result.tool_name
        )
        
        # Generate a response using the provider
        try:
            return await self.generate_response([message])
        except Exception as e:
            logger.error(f"Error processing tool result: {e}")
            return f"I'm sorry, I encountered an error while processing the tool result: {str(e)}"
