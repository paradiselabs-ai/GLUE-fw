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
        self.model_name = self.config.get('model', 'gemini-1.5-pro')
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
    
    def add_tool(self, name: str, tool: Any):
        """Add a tool to this model.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        self.tools[name] = tool
    
    async def generate(self, content: str) -> str:
        """Generate a response from the model.
        
        Args:
            content: The content to generate a response for
            
        Returns:
            The generated response
        """
        # Create a simple message from the content
        message = Message(role="user", content=content)
        
        # Generate a response using the provider
        try:
            return await self.generate_response([message])
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    
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
        prompt_parts.append(f"# GLUE Model: {self.name}")
        
        # Add the model role if defined
        if self.role:
            prompt_parts.append(f"\n## Your Role\nYou are a model in the GLUE framework with the role: {self.role}")
        
        # Add framework context
        prompt_parts.append("""
## GLUE Framework
You are operating as part of the GLUE (GenAI Linking & Unification Engine) framework, which organizes AI models into teams with tools and defined communication patterns.

### Key Concepts
1. **Adhesive Tool Usage**: Tools can be used with different binding types.
2. **Team Communication**: You can communicate freely with team members.
3. **Magnetic Information Flow**: Information flows between teams in defined patterns.
""")
        
        # Add team context if available
        if hasattr(self, 'team') and self.team is not None:
            prompt_parts.append(f"\n## Your Team\nYou are part of the '{self.team.name}' team.")
            
            # Include team members if available
            if hasattr(self.team, 'models') and isinstance(self.team.models, dict):
                team_members = [name for name in self.team.models.keys() if name != self.name]
                if team_members:
                    members_str = ", ".join(team_members)
                    prompt_parts.append(f"Team members: {members_str}")
                else:
                    prompt_parts.append("You are the only member of this team.")
        
        # Add adhesive capabilities
        adhesives = getattr(self, 'adhesives', set())
        if adhesives:
            prompt_parts.append("\n## Your Adhesive Capabilities")
            adhesive_types = []
            
            if AdhesiveType.GLUE in adhesives:
                adhesive_types.append("**GLUE**: Permanent binding with team-wide persistence. Results are automatically shared with your team.")
            
            if AdhesiveType.VELCRO in adhesives:
                adhesive_types.append("**VELCRO**: Session-based binding with model-level persistence. Results persist for your current session and are private to you.")
            
            if AdhesiveType.TAPE in adhesives:
                adhesive_types.append("**TAPE**: One-time binding with no persistence. Results are used once and discarded.")
            
            prompt_parts.append("\n".join(adhesive_types))
        
        # Add tool information
        if self.tools:
            prompt_parts.append("\n## Available Tools")
            for name, tool in self.tools.items():
                # Get tool description
                description = getattr(tool, 'description', 'No description available')
                
                # Add tool information
                prompt_parts.append(f"- **{name}**: {description}")
        
        # Add response guidelines
        prompt_parts.append("""
## Response Guidelines
1. Stay focused on your role and the current task.
2. Use tools appropriately based on their adhesive types.
3. Collaborate effectively with team members.
4. Follow team-specific communication patterns.
5. Maintain a professional and helpful tone.
""")
        
        # Provider-specific instructions
        provider_name = self.provider_name.lower()
        if provider_name == "gemini":
            prompt_parts.append("""
## Tool Usage Instructions
When using tools, follow this specific function calling syntax for Gemini models:

```tool_code
tool_name(parameter1="value1", parameter2="value2")
```

Always use the exact tool names as provided below. Do not invent or modify tool names.
""")
            
            # Add specific examples for each available tool
            if self.tools:
                prompt_parts.append("\n### Tool Usage Examples")
                for name, tool in self.tools.items():
                    # Create a simple example for each tool
                    if name == "web_search":
                        prompt_parts.append(f"""
Example for {name}:
```tool_code
{name}(query="latest news in AI development")
```
""")
                    elif name == "file_handler":
                        prompt_parts.append(f"""
Example for {name}:
```tool_code
{name}(action="read", file_path="example.txt")
```
""")
                    else:
                        # Generic example for other tools
                        prompt_parts.append(f"""
Example for {name}:
```tool_code
{name}(parameter="example")
```
""")
        elif provider_name == "anthropic":
            prompt_parts.append("""
## Tool Usage Instructions
When using tools, use the <tool></tool> XML tags to indicate tool calls. Always provide required parameters.
""")
        elif provider_name == "openai":
            prompt_parts.append("""
## Tool Usage Instructions
When using tools (functions), clearly indicate your intent to call a function and provide all required parameters.
""")
        elif provider_name == "openrouter":
            # Generic instructions for OpenRouter (which might route to different models)
            prompt_parts.append("""
## Tool Usage Instructions
Follow the appropriate function calling format for the underlying model. Always provide all required parameters.
""")
        else:
            # Generic instructions for other providers
            prompt_parts.append("""
## Tool Usage Instructions
When using tools, clearly indicate which tool you're using and provide all required parameters.
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
                # Append tool descriptions to the system message
                for i, msg in enumerate(formatted_messages):
                    if (isinstance(msg, dict) and msg.get('role') == 'system') or (hasattr(msg, 'role') and msg.role == 'system'):
                        if isinstance(msg, dict):
                            msg['content'] += "\n\n" + self._prepare_tools_description(tools)
                        else:
                            msg.content += "\n\n" + self._prepare_tools_description(tools)
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
                
                # Get available tools if any
                provider_tools = None
                if tools:
                    provider_tools = tools
                elif hasattr(self, 'tools') and self.tools:
                    # Convert tools dictionary to list of tool definitions
                    provider_tools = []
                    for tool_name, tool in self.tools.items():
                        if isinstance(tool, dict) and "name" in tool:
                            # Tool is already formatted
                            provider_tools.append(tool)
                        else:
                            # Format the tool
                            formatted_tool = self._format_tool_for_provider(tool_name, tool)
                            provider_tools.append(formatted_tool)
                
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
