"""
Base model implementation for the GLUE framework.

This module provides a base model implementation that can be used with any provider.
It includes prompt engineering capabilities and standardized interfaces for all models.
"""

import os
import importlib
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Type
import asyncio
from enum import Enum

from .types import Message, ToolResult

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
                # Convert messages to the format expected by the provider
                provider_messages = []
                for msg in messages:
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
                elif self.tools:
                    provider_tools = list(self.tools.values())
                
                # Generate response using the provider
                if hasattr(self.provider, 'generate_response') and callable(self.provider.generate_response):
                    logger.debug("Calling provider's generate_response method")
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
