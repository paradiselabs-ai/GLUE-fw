"""
Base model implementation for the GLUE framework.

This module contains the base model class that provides abstraction
over different AI model providers and handles tool usage capabilities.
"""
import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Union, Type, AsyncIterable, Set

from glue.core.schemas import Message, ToolCall, ToolResult, ModelConfig, AdhesiveType


# Set up logging
logger = logging.getLogger("glue.model")


class ModelProvider(str, Enum):
    """Enumeration of supported model providers."""
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"


class BaseModel:
    """Base class for all models in the GLUE framework."""
    
    def __init__(self, config: ModelConfig):
        """Initialize a new model.
        
        Args:
            config: Model configuration
        """
        self.name = config.name
        self.provider = config.provider
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.description = config.description
        self.api_key = config.api_key
        self.api_params = config.api_params
        self.provider_class = config.provider_class
        self.client = None
        self.provider_instance = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        provider_class = self.get_provider_class()
        self.provider_instance = provider_class(self)
        self.client = self.provider_instance.client
    
    def get_provider_class(self) -> Type:
        """Get the provider class for this model.
        
        Returns:
            The provider class
            
        Raises:
            ImportError: If the provider class cannot be imported
            ValueError: If the provider is not supported
        """
        if self.provider == ModelProvider.OPENAI:
            from glue.core.providers.openai import OpenAIProvider
            return OpenAIProvider
        elif self.provider == ModelProvider.ANTHROPIC:
            from glue.core.providers.anthropic import AnthropicProvider
            return AnthropicProvider
        elif self.provider == ModelProvider.OPENROUTER:
            from glue.core.providers.openrouter import OpenRouterProvider
            return OpenRouterProvider
        elif self.provider == ModelProvider.CUSTOM:
            # Import custom provider class
            if not self.provider_class:
                raise ValueError("Custom provider requires provider_class to be set")
                
            module_path, class_name = self.provider_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
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
        return await self._generate_response(messages, tools)
    
    async def _generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a response from the model (provider-specific implementation).
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model
            
        Returns:
            The generated response
        
        Raises:
            NotImplementedError: This method must be implemented by provider-specific classes
        """
        raise NotImplementedError("This method must be implemented by provider-specific classes")
    
    async def process_tool_calls(
        self, 
        tool_calls: List[ToolCall], 
        tool_executor: Callable[[ToolCall], AsyncIterable[ToolResult]]
    ) -> List[ToolResult]:
        """Process tool calls from the model.
        
        Args:
            tool_calls: List of tool calls to process
            tool_executor: Function to execute a tool call
            
        Returns:
            List of tool results
        """
        results = []
        for tool_call in tool_calls:
            result = await tool_executor(tool_call)
            results.append(result)
        return results


# Provider-specific implementations will be in separate modules
class ProviderBase(ABC):
    """Base class for provider-specific implementations."""
    
    def __init__(self, model: BaseModel):
        """Initialize a new provider.
        
        Args:
            model: The model using this provider
        """
        self.model = model
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def process_tool_calls(
        self, 
        tool_calls: List[ToolCall], 
        tool_executor: Callable[[ToolCall], AsyncIterable[ToolResult]]
    ) -> List[ToolResult]:
        """Process tool calls from the model.
        
        Args:
            tool_calls: List of tool calls to process
            tool_executor: Function to execute a tool call
            
        Returns:
            List of tool results
        """
        pass

class Model(BaseModel):
    """Concrete implementation of the model class that can be used in teams."""
    
    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        """Initialize a new model.
        
        Args:
            config: Model configuration
            **kwargs: Keyword arguments for backward compatibility
        """
        # Flag to track if this is a test instance
        self._is_test_instance = False
        
        # Handle backward compatibility with tests that pass arguments directly
        if config is None and kwargs:
            self._is_test_instance = True
            # For tests, create a minimal BaseModel instance directly
            # Extract required parameters with defaults for backward compatibility
            name = kwargs.get('name', 'test_model')
            provider = kwargs.get('provider', 'custom')
            model_name = kwargs.get('model', provider)  # Use provider as model name if not specified
            
            # Create a minimal config
            mock_config = type('MockConfig', (), {
                'name': name,
                'provider': provider,
                'model': model_name,
                'temperature': 0.7,
                'max_tokens': 1024,
                'description': '',
                'api_key': None,
                'api_params': {},
                'provider_class': 'glue.core.providers.mock.MockProvider',
                'role': kwargs.get('role', 'assistant')
            })()
            
            config = mock_config
            
            # Extract adhesives separately as they're handled differently
            self.adhesives = kwargs.get('adhesives', set())
        else:
            self.adhesives = set()
            # Add adhesives from config
            if hasattr(config, 'adhesives') and config.adhesives:
                for adhesive in config.adhesives:
                    self.adhesives.add(AdhesiveType(adhesive))
        
        # Initialize base properties without calling _initialize_client
        self.name = config.name
        self.provider = config.provider
        self.model = config.model
        self.temperature = getattr(config, 'temperature', 0.7)
        self.max_tokens = getattr(config, 'max_tokens', 1024)
        self.description = getattr(config, 'description', '')
        self.api_key = getattr(config, 'api_key', None)
        self.api_params = getattr(config, 'api_params', {})
        self.provider_class = getattr(config, 'provider_class', None)
        self.client = None
        self.provider_instance = None
        
        # Only initialize client for non-test instances
        if not self._is_test_instance:
            self._initialize_client()
        
        # Set up additional properties
        self.team = None
        self.role = kwargs.get('role', getattr(config, 'role', 'assistant'))
        self.tools = {}
    
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
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all tools available to this model.
        
        Returns:
            Dictionary of tool name to tool instance
        """
        return self.tools
    
    def has_adhesive(self, adhesive: AdhesiveType) -> bool:
        """Check if this model supports the given adhesive type.
        
        Args:
            adhesive: Adhesive type to check
            
        Returns:
            True if the model supports the adhesive, False otherwise
        """
        return adhesive in self.adhesives
