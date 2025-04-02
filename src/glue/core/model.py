"""
Base model implementation for the GLUE framework.

This module contains the base model class that provides abstraction
over different AI model providers and handles tool usage capabilities.
"""
import importlib
import logging
import os
import uuid
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
    GEMINI = "gemini"
    CUSTOM = "custom"
    TEST = "test"
    MOCK = "mock"


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
        self.development = getattr(config, 'development', False)
        
        # Initialize the client if not in development mode
        if not self.development:
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
        elif self.provider == ModelProvider.GEMINI:
            from glue.core.providers.gemini import GeminiProvider
            return GeminiProvider
        elif self.provider == ModelProvider.TEST:
            from glue.core.providers.test import TestProvider
            return TestProvider
        elif self.provider == ModelProvider.MOCK:
            # For mock provider, use the test provider
            from glue.core.providers.test import TestProvider
            return TestProvider
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
        
        # Generate a trace ID for Portkey tracking
        self._trace_id = str(uuid.uuid4())
        
        # Handle backward compatibility with tests that pass arguments directly
        if config is None and kwargs:
            self._is_test_instance = True
            # For tests, create a minimal BaseModel instance directly
            # Extract required parameters with defaults for backward compatibility
            name = kwargs.get('name', 'test_model')
            provider = kwargs.get('provider', 'custom')
            model_name = kwargs.get('model', provider)  # Use provider as model name if not specified
            
            # Create a config attribute for test compatibility
            self.config = {
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
            }
            
            # Create a minimal config for BaseModel initialization
            mock_config = type('MockConfig', (), self.config)()
            config = mock_config
            
            # Extract adhesives separately as they're handled differently
            self.adhesives = kwargs.get('adhesives', set())
        elif isinstance(config, dict):
            # Handle dictionary config directly (e.g., from app setup)
            self.adhesives = set(config.get('adhesives', []))
            # Store the dictionary itself, ensure critical keys exist for provider setup
            self.config = config
            # Basic validation or default setting if keys are missing
            self.config.setdefault('provider', 'mock') # Default if provider missing
            self.config.setdefault('model', self.config['provider']) # Default model to provider name
            self.config.setdefault('role', 'assistant') # Default role

        else: # Assumes config is ModelConfig object
            self.adhesives = set()
            # Add adhesives from config
            if hasattr(config, 'adhesives') and config.adhesives:
                for adhesive in config.adhesives:
                    self.adhesives.add(adhesive)
            # Store the config dictionary for test compatibility
            if hasattr(config, '__dict__'):
                 self.config = config.__dict__
            else:
                 # Fallback if it's not a dict or object with __dict__
                 self.config = {'model': getattr(config, 'model', 'unknown')} if config else {}

        # Only add GLUE adhesive type for test compatibility if no adhesives were specified
        if not self.adhesives:
            self.adhesives.add(AdhesiveType.GLUE)

        # Ensure self.config is always a dictionary for provider loading
        if not isinstance(self.config, dict):
            # Attempt conversion if possible, otherwise initialize empty
            try:
                self.config = vars(self.config)
            except TypeError:
                 # Handle cases where config might be a simple type or non-dict-like object
                 print(f"Warning: Model config is not a dictionary or easily convertible: {type(config)}. Initializing provider might fail.")
                 self.config = {}

        # Dynamically load the provider based on the config
        provider_name = self.config.get('provider', 'mock') # Use mock if no provider specified
        self.provider = self._load_provider(provider_name, self.config)
        
        # Ensure name is always set as an attribute (critical for tests)
        self.name = kwargs.get('name', self.config.get('name', 'unnamed_model'))
        
        # Create a ModelConfig object for BaseModel initialization
        model_config = type('ModelConfig', (), {
            'name': self.name,
            'provider': self.config.get('provider', 'mock'),
            'model': self.config.get('model', 'mock'),
            'temperature': self.config.get('temperature', 0.7),
            'max_tokens': self.config.get('max_tokens', 1024),
            'description': self.config.get('description', ''),
            'api_key': self.config.get('api_key', None),
            'api_params': self.config.get('api_params', {}),
            'provider_class': self.config.get('provider_class', 'glue.core.providers.mock.MockProvider')
        })()
        
        super().__init__(model_config)
        
        # Only initialize client for non-test instances
        if not self._is_test_instance:
            self._initialize_client()
        
        # Set up additional properties
        self.team = None
        self.role = kwargs.get('role', self.config.get('role', 'assistant'))
        self.tools = {}
    
    def _load_provider(self, provider_name: str, config: dict):
        """Load a provider based on name and configuration.
        
        This method dynamically loads provider classes and optionally
        wraps them with Portkey for API key management and tracking.
        
        Args:
            provider_name: Name of the provider to load
            config: Provider configuration
            
        Returns:
            Provider instance
        """
        # Map provider names to module paths
        provider_modules = {
            'openai': 'glue.core.providers.openai',
            'anthropic': 'glue.core.providers.anthropic',
            'openrouter': 'glue.core.providers.openrouter',
            'test': 'glue.core.providers.test',
            'mock': 'glue.core.providers.mock',
            'custom': config.get('provider_class', 'glue.core.providers.mock')
        }
        
        # Get the module path for the provider
        module_path = provider_modules.get(provider_name.lower())
        if not module_path:
            logger.warning(f"Unknown provider: {provider_name}. Using mock provider.")
            module_path = provider_modules['mock']
        
        try:
            # Import the provider module
            module = importlib.import_module(module_path)
            
            # Get the provider class
            if provider_name.lower() == 'custom':
                # For custom providers, the class name is in the path
                class_name = module_path.split('.')[-1]
            else:
                # For built-in providers, use the capitalized name + "Provider"
                class_name = f"{provider_name.capitalize()}Provider"
            
            provider_class = getattr(module, class_name)
            
            # Create an instance of the provider
            provider_instance = provider_class(self)
            
            # Check if Portkey integration is enabled
            portkey_enabled = os.environ.get("PORTKEY_ENABLED", "").lower() in ("true", "1", "yes")
            
            if portkey_enabled and not self._is_test_instance:
                # Import the Portkey wrapper
                from glue.core.providers.portkey_wrapper import wrap_provider
                
                # Wrap the provider with Portkey
                model_name = config.get('model', 'unknown')
                provider_instance = wrap_provider(provider_instance, model_name, self._trace_id)
                logger.info(f"Provider {provider_name} wrapped with Portkey (trace_id: {self._trace_id})")
            
            return provider_instance
        
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading provider {provider_name}: {str(e)}")
            
            # Fall back to mock provider
            from glue.core.providers.mock import MockProvider
            return MockProvider(self)

    def set_team(self, team):
        """Set the team this model belongs to."""
        self.team = team
    
    async def add_tool(self, name: str, tool: Any):
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
    
    async def setup(self) -> None:
        """Set up the model by initializing any required resources.
        
        This is a placeholder implementation for test compatibility.
        In a real implementation, this would initialize any required resources.
        """
        # Nothing to do in the base implementation
        pass
