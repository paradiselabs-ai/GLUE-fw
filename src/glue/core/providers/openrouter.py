"""
OpenRouter provider implementation for the GLUE framework.

This module contains the OpenRouter provider class that handles
communication with the OpenRouter API for model interactions.
OpenRouter provides a unified API for accessing various LLM providers,
including free models that are perfect for testing during development.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, Callable, AsyncIterable
import httpx

from glue.core.schemas import Message, ToolCall, ToolResult

# Set up logging
logger = logging.getLogger("glue.model.openrouter")

# Environment variable name for the API key
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenrouterProvider:
    """Provider implementation for OpenRouter models.
    
    OpenRouter provides a unified API for accessing various LLM providers,
    including free models that are perfect for testing during development.
    """
    
    def __init__(self, model):
        """Initialize a new OpenRouter provider.
        
        Args:
            model: The model using this provider
        """
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenRouter client."""
        try:
            # Import OpenAI client for OpenRouter (they use the same API format)
            from openai import AsyncOpenAI
            
            # Get API key from model config, environment, or use a dummy key in development mode
            api_key = self._get_api_key()
            
            # Create the client
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                http_client=httpx.AsyncClient(timeout=60.0),
            )
            logger.info(f"Initialized OpenRouter client for model {self.model.model_name}")
        except ImportError:
            logger.error("Failed to import OpenAI client. Install with: pip install openai")
            # Create a mock client for testing
            self.client = type('MockClient', (), {
                'chat': type('MockChat', (), {
                    'completions': type('MockCompletions', (), {
                        'create': None
                    })
                })
            })
    
    def _get_api_key(self) -> str:
        """Get the API key from the model config or environment.
        
        Returns:
            The API key
            
        Raises:
            ValueError: If no API key is found and not in development mode
        """
        # Try to get the API key from the model config
        api_key = self.model.api_key
        
        # If not in the model config, try to get it from the environment
        if not api_key:
            api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
            
            # If still no API key, try to load from .env file
            if not api_key:
                api_key = self._load_api_key_from_dotenv()
            
        # If still no API key, check if we're in development mode
        if not api_key:
            # Check if development mode is enabled
            is_development = getattr(self.model, 'development', False)
            
            if is_development:
                logger.warning(f"No OpenRouter API key provided but running in development mode. Using dummy key.")
                return "dummy_key_for_development"
            else:
                raise ValueError(f"No OpenRouter API key found. Set {OPENROUTER_API_KEY_ENV} environment variable or provide it in the model config.")
        
        return api_key
    
    def _load_api_key_from_dotenv(self) -> Optional[str]:
        """Load the API key from a .env file.
        
        Returns:
            The API key if found, None otherwise
        """
        try:
            # Check if dotenv is available
            import dotenv
            
            # Look for .env file in the current directory and load it
            if os.path.exists(".env"):
                dotenv.load_dotenv(".env")
                logger.info("Loaded environment variables from .env file")
                
                # Check if the API key is now in the environment
                return os.environ.get(OPENROUTER_API_KEY_ENV)
            
            # Also check for .env file in the parent directory
            parent_env = os.path.join("..", ".env")
            if os.path.exists(parent_env):
                dotenv.load_dotenv(parent_env)
                logger.info(f"Loaded environment variables from {parent_env} file")
                
                # Check if the API key is now in the environment
                return os.environ.get(OPENROUTER_API_KEY_ENV)
                
        except ImportError:
            logger.warning("python-dotenv not installed, cannot load .env file")
        
        return None
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Generate a response from the model.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model
            
        Returns:
            The generated response or a dict with tool calls
        """
        # Convert messages to the format expected by OpenRouter
        openrouter_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Prepare the API call parameters
        api_params = {
            "model": self.model.model,
            "messages": openrouter_messages,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            **self.model.api_params
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        try:
            # Call the OpenRouter API
            response = await self.client.chat.completions.create(**api_params)
            
            # Process the response
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                # Process tool calls
                tool_calls = []
                for tool_call in response.choices[0].message.tool_calls:
                    # Parse the arguments from JSON string to dict
                    arguments = json.loads(tool_call.function.arguments)
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments
                    })
                return {"tool_calls": tool_calls}
            else:
                # Return the text response
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}")
            raise
    
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
            try:
                # Execute the tool call
                result = await tool_executor(tool_call)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing tool call {tool_call.id}: {e}")
                # Create an error result
                results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Error executing tool: {str(e)}"
                ))
        
        return results
