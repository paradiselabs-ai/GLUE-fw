"""
Anthropic provider implementation for the GLUE framework.

This module contains the Anthropic provider class that handles
communication with the Anthropic API for Claude model interactions.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, Callable, AsyncIterable

from glue.core.schemas import Message, ToolCall, ToolResult

# Set up logging
logger = logging.getLogger("glue.model.anthropic")

# Environment variable name for the API key
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider:
    """Provider implementation for Anthropic Claude models."""
    
    def __init__(self, model):
        """Initialize a new Anthropic provider.
        
        Args:
            model: The model using this provider
        """
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            # Import Anthropic client
            from anthropic import Anthropic
            
            # Get API key from model config, environment, or use a dummy key in development mode
            api_key = self._get_api_key()
            
            # Create the client
            self.client = Anthropic(api_key=api_key)
            
        except ImportError:
            logger.warning("Anthropic package not installed. Using mock client.")
            # Create a mock client for testing
            self.client = type('MockClient', (), {
                'messages': type('MockMessages', (), {
                    'create': None
                })
            })
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            raise
    
    def _get_api_key(self) -> str:
        """Get the API key for Anthropic.
        
        Returns:
            The API key as a string
        
        Raises:
            ValueError: If no API key is found and not in development mode
        """
        # First, check if the API key is provided in the model config
        if hasattr(self.model, "api_key") and self.model.api_key:
            return self.model.api_key
        
        # Next, check environment variables
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
        if api_key:
            return api_key
        
        # If not found in environment, try to load from .env file
        api_key = self._load_api_key_from_dotenv()
        if api_key:
            return api_key
        
        # If we're in development mode, use a dummy key
        if hasattr(self.model, "development") and self.model.development:
            logger.warning("No Anthropic API key provided but running in development mode. Using dummy key.")
            return "dummy_key_for_development"
        
        # If we get here, no API key was found
        raise ValueError(
            f"No Anthropic API key found. Please set the {ANTHROPIC_API_KEY_ENV} environment variable, "
            "add it to a .env file, or provide it in the model configuration."
        )
    
    def _load_api_key_from_dotenv(self) -> Optional[str]:
        """Load the API key from a .env file.
        
        Returns:
            The API key as a string, or None if not found
        """
        # Check if .env file exists
        if not os.path.exists(".env"):
            return None
        
        try:
            # Import dotenv
            from dotenv import load_dotenv
            
            # Load environment variables from .env file
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
            
            # Check if the API key is now in the environment
            return os.environ.get(ANTHROPIC_API_KEY_ENV)
        except ImportError:
            logger.warning("python-dotenv package not installed. Unable to load from .env file.")
            return None
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
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
        # Convert messages to the format expected by Anthropic
        anthropic_messages = [
            {"role": "user" if msg.role == "user" else "assistant", "content": msg.content}
            for msg in messages
        ]
        
        # Prepare the API call parameters
        api_params = {
            "model": self.model.model,
            "messages": anthropic_messages,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
        }
        
        # Add additional parameters if available
        if hasattr(self.model, "api_params") and self.model.api_params:
            api_params.update(self.model.api_params)
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
        
        try:
            # Call the Anthropic API
            response = await self.client.messages.create(**api_params)
            
            # Process the response
            if tools and hasattr(response, "content") and any(c.type == "tool_use" for c in response.content):
                # Extract tool calls from the response
                tool_calls = []
                for content in response.content:
                    if content.type == "tool_use":
                        tool_calls.append({
                            "id": content.id,
                            "name": content.name,
                            "arguments": content.input
                        })
                
                return {"tool_calls": tool_calls}
            else:
                # Extract text content from the response
                for content in response.content:
                    if content.type == "text":
                        return content.text
                
                # If no text content found, return the first content
                return response.content[0].text if response.content else ""
        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {e}")
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
            result = await tool_executor(tool_call)
            results.append(result)
        return results
