"""
Google Gemini provider implementation for the GLUE framework.

This module contains the provider class that handles communication
with the Google Generative AI API for Gemini model interactions.
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional, Callable, AsyncIterable, Union

from ..base_model import BaseModel
from ..types import Message, ToolResult

# Set up logging
logger = logging.getLogger("glue.providers.gemini")

class GeminiProvider:
    """Provider implementation for Google Gemini models."""
    
    # Class-level cache to prevent multiple initializations
    _instances = {}
    
    def __new__(cls, model: BaseModel):
        """Singleton pattern to ensure only one instance per model."""
        # Use model name as the key for the cache
        model_name = getattr(model, 'name', str(id(model)))
        
        # If we already have an instance for this model, return it
        if model_name in cls._instances:
            logger.debug(f"Reusing existing GeminiProvider instance for model {model_name}")
            return cls._instances[model_name]
        
        # Create a new instance
        instance = super(GeminiProvider, cls).__new__(cls)
        cls._instances[model_name] = instance
        return instance
    
    def __init__(self, model: BaseModel):
        """Initialize a new Gemini provider.
        
        Args:
            model: The model using this provider
        """
        # Skip initialization if this instance has already been initialized
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2]  # Get the caller of this function
        logger.debug(f"GeminiProvider initialized from {caller.filename}:{caller.lineno}")
        
        self.model = model
        self.client = None
        self.genai = None
        self._initialize_client()
        
        # Mark as initialized
        self.initialized = True
    
    def _get_api_key(self) -> str:
        """Get the API key for Google Generative AI.
        
        Returns:
            The API key as a string
        """
        # Check if we're in development mode
        if hasattr(self.model, 'development') and self.model.development:
            logger.warning("Using dummy API key in development mode")
            return "dummy_key_for_development"
        
        # Check if the API key is provided in the model config
        if hasattr(self.model, 'api_key') and self.model.api_key:
            return self.model.api_key
            
        # Check if the API key is in the model's config dictionary
        if hasattr(self.model, 'config') and isinstance(self.model.config, dict) and 'api_key' in self.model.config:
            api_key = self.model.config.get('api_key')
            if api_key:
                return api_key
        
        # Check environment variable
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            return api_key
            
        # Check for GEMINI_API_KEY as a fallback
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("Using GEMINI_API_KEY environment variable (consider renaming to GOOGLE_API_KEY)")
            return api_key
        
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            if os.path.exists(".env"):
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    return api_key
                    
                # Check for GEMINI_API_KEY in .env as a fallback
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    logger.info("Using GEMINI_API_KEY from .env file (consider renaming to GOOGLE_API_KEY)")
                    return api_key
        except ImportError:
            logger.warning("dotenv package not installed, skipping .env loading")
        
        # Raise error if no API key is found
        raise ValueError(
            "No API key found for Google Generative AI. "
            "Please set the GOOGLE_API_KEY environment variable "
            "or provide it in the model configuration."
        )
    
    def _initialize_client(self):
        """Initialize the Google Generative AI client."""
        try:
            import google.generativeai as genai
            
            # Store the module for later use
            self.genai = genai
            
            # Configure the API key
            api_key = self._get_api_key()
            genai.configure(api_key=api_key)
            
            # Get the model name from various possible locations
            model_name = None
            
            # Check direct model attribute
            if hasattr(self.model, 'model'):
                model_name = self.model.model
            
            # Check config dictionary
            elif hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                model_name = self.model.config.get('model')
            
            # Default to a sensible model if none specified
            if not model_name:
                # Use Gemini 2.5 Pro as the default model for the best performance
                # This is the recommended model for GLUE Forge
                model_name = "gemini-1.5-pro"
                logger.info(f"No model specified, using default model: {model_name}")
            
            # Create the client
            self.client = genai.GenerativeModel(model_name=model_name)
            
            # Set generation config
            generation_config = {
                "temperature": 0.7,  # Default temperature
                "max_output_tokens": 1024,  # Default max tokens
                "top_p": 0.95,
                "top_k": 0,
            }
            
            # Update with model temperature if available
            if hasattr(self.model, 'temperature'):
                generation_config["temperature"] = self.model.temperature
            elif hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                if 'temperature' in self.model.config:
                    generation_config["temperature"] = self.model.config.get('temperature')
            
            # Update with model max_tokens if available
            if hasattr(self.model, 'max_tokens'):
                generation_config["max_output_tokens"] = self.model.max_tokens
            elif hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                if 'max_tokens' in self.model.config:
                    generation_config["max_output_tokens"] = self.model.config.get('max_tokens')
            
            # Apply any additional API parameters from the model config
            if hasattr(self.model, 'api_params') and self.model.api_params:
                generation_config.update(self.model.api_params)
            elif hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                if 'api_params' in self.model.config:
                    api_params = self.model.config.get('api_params', {})
                    if isinstance(api_params, dict):
                        generation_config.update(api_params)
            
            self.client.generation_config = generation_config
            
            logger.info(f"Initialized Google Generative AI client for model {model_name}")
        
        except ImportError:
            logger.error("Failed to import google.generativeai. Please install it with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google Generative AI client: {e}")
            raise
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response from the Gemini model.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model
            
        Returns:
            Either a string response or a dictionary with tool calls
        """
        try:
            # Convert messages to the format expected by the Gemini API
            # The Google Generative AI API expects a list of dictionaries with 'role' and 'parts'
            gemini_messages = []
            
            for msg in messages:
                # Handle different message formats
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role
                    content = msg.content
                else:
                    # Fallback for string or other types
                    role = 'user'
                    content = str(msg)
                
                # Map roles to Gemini's expected format
                if role == "system":
                    role = "user"
                
                # Create a new message in the format expected by the Gemini API
                gemini_message = {
                    "role": role,
                    "parts": [{"text": content}]
                }
                
                gemini_messages.append(gemini_message)
            
            # Configure tool calling if tools are provided
            if tools:
                # Convert tools to the format expected by the Gemini API
                function_declarations = []
                
                for tool in tools:
                    function_declaration = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                    function_declarations.append(function_declaration)
                
                # Generate response with tools
                response = await self.client.generate_content_async(
                    gemini_messages,
                    generation_config=self.client.generation_config,
                    tools=[{"function_declarations": function_declarations}]
                )
            else:
                # Generate response without tools
                response = await self.client.generate_content_async(
                    gemini_messages,
                    generation_config=self.client.generation_config
                )
            
            # Extract the response text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                parts = response.parts
                if parts and len(parts) > 0:
                    return parts[0].text
            
            # Handle tool calls if present
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0:
                        return parts[0].text
            
            # Fallback to string representation
            return str(response)
        
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise
    
    async def process_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]], 
        tool_executor: Callable[[Dict[str, Any]], AsyncIterable[ToolResult]]
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
            # Execute the tool call
            tool_result = await tool_executor(tool_call)
            results.append(tool_result)
        
        return results
