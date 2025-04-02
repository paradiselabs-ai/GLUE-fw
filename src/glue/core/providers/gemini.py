"""
Google Gemini provider implementation for the GLUE framework.

This module contains the provider class that handles communication
with the Google Generative AI API for Gemini model interactions.
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional, Callable, AsyncIterable, Union

from glue.core.model import BaseModel, ProviderBase
from glue.core.schemas import Message, ToolCall, ToolResult

# Set up logging
logger = logging.getLogger("glue.providers.gemini")

class GeminiProvider(ProviderBase):
    """Provider implementation for Google Gemini models."""
    
    def __init__(self, model: BaseModel):
        """Initialize a new Gemini provider.
        
        Args:
            model: The model using this provider
        """
        super().__init__(model)
        self.client = None
        self.genai = None
        self._initialize_client()
    
    def _get_api_key(self) -> str:
        """Get the API key for Google Generative AI.
        
        Returns:
            The API key as a string
        """
        # Check if we're in development mode
        if getattr(self.model, 'development', False):
            logger.warning("Using dummy API key in development mode")
            return "dummy_key_for_development"
        
        # Check if the API key is provided in the model config
        if self.model.api_key:
            return self.model.api_key
        
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
            
            # Initialize the model
            model_name = self.model.model
            
            # Create the client
            self.client = genai.GenerativeModel(model_name=model_name)
            
            # Set generation config
            generation_config = {
                "temperature": self.model.temperature,
                "max_output_tokens": self.model.max_tokens,
                "top_p": 0.95,
                "top_k": 0,
            }
            
            # Apply any additional API parameters from the model config
            if hasattr(self.model, 'api_params') and self.model.api_params:
                generation_config.update(self.model.api_params)
            
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
        # Convert messages to the format expected by the Gemini API
        gemini_messages = []
        
        for msg in messages:
            role = msg.role
            # Map roles to Gemini's expected format
            if role == "system":
                role = "user"
            
            content = msg.content
            
            # Create a new message
            gemini_message = self.genai.types.Content(
                role=role,
                parts=[self.genai.types.Part.from_text(content)]
            )
            
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
            
            # Set up the tools for the model
            self.client.tools = [
                {
                    "function_declarations": function_declarations
                }
            ]
        
        try:
            # Generate the response
            response = await self.client.generate_content(gemini_messages)
            
            # Check if the response contains a function call
            if tools and hasattr(response.candidates[0].content.parts[0], 'function_call'):
                function_call = response.candidates[0].content.parts[0].function_call
                
                # Extract the tool call information
                tool_call = {
                    "name": function_call["name"],
                    "arguments": function_call["args"]
                }
                
                # Return a dictionary with tool calls
                return {
                    "tool_calls": [tool_call]
                }
            
            # Return the text response
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            # Return a generic error message
            return f"Error generating response: {str(e)}"
    
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
            # Execute the tool call
            tool_result = await tool_executor(tool_call)
            results.append(tool_result)
        
        return results
