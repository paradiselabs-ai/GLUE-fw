"""
Google Gemini provider implementation for the GLUE framework.

This module contains the provider class that handles communication
with the Google Generative AI API for Gemini model interactions.
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional, Callable, AsyncIterable, Union
import uuid

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
        
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            if os.path.exists(".env"):
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
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
            if hasattr(self.model, 'model_name'):
                model_name = self.model.model_name
            elif hasattr(self.model, 'model'):
                model_name = self.model.model
            
            # Check config dictionary
            elif hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                model_name = self.model.config.get('model')
                
            # Debug log to see what we're finding
            logger.debug(f"Model config: {getattr(self.model, 'config', {})}")
            logger.debug(f"Found model name: {model_name}")
            
            # Default to a sensible model if none specified
            if not model_name:
                # Use Gemini 1.5 Pro as the default model for the best performance
                # This is the recommended model for GLUE Forge
                model_name = "gemini-1.5-pro"
                logger.info(f"No model specified, using default model: {model_name}")
            else:
                logger.info(f"Using specified model: {model_name}")
            
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
            
            # Track if we've seen a system message
            has_system_message = False
            system_content = ""
            
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
                
                # Special handling for system messages with Gemini
                if role == "system":
                    has_system_message = True
                    system_content = content
                    # Don't add system message directly, we'll prepend it to the first user message
                    continue
                
                # Create a new message in the format expected by the Gemini API
                gemini_message = {
                    "role": role,
                    "parts": [{"text": content}]
                }
                
                # If this is the first user message and we have a system message,
                # prepend the system content to it
                if role == "user" and has_system_message and system_content:
                    gemini_message["parts"] = [{"text": f"{system_content}\n\n# User Query:\n{content}"}]
                    has_system_message = False  # Reset so we only prepend once
                
                gemini_messages.append(gemini_message)
            
            # If we still have an unprocessed system message (no user messages followed it),
            # add it as a user message
            if has_system_message and system_content:
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": system_content}]
                })
            
            logger.debug(f"Sending {len(gemini_messages)} messages to Gemini API")
            
            # Configure tool calling if tools are provided
            if tools:
                # Convert tools to the format expected by the Gemini API
                function_declarations = []
                
                for tool in tools:
                    # Ensure tool has the required fields
                    function_declaration = {
                        "name": tool["name"],
                        "description": tool.get("description", "")
                    }
                    
                    # Ensure parameters have the required structure
                    parameters = tool.get("parameters", {})
                    if "type" not in parameters:
                        parameters["type"] = "object"
                    
                    # Ensure properties exists
                    if "properties" not in parameters:
                        parameters["properties"] = {}
                        
                    function_declaration["parameters"] = parameters
                    
                    function_declarations.append(function_declaration)
                    logger.debug(f"Configured tool: {tool['name']}")
                    logger.debug(f"Tool parameters: {json.dumps(tool.get('parameters', {}), indent=2)}")
                
                logger.debug(f"Configured {len(function_declarations)} tools for Gemini API")
                logger.debug(f"Function declarations: {json.dumps(function_declarations, indent=2)}")
                
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
            
            logger.debug(f"Received response from Gemini API: {response}")
            
            # Extract the response text first
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'parts'):
                parts = response.parts
                if parts and len(parts) > 0:
                    response_text = parts[0].text
            # Handle candidates if present
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                        response_text = parts[0].text
            
            # Check for custom tool_code format in the response text
            function_calls = []
            
            # Check for ```tool_code ... ``` blocks in the response text
            import re
            tool_code_pattern = r'```tool_code\s*(.*?)\s*```'
            tool_code_matches = re.findall(tool_code_pattern, response_text, re.DOTALL)
            
            if tool_code_matches:
                logger.debug(f"Found {len(tool_code_matches)} tool_code blocks in response")
                
                for tool_code in tool_code_matches:
                    # Parse the tool call - expected format: tool_name(param1="value1", param2="value2")
                    tool_pattern = r'(\w+)\s*\((.*)\)'
                    tool_match = re.match(tool_pattern, tool_code.strip())
                    
                    if tool_match:
                        tool_name = tool_match.group(1)
                        args_str = tool_match.group(2)
                        
                        # Parse arguments
                        args_dict = {}
                        try:
                            # Try to evaluate the arguments as a Python dict
                            # This is safe because we're only parsing simple key-value pairs
                            from ast import literal_eval
                            args_dict = {k.strip(): v for k, v in [arg.split("=", 1) for arg in args_str.split(",") if "=" in arg]}
                            
                            # Clean up the values (remove quotes)
                            for k, v in args_dict.items():
                                if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
                                    args_dict[k] = v[1:-1]
                                elif isinstance(v, str) and v.startswith("'") and v.endswith("'"):
                                    args_dict[k] = v[1:-1]
                        except Exception as e:
                            logger.warning(f"Error parsing tool arguments: {e}")
                        
                        # Create function call
                        tool_call = {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": args_dict
                            }
                        }
                        function_calls.append(tool_call)
            
            # Also check for function calls in the standard Gemini format
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        # Check for function calls in the content
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            logger.debug(f"Response has {len(candidate.content.parts)} parts")
                            for i, part in enumerate(candidate.content.parts):
                                logger.debug(f"Examining part {i}: {type(part)}")
                                if hasattr(part, 'function_call'):
                                    function_call = part.function_call
                                    logger.debug(f"Found function call: name={function_call.name}, args type={type(function_call.args)}")
                                    logger.debug(f"Function call args: {function_call.args}")
                                    
                                    # Only process valid function calls with a name
                                    if hasattr(function_call, 'name') and function_call.name:
                                        # Convert function call to the expected format
                                        
                                        # Parse arguments - they might be a string that needs to be converted to JSON
                                        arguments = function_call.args
                                        if isinstance(arguments, str):
                                            try:
                                                # Try to parse as JSON if it's a string
                                                arguments = json.loads(arguments)
                                            except json.JSONDecodeError:
                                                # If not valid JSON, keep as is
                                                logger.warning(f"Could not parse function arguments as JSON: {arguments}")
                                        
                                        tool_call = {
                                            "id": str(uuid.uuid4()),
                                            "type": "function",
                                            "function": {
                                                "name": function_call.name,
                                                "arguments": arguments
                                            }
                                        }
                                        function_calls.append(tool_call)
            
            # If we have valid function calls, return them
            if function_calls:
                logger.info(f"Extracted {len(function_calls)} function calls from Gemini response")
                return {
                    "content": "",
                    "tool_calls": function_calls
                }
            
            # If no valid function calls were found, return the text response
            if response_text:
                return response_text
            
            # Fallback to string representation
            return str(response)
        
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            logger.exception("Exception details:")
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
            try:
                # Extract the function call details
                function_call = tool_call.get("function", {})
                tool_name = function_call.get("name", "")
                arguments = function_call.get("arguments", "{}")
                
                # Ensure arguments is a string (JSON)
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments)
                    except Exception as e:
                        logger.warning(f"Failed to convert arguments to JSON string: {e}")
                        arguments = str(arguments)
                
                logger.debug(f"Processing tool call: {tool_name} with arguments: {arguments}")
                
                # Create a tool call object in the format expected by the tool executor
                formatted_tool_call = {
                    "id": tool_call.get("id", str(uuid.uuid4())),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
                
                # Execute the tool call
                async for tool_result in tool_executor(formatted_tool_call):
                    # Add the result to our list
                    results.append(tool_result)
                    logger.debug(f"Tool {tool_name} returned result: {tool_result.result}")
                    
            except Exception as e:
                logger.error(f"Error processing tool call: {e}")
                # Create an error result
                error_result = ToolResult(
                    tool_name=tool_call.get("function", {}).get("name", "unknown_tool"),
                    result=f"Error: {str(e)}",
                    error=True
                )
                results.append(error_result)
        
        return results
