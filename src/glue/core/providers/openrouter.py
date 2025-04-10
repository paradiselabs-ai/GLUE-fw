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
import openai

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
            model_name = getattr(self.model, 'model_name', getattr(self.model, 'model', 'unknown'))
            logger.info(f"Initialized OpenRouter client for model {model_name}")
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
        openrouter_messages = []
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
            
            openrouter_messages.append({"role": role, "content": content})
        
        # Prepare the API call parameters
        # Use a valid default model for OpenRouter if none is specified
        default_model = "meta-llama/llama-3.1-8b-instruct:free"
        api_params = {
            "model": getattr(self.model, 'model_name', getattr(self.model, 'model', default_model)),
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
        except openai.NotFoundError as e:
            # Check if the error is the specific one about tool use incompatibility
            if "No endpoints found that support tool use" in str(e):
                logger.warning(
                    f"Model {self.model.model_name} does not support tool use on OpenRouter. "
                    f"Retrying with simulated tool use instructions."
                )
                # Remove tool parameters
                api_params.pop("tools", None)
                api_params.pop("tool_choice", None)
                
                # Add simulated tool use instructions to the system message
                if tools:
                    # Find or create a system message
                    system_msg_idx = None
                    for i, msg in enumerate(openrouter_messages):
                        if msg.get('role') == 'system':
                            system_msg_idx = i
                            break
                    
                    # Create tool simulation instructions using a parseable JSON format - MORE FORCEFUL
                    tool_simulation_instructions = "\n\n## ACTION REQUIRED: TOOL USE VIA JSON\n"
                    tool_simulation_instructions += "This model interface doesn't support native tool calls. To execute a tool, you MUST respond with ONLY a valid JSON object formatted EXACTLY like this, and nothing else:\n\n"
                    tool_simulation_instructions += "```json\n"
                    tool_simulation_instructions += "{\n"
                    tool_simulation_instructions += "  \"tool_name\": \"<name_of_tool>\",\n"
                    tool_simulation_instructions += "  \"arguments\": { <parameters_object> }\n"
                    tool_simulation_instructions += "}\n"
                    tool_simulation_instructions += "```\n\n"
                    tool_simulation_instructions += "Replace `<name_of_tool>` with the exact name of the tool you want to execute.\n"
                    tool_simulation_instructions += "Replace `<parameters_object>` with a valid JSON object containing the arguments for the tool (e.g., `{\"target_type\": \"model\", \"target_name\": \"assistant\", \"message\": \"Hello!\"}`).\n"
                    tool_simulation_instructions += "**CRITICAL: Your entire response must be ONLY this JSON object when calling a tool. Do not include any other text, explanation, or markdown formatting around the JSON.** If you are not calling a tool, respond normally.\n\n"

                    # Add tool descriptions using the likely internal format
                    tool_simulation_instructions += "Available tools:\n"
                    for tool in tools:
                        # Access name and description directly (assuming flatter structure)
                        tool_name = tool.get('name', 'unknown_tool')
                        tool_description = tool.get('description', 'No description available.')
                        tool_desc = f"- {tool_name}: {tool_description}\n"

                        # Access parameters if available (assuming 'parameters' key holds the schema)
                        parameters_schema = tool.get('parameters')
                        if isinstance(parameters_schema, dict) and 'properties' in parameters_schema:
                            tool_desc += "  Parameters:\n"
                            props = parameters_schema.get('properties', {})
                            required_params = parameters_schema.get('required', [])
                            for param_name, param_info in props.items():
                                if isinstance(param_info, dict):
                                    req_marker = " (required)" if param_name in required_params else ""
                                    param_type = param_info.get('type', 'any')
                                    param_desc = param_info.get('description', '')
                                    tool_desc += f"    - {param_name}{req_marker} ({param_type}): {param_desc}\n"
                                else:
                                     # Handle cases where param_info might not be a dict
                                     tool_desc += f"    - {param_name}: (details unavailable)\n"
                        tool_simulation_instructions += tool_desc
                    
                    # Add the instructions to the system message or create a new one
                    if system_msg_idx is not None:
                        openrouter_messages[system_msg_idx]['content'] += tool_simulation_instructions
                    else:
                        # Insert a system message at the beginning
                        openrouter_messages.insert(0, {
                            "role": "system",
                            "content": tool_simulation_instructions
                        })
                
                # Retry with the modified messages
                api_params["messages"] = openrouter_messages
                response = await self.client.chat.completions.create(**api_params)
            else:
                # If it's a different NotFoundError, re-raise it
                logger.error(f"Error generating response from OpenRouter (NotFound): {e}")
                raise
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}")
            raise
        
        # Process the response (moved the processing logic here to handle both original and retry responses)
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
