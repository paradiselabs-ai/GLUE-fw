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

from rich.console import Console
from ..schemas import Message, ToolCall, ToolResult
from glue.utils.ui_utils import display_warning

# Set up logging
logger = logging.getLogger("glue.model.openrouter")

# Environment variable name for the API key
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# Create a console instance for warnings
_console = Console()


class OpenrouterProvider:
    """Provider implementation for OpenRouter models.

    OpenRouter provides a unified API for accessing various LLM providers,
    including free models that are perfect for testing during development.
    """

    # Class-level cache of models that don't support tool use
    _models_without_tool_support = set()

    def __init__(self, model):
        """Initialize a new OpenRouter provider.

        Args:
            model: The model using this provider
        """
        self.model = model
        self.client = None
        self._simulated_instructions_added_this_request = False
        self._initialize_client()

    @classmethod
    def clear_capability_cache(cls):
        """Clear the cached model capabilities."""
        cls._models_without_tool_support.clear()
        logger.info("Cleared model capability cache")

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
            model_name = getattr(
                self.model, "model_name", getattr(self.model, "model", "unknown")
            )
            logger.info(f"Initialized OpenRouter client for model {model_name}")
        except ImportError:
            logger.error(
                "Failed to import OpenAI client. Install with: pip install openai"
            )
            # Create a mock client for testing
            self.client = type(
                "MockClient",
                (),
                {
                    "chat": type(
                        "MockChat",
                        (),
                        {"completions": type("MockCompletions", (), {"create": None})},
                    )
                },
            )

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
            is_development = getattr(self.model, "development", False)

            if is_development:
                logger.warning(
                    "No OpenRouter API key provided but running in development mode. Using dummy key."
                )
                return "dummy_key_for_development"
            else:
                raise ValueError(
                    f"No OpenRouter API key found. Set {OPENROUTER_API_KEY_ENV} environment variable or provide it in the model config."
                )

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

    def _add_simulated_tool_use_instructions(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]
    ) -> None:
        """Add simulated tool use instructions to the messages.

        Args:
            messages: List of messages to modify
            tools: List of tools to include in the instructions
        """
        # Find the system message if it exists
        system_msg_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_msg_idx = i
                break

        # Create the tool simulation instructions
        tool_simulation_instructions = "\n\n## ACTION REQUIRED: TOOL USE VIA JSON\n"
        tool_simulation_instructions += "This model interface does not support native tool calls. **Your ability to use tools depends entirely on following the specific format below.**\n\n"
        tool_simulation_instructions += "To execute a tool, you **MUST** respond with **ONLY** a single valid JSON object formatted **EXACTLY** like this:\n\n"
        tool_simulation_instructions += "```json\n"
        tool_simulation_instructions += "{\n"
        tool_simulation_instructions += '  "tool_name": "<name_of_tool>",\n'
        tool_simulation_instructions += '  "arguments": { <parameters_object> }\n'
        tool_simulation_instructions += "}\n"
        tool_simulation_instructions += "```\n\n"
        tool_simulation_instructions += "Replace `<name_of_tool>` with the exact name of the tool you want to execute from the list below.\n"
        tool_simulation_instructions += 'Replace `<parameters_object>` with a valid JSON object containing the arguments for the tool (e.g., `{"target_type": "model", "target_name": "assistant", "message": "Hello!"}`).\n\n'
        tool_simulation_instructions += "**CRITICAL REQUIREMENT:** Your response **MUST** start *immediately* with the opening curly brace `{` of the JSON object. The very first character of your output must be `{`.\n"
        tool_simulation_instructions += (
            "- End your response immediately with the closing curly brace `}`.\n"
        )
        tool_simulation_instructions += "- No other text, explanation, or formatting before or after the JSON is permitted.\n"
        tool_simulation_instructions += (
            "- Failure to start with `{` means the tool call will **fail**.\n"
        )

        tool_simulation_instructions += "Available tools:\n"
        for tool in tools:
            tool_name = tool.get("name", "unknown_tool")
            tool_description = tool.get("description", "No description available.")
            tool_desc = f"- {tool_name}: {tool_description}\n"
            parameters_schema = tool.get("parameters")
            if (
                isinstance(parameters_schema, dict)
                and "properties" in parameters_schema
            ):
                tool_desc += "  Parameters:\n"
                props = parameters_schema.get("properties", {})
                required_params = parameters_schema.get("required", [])
                for param_name, param_info in props.items():
                    if isinstance(param_info, dict):
                        req_marker = (
                            " (required)" if param_name in required_params else ""
                        )
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        tool_desc += f"    - {param_name}{req_marker} ({param_type}): {param_desc}\n"
                    else:
                        tool_desc += f"    - {param_name}: (details unavailable)\n"
            tool_simulation_instructions += tool_desc

        # Add the instructions to the system message or create a new one
        if system_msg_idx is not None:
            messages[system_msg_idx]["content"] += tool_simulation_instructions
        else:
            messages.insert(
                0, {"role": "system", "content": tool_simulation_instructions}
            )

    async def generate_response(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Generate a response from the model.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model

        Returns:
            The generated response or a dict with tool calls
        """
        # Reset the flag at the beginning of each generation request
        self._simulated_instructions_added_this_request = False

        # Convert messages to the format expected by OpenRouter
        openrouter_messages = []
        for msg in messages:
            # Handle different message formats
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
            else:
                # Fallback for string or other types
                role = "user"
                content = str(msg)

            openrouter_messages.append({"role": role, "content": content})

        # Prepare the API call parameters
        # Use a valid default model for OpenRouter if none is specified
        default_model = "meta-llama/llama-3.1-8b-instruct:free"
        api_params = {
            "model": getattr(
                self.model, "model_name", getattr(self.model, "model", default_model)
            ),
            "messages": openrouter_messages,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            **self.model.api_params,
        }

        # Add tools if provided
        tools_originally_present = bool(tools)  # Track if tools were passed initially

        # Get the model name for cache lookup
        model_name = api_params["model"]

        # Check if this model is known to not support tool use
        model_in_cache = model_name in self._models_without_tool_support

        # If tools are provided and the model is in the cache as not supporting tool use,
        # skip the initial API call and go straight to simulated tool use
        if tools and model_in_cache:
            logger.info(
                f"Model {model_name} is known to not support tool use. Using simulated tool use instructions directly."
            )
            # Skip adding tools to API params
            if "tools" in api_params:
                api_params.pop("tools")
            if "tool_choice" in api_params:
                api_params.pop("tool_choice")

            # Add simulated tool use instructions to the system message
            self._add_simulated_tool_use_instructions(openrouter_messages, tools)

            # Update the messages in the API params
            api_params["messages"] = openrouter_messages

            # Set the flag indicating instructions were added
            self._simulated_instructions_added_this_request = True
        elif tools:
            # If tools are provided and the model is not in the cache, add them to the API params
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        try:
            # Call the OpenRouter API
            response = await self.client.chat.completions.create(**api_params)

            # Process the response
            choice = None
            if response and getattr(response, "choices", None):
                choice = response.choices[0]
            msg = getattr(choice, "message", None)
            if msg and getattr(msg, "tool_calls", None):
                tool_calls = []
                for tool_call in msg.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": arguments,
                        }
                    )
                return {"tool_calls": tool_calls}
            if msg and hasattr(msg, "content"):
                return msg.content
            logger.error(
                "OpenRouter response has no message content; returning empty string"
            )
            return ""
        except openai.NotFoundError as e:
            # Check the specific error, if tools were originally present, AND if instructions haven't been added yet
            if (
                tools_originally_present
                and "No endpoints found that support tool use" in str(e)
                and not self._simulated_instructions_added_this_request
            ):
                #  Display warning using the warning function with a console instance
                display_warning(
                    _console,  # Pass the console instance
                    f"Model {model_name} does not support tool use on OpenRouter. "
                    f"Retrying with simulated tool use instructions.",
                )

                # Add the model to the cache of models without tool support
                if model_name not in self._models_without_tool_support:
                    self._models_without_tool_support.add(model_name)
                    logger.info(
                        f"Added model {model_name} to the cache of models without tool support"
                    )

                # Remove tool parameters for the retry
                api_params.pop("tools", None)
                api_params.pop("tool_choice", None)

                # Add simulated tool use instructions
                self._add_simulated_tool_use_instructions(openrouter_messages, tools)

                # Set the flag indicating instructions were added
                self._simulated_instructions_added_this_request = True

                # Retry with the modified messages
                api_params["messages"] = openrouter_messages
                logger.debug("Retrying API call with simulated tool instructions.")
                response = await self.client.chat.completions.create(**api_params)
            else:
                # If it's a different error, tools weren't present, or instructions already added, re-raise
                logger.error(
                    f"Error generating response from OpenRouter (NotFoundError not retryable or already retried): {e}"
                )
                raise
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}")
            raise

        # Process the response (from original call or retry)
        choice = None
        if response and getattr(response, "choices", None):
            choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if msg and getattr(msg, "tool_calls", None):
            tool_calls = []
            for tool_call in msg.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    }
                )
            return {"tool_calls": tool_calls}
        if msg and hasattr(msg, "content"):
            return msg.content
        logger.error(
            "OpenRouter retry response has no message content; returning empty string"
        )
        return ""

    async def process_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tool_executor: Callable[[ToolCall], AsyncIterable[ToolResult]],
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
                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        content=f"Error executing tool: {str(e)}",
                    )
                )

        return results
