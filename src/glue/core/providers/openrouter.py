"""
OpenRouter provider implementation for the GLUE framework.

This module contains the OpenRouter provider class that handles
communication with the OpenRouter API for model interactions.
OpenRouter provides a unified API for accessing various LLM providers,
including free models that are perfect for testing during development.
"""

import logging
import os
from typing import Dict, List, Any, Callable, AsyncIterable
import asyncio
import openai

from ..schemas import Message, ToolCall, ToolResult
from smolagents import OpenAIServerModel

# Set up logging
logger = logging.getLogger(__name__)

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
        """Initialize the OpenRouter client using SmolAgents OpenAIServerModel."""
        # Get API key from model config or environment
        api_key = getattr(self.model, "api_key", None) or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"Please set the environment variable '{OPENROUTER_API_KEY_ENV}' or provide api_key in the model config.")

        # Determine model ID to use
        default_model = "meta-llama/llama-3.1-8b-instruct:free"
        model_id = getattr(self.model, "model_name", getattr(self.model, "model", default_model))

        # Instantiate the SmolAgents OpenAIServerModel client
        self.client = OpenAIServerModel(
            model_id=model_id,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
        )
        logger.info(f"Initialized OpenRouter OpenAIServerModel client for model {model_id}")

    async def generate_response(
        self, messages: List[Message]
    ) -> Any:
        """Generate a response from the model.

        Args:
            messages: List of messages in the conversation

        Returns:
            The generated response or a dict with tool calls
        """
        # Convert messages to the format expected by OpenRouter
        openrouter_messages = []
        for msg in messages:
            # Handle different message formats
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                raw_content = msg.get("content", "")
                # If content is a list of text segments, flatten into a single string
                if isinstance(raw_content, list):
                    content = "".join(
                        [segment.get("text", "") for segment in raw_content if isinstance(segment, dict)]
                    )
                else:
                    content = raw_content
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
        # Determine the final model name to use
        model_name_to_use = getattr(self.model, "model_name", getattr(self.model, "model", default_model))
        # Base API params
        api_params: Dict[str, Any] = {
            "model": model_name_to_use,
            "messages": openrouter_messages,
        }
        # Merge in model-specific API params (allow overrides)
        base_api_params = getattr(self.model, "api_params", {}) or {}
        api_params.update(base_api_params)
        # Add temperature if defined and not already overridden
        temp = getattr(self.model, "temperature", None)
        if temp is not None and "temperature" not in api_params:
            api_params["temperature"] = temp
        # Add max_tokens if defined and not already overridden
        max_toks = getattr(self.model, "max_tokens", None)
        if max_toks is not None and "max_tokens" not in api_params:
            api_params["max_tokens"] = max_toks

        try:
            logger.debug(f"OpenrouterProvider INPUT messages: {openrouter_messages}")
            
            # Use underlying OpenAI-compatible client for chat if available
            openai_client = getattr(self.client, "client", None)
            if openai_client and hasattr(openai_client, "chat"):
                response = openai_client.chat.completions.create(**api_params)
            else:
                # Fallback: use SmolAgents model .generate()
                logger.debug("OpenrouterProvider using direct generate on OpenAIServerModel")
                # Remove 'model' arg since OpenAIServerModel.generate uses model_id internally
                direct_params = {k: v for k, v in api_params.items() if k != "model"}
                msg_obj = self.client.generate(**direct_params)
                # Extract string content
                if hasattr(msg_obj, "content"):
                    return msg_obj.content
                if isinstance(msg_obj, str):
                    return msg_obj
                # No valid content
                logger.error("OpenrouterProvider direct generate returned no content")
                return "ERROR_IN_OPENROUTER_PROVIDER_NO_CONTENT"

            logger.debug(f"OpenrouterProvider RAW API RESPONSE: {response}")

            # Process the response and return non-empty content
            choice = None
            if response and getattr(response, "choices", None):
                choice = response.choices[0]
            msg = getattr(choice, "message", None)
            if msg and hasattr(msg, "content") and msg.content and msg.content.strip():
                content = msg.content
                # Ignore structured tool call outputs
                if content.strip().startswith("Calling tools:"):
                    logger.warning("OpenrouterProvider ignoring structured tool call content")
                else:
                    logger.debug(f"OpenrouterProvider RETURNING content: {content}")
                    return content
            
            # Warm-up retry for empty content when finish_reason is None
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is None:
                logger.warning("OpenRouter response had no finish reason and empty content; retrying after delay")
                await asyncio.sleep(2)
                response_warmup = self.client.chat.completions.create(**api_params)
                logger.debug(f"OpenrouterProvider RAW API RESPONSE (warm-up retry): {response_warmup}")
                choice_warmup = None
                if response_warmup and getattr(response_warmup, "choices", None):
                    choice_warmup = response_warmup.choices[0]
                msg_warmup = getattr(choice_warmup, "message", None)
                if msg_warmup and hasattr(msg_warmup, "content") and msg_warmup.content and msg_warmup.content.strip():
                    content_warmup = msg_warmup.content
                    # Ignore structured tool call outputs
                    if content_warmup.strip().startswith("Calling tools:"):
                        logger.warning("OpenrouterProvider ignoring structured tool call content in warm-up retry")
                    else:
                        logger.debug(f"OpenrouterProvider RETURNING content (warm-up retry): {content_warmup}")
                        return content_warmup
            # Retry once on empty content
            logger.warning("OpenRouter response contained empty content; retrying once")
            response_retry = self.client.chat.completions.create(**api_params)
            logger.debug(f"OpenrouterProvider RAW API RESPONSE (retry): {response_retry}")
            choice_retry = None
            if response_retry and getattr(response_retry, "choices", None):
                choice_retry = response_retry.choices[0]
            msg_retry = getattr(choice_retry, "message", None)
            if msg_retry and hasattr(msg_retry, "content") and msg_retry.content and msg_retry.content.strip():
                content_retry = msg_retry.content
                # Ignore structured tool call outputs
                if content_retry.strip().startswith("Calling tools:"):
                    logger.warning("OpenrouterProvider ignoring structured tool call content in retry")
                else:
                    logger.debug(f"OpenrouterProvider RETURNING content (retry): {content_retry}")
                    return content_retry
            logger.error("OpenRouter response contained empty content after retry; returning error code")
            return "ERROR:LLM_EMPTY_CONTENT_NULL"
            
        except openai.NotFoundError as e:
            logger.error(f"Error generating response from OpenRouter: {e}", exc_info=True)
            return f"ERROR_IN_OPENROUTER_PROVIDER_NOTFOUND: {type(e).__name__}: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}", exc_info=True)
            logger.error("OpenRouter generic exception; RETURNING ERROR MESSAGE STRING")
            return f"ERROR_IN_OPENROUTER_PROVIDER: {type(e).__name__}: {str(e)}"

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

    async def close(self):
        """Close the underlying HTTP client to avoid pending tasks on shutdown."""
        try:
            http_client = getattr(self.client, "_client", None)
            if http_client:
                logger.info(
                    f"Closing OpenRouter httpx client for model {getattr(self.model, 'model_name', 'unknown')}"
                )
                await http_client.aclose()  # type: ignore
        except Exception as e:
            logger.error(f"Error closing OpenRouter client: {e}", exc_info=True)
        finally:
            self.client = None