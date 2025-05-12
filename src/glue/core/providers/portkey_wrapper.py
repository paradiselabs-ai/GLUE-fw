"""
Portkey wrapper for model providers in the GLUE framework.

This module provides a wrapper for model providers that integrates with Portkey.ai
for API key management, usage tracking, and cost optimization.
"""

import os
import logging
from typing import Dict, List, Any, Optional, AsyncIterable

from ..schemas import Message
from glue.utils.portkey_client import get_portkey_client

# Set up logging
logger = logging.getLogger("glue.model.portkey")


class PortkeyProviderWrapper:
    """Wrapper for model providers that integrates with Portkey.ai."""

    PROVIDER_MAPPING = {
        "openai": "openai",
        "anthropic": "anthropic",
        "cohere": "cohere",
        # Add more provider mappings as needed
    }

    def __init__(self, provider, model_name: str, trace_id: Optional[str] = None):
        """Initialize a new Portkey provider wrapper.

        Args:
            provider: The provider to wrap
            model_name: Name of the model being used
            trace_id: Optional trace ID for request tracking
        """
        self.provider = provider
        self.model_name = model_name
        self.portkey_client = get_portkey_client()

        # Set trace ID if provided
        if trace_id:
            self.portkey_client.config.trace_id = trace_id

        # Add model info to tags
        self.portkey_client.config.tags.update(
            {"model": model_name, "framework": "glue", "version": "0.1.0-alpha"}
        )

        # Determine the provider type for Portkey
        provider_class_name = provider.__class__.__name__.lower()
        self.portkey_provider = None
        for key, value in self.PROVIDER_MAPPING.items():
            if key in provider_class_name:
                self.portkey_provider = value
                break

        if not self.portkey_provider:
            logger.warning(
                f"Could not determine Portkey provider for {provider_class_name}. "
                "Falling back to direct provider calls."
            )

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a response using the provider through Portkey.

        Args:
            messages: List of messages to send to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional arguments to pass to the provider

        Returns:
            Response from the model
        """
        if not self.portkey_provider:
            # Fall back to direct provider call if Portkey provider not determined
            return await self.provider.generate(messages, tools, **kwargs)

        # Prepare the payload based on the provider
        if self.portkey_provider == "openai":
            endpoint = "chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [msg.dict() for msg in messages],
                **kwargs,
            }
            if tools:
                payload["tools"] = tools

        elif self.portkey_provider == "anthropic":
            endpoint = "messages"
            payload = {
                "model": self.model_name,
                "messages": [msg.dict() for msg in messages],
                **kwargs,
            }
            if tools:
                payload["tools"] = tools

        else:
            # For other providers, customize as needed
            logger.warning(
                f"Portkey integration for {self.portkey_provider} not fully implemented"
            )
            return await self.provider.generate(messages, tools, **kwargs)

        try:
            # Make the request through Portkey
            response = await self.portkey_client.proxy_request(
                self.portkey_provider, endpoint, payload
            )
            return response
        except Exception as e:
            logger.error(f"Error using Portkey: {str(e)}")
            # Fall back to direct provider call
            return await self.provider.generate(messages, tools, **kwargs)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterable[Dict[str, Any]]:
        """Stream a response using the provider.

        Note: Currently falls back to the wrapped provider's stream method
        as Portkey streaming support is provider-specific.

        Args:
            messages: List of messages to send to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional arguments to pass to the provider

        Returns:
            Async iterable of response chunks
        """
        # For streaming, we currently fall back to the provider's implementation
        # as Portkey streaming support varies by provider
        return await self.provider.stream(messages, tools, **kwargs)


def wrap_provider(provider, model_name: str, trace_id: Optional[str] = None):
    """Wrap a provider with Portkey integration.

    Args:
        provider: The provider to wrap
        model_name: Name of the model being used
        trace_id: Optional trace ID for request tracking

    Returns:
        Wrapped provider with Portkey integration
    """
    # Only wrap if Portkey is enabled
    if os.environ.get("PORTKEY_ENABLED", "").lower() in ("true", "1", "yes"):
        return PortkeyProviderWrapper(provider, model_name, trace_id)
    return provider
