"""
Portkey.ai integration for GLUE framework.

This module provides utilities for integrating with Portkey.ai for API key management,
usage tracking, and cost optimization for AI services.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import aiohttp


@dataclass
class PortkeyConfig:
    """Configuration for Portkey.ai integration."""

    api_key: str
    base_url: str = "https://api.portkey.ai/v1"
    trace_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class PortkeyClient:
    """Client for interacting with Portkey.ai."""

    def __init__(self, config: Optional[PortkeyConfig] = None):
        """Initialize the Portkey client.

        Args:
            config: Portkey configuration. If None, will attempt to load from environment.
        """
        if config is None:
            api_key = os.environ.get("PORTKEY_API_KEY")
            if not api_key:
                raise ValueError(
                    "Portkey API key not provided and PORTKEY_API_KEY environment variable not set"
                )
            config = PortkeyConfig(api_key=api_key)

        self.config = config

    async def get_headers(
        self, additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Get headers for Portkey API requests.

        Args:
            additional_headers: Additional headers to include.

        Returns:
            Headers dictionary.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "X-Portkey-Mode": "proxy",
        }

        if self.config.trace_id:
            headers["X-Portkey-Trace-Id"] = self.config.trace_id

        if self.config.tags:
            headers["X-Portkey-Tags"] = ",".join(
                f"{k}={v}" for k, v in self.config.tags.items()
            )

        if additional_headers:
            headers.update(additional_headers)

        return headers

    async def proxy_request(
        self,
        provider: str,
        endpoint: str,
        payload: Dict[str, Any],
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Proxy a request through Portkey to an AI provider.

        Args:
            provider: AI provider (e.g., "openai", "anthropic")
            endpoint: API endpoint (e.g., "chat/completions")
            payload: Request payload
            additional_headers: Additional headers to include

        Returns:
            Response from the AI provider
        """
        headers = await self.get_headers(additional_headers)
        url = f"{self.config.base_url}/{provider}/{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    async def get_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get usage statistics from Portkey.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Usage statistics
        """
        headers = await self.get_headers()
        url = (
            f"{self.config.base_url}/usage?start_date={start_date}&end_date={end_date}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()


def get_portkey_client() -> PortkeyClient:
    """Get a Portkey client instance.

    Returns:
        PortkeyClient instance
    """
    return PortkeyClient()
