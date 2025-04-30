"""
Web search tool for the GLUE framework.

This module implements a web search tool that can use different search providers
to search the web for information.
"""

import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
import os

from .tool_base import Tool, ToolConfig, ToolPermission
from .providers.search_base import SearchProvider
from .providers.serp_provider import SerpApiProvider
from .providers.tavily_provider import TavilyProvider

logger = logging.getLogger(__name__)


class SearchProviderType(str, Enum):
    """Types of supported search providers"""

    SERP = "serp"
    TAVILY = "tavily"


class WebSearchTool(Tool):
    """Tool for searching the web using various providers"""

    description = "Search the web for information using Serp API"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"}
        },
        "required": ["query"],
    }

    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information",
        provider_type: Union[SearchProviderType, str] = SearchProviderType.SERP,
        provider_config: Optional[Dict[str, Any]] = None,
        config: Optional[ToolConfig] = None,
    ):
        """
        Initialize the web search tool.

        Args:
            name: Name of the tool
            description: Description of the tool
            provider_type: Type of search provider to use
            provider_config: Configuration for the search provider
            config: Tool configuration
        """
        # Create default tool config if not provided
        if config is None:
            config = ToolConfig(required_permissions={ToolPermission.NETWORK})

        super().__init__(name, description, config)

        # Explicitly define input parameters for LLM schema generation
        # Overrides introspection of _execute if Tool base class respects this
        self.inputs = {
            "query": {
                "type": str,
                "description": "The search query string.",
                "required": True,
            },
            "max_results": {
                "type": int,
                "description": "Maximum number of results to return.",
                "required": False,
                "default": 5,
            },
        }

        # Convert string to enum if needed
        if isinstance(provider_type, str):
            try:
                self.provider_type = SearchProviderType(provider_type.lower())
            except ValueError:
                self.provider_type = provider_type  # Keep as string for error handling
        else:
            self.provider_type = provider_type

        # Extract API key from config for test compatibility
        self.api_key = None
        if provider_config and "api_key" in provider_config:
            self.api_key = provider_config.get("api_key")
        elif config and isinstance(config, dict) and "metadata" in config:
            self.api_key = config.get("metadata", {}).get("api_key")

        self.provider_config = provider_config or {}
        self.provider: Optional[SearchProvider] = None

    async def initialize(self, instance_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the tool and create the appropriate search provider.

        Args:
            instance_data: Optional instance data for initialization
        """
        if not self._initialized:
            # Initialize the provider based on the provider type
            self._create_provider()

            # Call the parent initialize method
            await super().initialize(instance_data)

    async def _execute(
        self, query: str, num_results: Optional[int] = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a web search.

        Args:
            query: The search query string.
            num_results: The maximum number of results to return (default 5).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Dictionary containing the search results

        Raises:
            ValueError: If the provider is not initialized or query is invalid.
        """
        if not self.provider:
            raise ValueError("Search provider not initialized")

        # Validate query (already typed as str)
        if not query:
            raise ValueError("Search query cannot be empty.")

        # Validate max_results (use num_results parameter)
        max_results = (
            num_results if num_results is not None else 5
        )  # Use default if None
        try:
            max_results = int(max_results)
            if max_results <= 0:
                logger.warning("num_results must be positive. Using default 5.")
                max_results = 5
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid num_results value: {num_results}. Using default 5."
            )
            max_results = 5

        # Execute the search using the direct parameters
        logger.debug(
            f"Executing web search for '{query}' with max_results={max_results}"
        )
        response = await self.provider.search(query, max_results)

        # Return the response as a dictionary
        return response.model_dump()

    def _create_provider(self) -> None:
        """
        Create the appropriate search provider based on the provider type.

        Raises:
            ValueError: If the API key is missing or the provider type is not supported
        """
        # Check if provider type is valid
        if not isinstance(self.provider_type, SearchProviderType):
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

        # Check for API key - First from config, then environment
        api_key = self.provider_config.get("api_key")
        if not api_key:
            # Try getting from environment variable if not in config
            api_key = os.getenv("SERP_API_KEY")

        if not api_key:
            # Raise error only if key is missing from both config and environment
            raise ValueError(
                f"API key is required for {self.provider_type} provider. Provide it in the tool config or set the SERP_API_KEY environment variable."
            )

        # Create the appropriate provider
        # Extract all config except api_key to pass as extra_config
        extra_config = {k: v for k, v in self.provider_config.items() if k != "api_key"}

        if self.provider_type == SearchProviderType.SERP:
            self.provider = SerpApiProvider(api_key, extra_config)
        elif self.provider_type == SearchProviderType.TAVILY:
            # Note: Tavily might use a different env var name, e.g., TAVILY_API_KEY
            # For now, we assume SERP_API_KEY covers the main case
            self.provider = TavilyProvider(api_key, extra_config)
        else:
            # This should never happen due to the check above
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

        logger.info(f"Created {self.provider_type} search provider")

    async def cleanup(self) -> None:
        """Clean up the tool resources"""
        self.provider = None
        await super().cleanup()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary.

        Returns:
            A dictionary representation of the tool
        """
        result = super().to_dict()
        result.update(
            {
                "provider_type": self.provider_type,
                "provider_config": {
                    k: v for k, v in self.provider_config.items() if k != "api_key"
                },
            }
        )
        return result
