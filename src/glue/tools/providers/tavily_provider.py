"""
Tavily search provider for the GLUE framework.

This module implements a search provider using the Tavily AI search API.
"""

import aiohttp
import logging
from typing import Dict, Any, Optional

from .search_base import SearchProvider, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class TavilyProvider(SearchProvider):
    """Search provider using the Tavily AI search API"""

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Tavily provider.

        Args:
            api_key: Tavily API key
            config: Optional additional configuration
        """
        super().__init__("tavily", config)
        self.api_key = api_key

        # Default configuration
        self.default_config = {
            "search_depth": "basic",  # basic or comprehensive
            "include_domains": [],
            "exclude_domains": [],
            "include_answer": False,
            "max_tokens": 500,
        }

        # Merge default config with provided config
        if config:
            self.default_config.update(config)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        Execute a search query using Tavily API.

        Args:
            query: The search query to execute
            max_results: Maximum number of results to return

        Returns:
            A SearchResponse object containing the search results
        """
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}

        payload = {"query": query, "max_results": max_results, **self.default_config}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"Tavily API error: {response.status} - {await response.text()}"
                        )
                        return SearchResponse(
                            query=query, results=[], total_results=0, provider=self.name
                        )

                    data = await response.json()
                    return self._parse_response(query, data, max_results)
        except Exception as e:
            logger.error(f"Error during Tavily search: {str(e)}")
            return SearchResponse(
                query=query, results=[], total_results=0, provider=self.name
            )

    def _parse_response(
        self, query: str, data: Dict[str, Any], max_results: int
    ) -> SearchResponse:
        """
        Parse the Tavily API response into a SearchResponse object.

        Args:
            query: The original search query
            data: The response data from Tavily API
            max_results: Maximum number of results to include

        Returns:
            A SearchResponse object
        """
        results = []

        # Extract results
        search_results = data.get("results", [])

        for i, result in enumerate(search_results[:max_results]):
            results.append(
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    position=i + 1,
                )
            )

        # Tavily doesn't provide total_results, so we use the length of results
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(search_results),
            provider=self.name,
        )
