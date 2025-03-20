"""
SERP API search provider for the GLUE framework.

This module implements a search provider using the SERP API service.
"""
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode

from .search_base import SearchProvider, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

class SerpApiProvider(SearchProvider):
    """Search provider using the SERP API service"""
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SERP API provider.
        
        Args:
            api_key: SERP API key
            config: Optional additional configuration
        """
        super().__init__("serp_api", config)
        self.api_key = api_key
        
        # Default configuration
        self.default_config = {
            "engine": "google",
            "gl": "us",  # Country to search from
            "hl": "en"   # Language
        }
        
        # Merge default config with provided config
        if config:
            self.default_config.update(config)
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        Execute a search query using SERP API.
        
        Args:
            query: The search query to execute
            max_results: Maximum number of results to return
            
        Returns:
            A SearchResponse object containing the search results
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": max_results
        }
        
        # Add default configuration
        params.update(self.default_config)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status != 200:
                        logger.error(f"SERP API error: {response.status} - {await response.text()}")
                        return SearchResponse(
                            query=query,
                            results=[],
                            total_results=0,
                            provider=self.name
                        )
                    
                    data = await response.json()
                    return self._parse_response(query, data, max_results)
        except Exception as e:
            logger.error(f"Error during SERP API search: {str(e)}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=self.name
            )
    
    def _parse_response(self, query: str, data: Dict[str, Any], max_results: int) -> SearchResponse:
        """
        Parse the SERP API response into a SearchResponse object.
        
        Args:
            query: The original search query
            data: The response data from SERP API
            max_results: Maximum number of results to include
            
        Returns:
            A SearchResponse object
        """
        results = []
        
        # Extract organic results
        organic_results = data.get("organic_results", [])
        total_results = int(data.get("search_information", {}).get("total_results", 0))
        
        for i, result in enumerate(organic_results[:max_results]):
            results.append(
                SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    position=i + 1
                )
            )
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=total_results,
            provider=self.name
        )
