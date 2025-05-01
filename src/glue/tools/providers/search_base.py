"""
Base classes for search providers in the GLUE framework.

This module defines the base classes and interfaces for implementing search providers
in the GLUE framework. Search providers allow agents to search the web for information.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class SearchResult(BaseModel):
    """Model for a search result"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "GLUE Framework Documentation",
                "url": "https://example.com/glue-docs",
                "snippet": "Official documentation for the GLUE framework for AI development.",
                "position": 1,
            }
        }
    )

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Snippet or description of the search result")
    position: int = Field(
        ..., description="Position of the result in the search results"
    )


class SearchResponse(BaseModel):
    """Model for a search response"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "GLUE framework for AI",
                "results": [],
                "total_results": 42,
                "provider": "example_search",
            }
        }
    )

    query: str = Field(..., description="The search query that was executed")
    results: List[SearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    total_results: int = Field(0, description="Total number of results available")
    provider: str = Field(..., description="Name of the search provider used")


class SearchProvider(ABC):
    """Base class for search providers"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a search provider.

        Args:
            name: Name of the search provider
            config: Optional configuration for the provider
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        Execute a search query.

        Args:
            query: The search query to execute
            max_results: Maximum number of results to return

        Returns:
            A SearchResponse object containing the search results
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the provider to a dictionary.

        Returns:
            A dictionary representation of the provider
        """
        return {
            "name": self.name,
            "type": "search_provider",
            "config": {k: v for k, v in self.config.items() if k != "api_key"},
        }
