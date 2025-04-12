"""
Web search tool for the GLUE framework.

This module implements a web search tool that can use different search providers
to search the web for information.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

from .tool_base import ToolConfig, ToolPermission
from .pydantic_validated_tool import PydanticValidatedTool
from .providers.search_base import SearchProvider, SearchResponse
from .providers.serp_provider import SerpApiProvider
from .providers.tavily_provider import TavilyProvider

logger = logging.getLogger(__name__)

class SearchProviderType(str, Enum):
    """Types of supported search providers"""
    SERP = "serp"
    TAVILY = "tavily"

class WebSearchInput(BaseModel):
    """Input schema for web search tool"""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class WebSearchOutput(BaseModel):
    """Output schema for web search tool"""
    results: List[Dict[str, Any]] = Field(description="Search results")
    query: str = Field(description="Search query that was executed")
    provider: str = Field(description="Search provider used")

class WebSearchTool(PydanticValidatedTool):
    """Tool for searching the web using various providers"""
    
    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information",
        provider_type: Union[SearchProviderType, str] = SearchProviderType.SERP,
        provider_config: Optional[Dict[str, Any]] = None,
        config: Optional[ToolConfig] = None
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
            config = ToolConfig(
                required_permissions={ToolPermission.NETWORK}
            )
        
        # Initialize the PydanticValidatedTool with our schemas
        super().__init__(
            name, 
            description, 
            input_schema=WebSearchInput,
            output_schema=WebSearchOutput,
            config=config
        )
        
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
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            input_data: Dictionary containing 'query' and optional 'max_results'
            
        Returns:
            Dictionary containing the search results
            
        Raises:
            ValueError: If the query is missing or the provider is not initialized
        """
        if not self.provider:
            raise ValueError("Search provider not initialized")
        
        # Input validation is already handled by PydanticValidatedTool
        query = input_data["query"]
        max_results = input_data["max_results"]
        
        # Execute the search
        response = await self.provider.search(query, max_results)
        
        # Return the response as a dictionary
        return {
            "results": response.results,
            "query": query,
            "provider": self.provider_type.value if isinstance(self.provider_type, SearchProviderType) else str(self.provider_type)
        }
    
    def _create_provider(self) -> None:
        """
        Create the appropriate search provider based on the provider type.
        
        Raises:
            ValueError: If the API key is missing or the provider type is not supported
        """
        # Check if provider type is valid
        if not isinstance(self.provider_type, SearchProviderType):
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
            
        # --- START MODIFICATION: Load API Key --- 
        api_key = self.api_key # Start with potentially pre-configured key

        if not api_key:
            # Determine environment variable name based on provider type
            if self.provider_type == SearchProviderType.SERP:
                env_var_name = "SERPAPI_API_KEY"
            elif self.provider_type == SearchProviderType.TAVILY:
                env_var_name = "TAVILY_API_KEY"
            else:
                # Should not happen due to the check above, but safeguard
                raise ValueError(f"Logic error: Unknown provider type {self.provider_type} for API key lookup.")
            
            # Attempt to load from environment
            api_key = os.getenv(env_var_name)
            logger.info(f"Attempted to load API key from environment variable: {env_var_name}")

        # Check if API key is available after checking config and environment
        if not api_key:
            raise ValueError(f"API key for {self.provider_type.value} provider is required. Set via config or environment variable {env_var_name}.")
        # --- END MODIFICATION ---
        
        # Create the provider instance
        if self.provider_type == SearchProviderType.SERP:
            self.provider = SerpApiProvider(api_key, self.provider_config)
            logger.info("Using SerpApiProvider")
        elif self.provider_type == SearchProviderType.TAVILY:
            self.provider = TavilyProvider(api_key, self.provider_config)
            logger.info("Using TavilyProvider")
        # No else needed due to the check at the beginning

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
        tool_dict = super().to_dict()
        tool_dict.update({
            "provider_type": self.provider_type.value if isinstance(self.provider_type, SearchProviderType) else str(self.provider_type),
            "provider_config": self.provider_config
        })
        return tool_dict
