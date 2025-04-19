"""
Provider base class for the GLUE framework.

This module provides a base class for all model providers in the GLUE framework.
It defines the common interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable

from ..schemas import Message, ToolCall, ToolResult


class ProviderBase(ABC):
    """Base class for all model providers in the GLUE framework."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize a new provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config or {}
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.client = None
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the provider client.
        
        This method should be overridden by provider implementations
        to initialize their specific client.
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate a response from the model.
        
        Args:
            messages: List of messages in the conversation
            tools: Optional list of tools available to the model
            
        Returns:
            The generated response
        """
        pass
    
    async def process_tool_calls(
        self, 
        tool_calls: List[ToolCall], 
        tool_executor: Callable[[ToolCall], ToolResult]
    ) -> List[ToolResult]:
        """Process tool calls and execute them.
        
        Args:
            tool_calls: List of tool calls to process
            tool_executor: Function to execute a tool call
            
        Returns:
            List of tool results
        """
        results = []
        for tool_call in tool_calls:
            result = await tool_executor(tool_call)
            results.append(result)
        return results
