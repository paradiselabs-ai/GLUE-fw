"""
Mock provider for testing purposes.

This module provides a mock implementation of a model provider
that can be used in tests to simulate model behavior without
making actual API calls.
"""
from typing import Dict, List, Any, Optional
import asyncio

from ..types import Message, ModelConfig


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock provider with optional configuration."""
        self.config = config or {}
        self.messages: List[Message] = []
        self.response_queue: List[str] = []
        self.client = self  # Self-reference for compatibility with other providers
        
    async def send_message(self, message: Message) -> str:
        """
        Simulate sending a message to the model.
        
        Args:
            message: The message to send
            
        Returns:
            A mock response
        """
        self.messages.append(message)
        
        # Return a predefined response if available, otherwise generate one
        if self.response_queue:
            return self.response_queue.pop(0)
        
        # Generate a simple response based on the message content
        return f"Mock response to: {message.content[:20]}..."
    
    async def get_completion(self, prompt: str) -> str:
        """
        Get a completion for a prompt.
        
        Args:
            prompt: The prompt to complete
            
        Returns:
            A mock completion
        """
        # Return a predefined response if available, otherwise generate one
        if self.response_queue:
            return self.response_queue.pop(0)
        
        # Generate a simple completion
        return f"Mock completion for: {prompt[:20]}..."
    
    async def process_messages(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Process a list of messages and generate a response.
        
        Args:
            messages: The messages to process
            tools: Optional tools to use
            
        Returns:
            A response string
        """
        # Store the messages for inspection
        self.messages.extend(messages)
        
        # Generate a simple response
        if messages:
            last_message = messages[-1]
            return f"Mock response to: {last_message.content[:20]}..."
        return "No messages provided"
    
    def queue_response(self, response: str) -> None:
        """
        Queue a response to be returned by the next send_message or get_completion call.
        
        Args:
            response: The response to queue
        """
        self.response_queue.append(response)
    
    def clear_messages(self) -> None:
        """Clear all stored messages."""
        self.messages.clear()
        
    def clear_responses(self) -> None:
        """Clear all queued responses."""
        self.response_queue.clear()


def create_provider(config: ModelConfig) -> MockProvider:
    """
    Create a mock provider instance.
    
    Args:
        config: The model configuration
        
    Returns:
        A MockProvider instance
    """
    return MockProvider(config.dict() if hasattr(config, 'dict') else config.model_dump())
