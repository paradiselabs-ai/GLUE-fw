"""
Test provider for the GLUE framework.

This module provides a real implementation of a test provider
that can be used in tests without mocking.
"""
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from ..types import Message, ModelConfig


class TestProvider:
    """Real test provider implementation."""
    
    def __init__(self, model):
        """Initialize the test provider with the model instance."""
        self.model = model
        self.client = self  # Self-reference for compatibility
        self.messages = []
        self.responses = {}
        
    async def send_message(self, message: Message) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: The message to send
            
        Returns:
            A response string
        """
        self.messages.append(message)
        
        # Return a simple response based on the message content
        return f"Response to: {message.content[:30]}..."
    
    async def get_completion(self, prompt: str) -> str:
        """
        Get a completion for a prompt.
        
        Args:
            prompt: The prompt to complete
            
        Returns:
            A completion string
        """
        return f"Completion for: {prompt[:30]}..."
    
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
            return f"Response to: {last_message.content[:30]}..."
        return "No messages provided"


def create_provider(config: ModelConfig) -> TestProvider:
    """
    Create a test provider instance.
    
    Args:
        config: The model configuration
        
    Returns:
        A TestProvider instance
    """
    return TestProvider(config)
