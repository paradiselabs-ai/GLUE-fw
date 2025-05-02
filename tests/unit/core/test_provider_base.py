"""
Tests for the provider base class in the GLUE framework.

This module contains tests for the provider base class that provides
a common interface for all model providers.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from glue.core.schemas import Message, ToolCall, ToolResult, ModelConfig
from glue.core.model import BaseModel, ModelProvider, ProviderBase
from glue.core.types import AdhesiveType


class MockProvider(ProviderBase):
    """Mock provider implementation for testing"""
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Mock implementation of generate_response"""
        return "Mock response"
    
    async def process_tool_calls(
        self, 
        tool_calls: List[ToolCall], 
        tool_executor: Any
    ) -> List[ToolResult]:
        """Mock implementation of process_tool_calls"""
        results = []
        for tool_call in tool_calls:
            result = await tool_executor(tool_call)
            results.append(result)
        return results


class TestProviderBase:
    """Tests for the ProviderBase class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4"
        )
        
        self.model = BaseModel(self.config)
        # Mock the _initialize_client method to prevent actual API client creation
        self.model._initialize_client = MagicMock()
        
        self.provider = MockProvider(self.model)
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test that generate_response returns the expected result"""
        messages = [
            Message(role="user", content="Hello, world!")
        ]
        
        response = await self.provider.generate_response(messages)
        
        assert response == "Mock response"
    
    @pytest.mark.asyncio
    async def test_process_tool_calls(self):
        """Test that process_tool_calls calls the tool executor for each tool call"""
        tool_calls = [
            ToolCall(
                tool_id="call_1",
                name="search",
                arguments={"query": "test query"}
            ),
            ToolCall(
                tool_id="call_2",
                name="calculator",
                arguments={"expression": "1 + 1"}
            )
        ]
        
        # Mock the tool executor
        tool_executor = AsyncMock()
        tool_executor.side_effect = [
            ToolResult(
                tool_name="search",
                result="Search results for test query",
                adhesive=AdhesiveType.GLUE,
                metadata={}
            ),
            ToolResult(
                tool_name="calculator",
                result="2",
                adhesive=AdhesiveType.GLUE,
                metadata={}
            )
        ]
        
        results = await self.provider.process_tool_calls(tool_calls, tool_executor)
        
        # Verify the tool executor was called for each tool call
        assert tool_executor.call_count == 2
        tool_executor.assert_any_call(tool_calls[0])
        tool_executor.assert_any_call(tool_calls[1])
        
        # Verify the results
        assert len(results) == 2
        assert results[0].tool_name == "search"
        assert results[0].result == "Search results for test query"
        assert results[1].tool_name == "calculator"
        assert results[1].result == "2"
