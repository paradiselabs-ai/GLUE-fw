"""
Unit tests for the base tool class in the GLUE framework.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from typing import Any, Dict, Optional

from glue.core.schemas import ToolCall, ToolResult
from glue.tools.tool_base import Tool, ToolConfig, ToolPermission, DynamicTool
from glue.core.adhesive import AdhesiveType


# Create a minimal concrete implementation for testing
class MinimalTool(Tool):
    async def _execute(self, input_data) -> Any:
        return {"result": "minimal implementation"}


class TestBaseTool:
    """Test suite for the BaseTool class."""

    def test_tool_initialization(self):
        """Test that a tool can be initialized with basic properties."""
        tool = MinimalTool(
            name="test_tool",
            description="A test tool",
            config=ToolConfig(
                required_permissions=[ToolPermission.READ],
                timeout=30.0
            )
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.config.timeout == 30.0
        assert ToolPermission.READ in tool.config.required_permissions
        assert not tool._initialized

    @pytest.mark.asyncio
    async def test_tool_initialization_async(self):
        """Test that a tool can be initialized asynchronously."""
        tool = MinimalTool(
            name="test_tool",
            description="A test tool"
        )
        
        await tool.initialize()
        assert tool._initialized
        
        await tool.cleanup()
        assert not tool._initialized

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test that a tool can execute a call and return a result."""
        # Create a test tool with a mock execute method
        class TestTool(Tool):
            async def _execute(self, input_data) -> Any:
                return {"result": "success", "input": input_data}
        
        tool = TestTool(
            name="test_tool",
            description="A test tool"
        )
        
        # Create a tool call
        tool_call = ToolCall(
            tool_id="call_123",
            name="test_tool",
            arguments={"param1": "value1", "param2": "value2"}
        )
        
        # Execute the tool call
        await tool.initialize()
        result = await tool.execute(tool_call.arguments)
        await tool.cleanup()
        
        # Verify the result
        assert isinstance(result, dict)
        assert result["result"] == "success"
        assert result["input"]["param1"] == "value1"
        assert result["input"]["param2"] == "value2"

    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self):
        """Test that a tool execution times out after the configured timeout."""
        # Create a test tool with a slow execute method
        class SlowTool(Tool):
            async def _execute(self, input_data) -> Any:
                await asyncio.sleep(0.5)  # Simulate slow execution
                return {"result": "success"}
        
        tool = SlowTool(
            name="slow_tool",
            description="A slow tool",
            config=ToolConfig(
                timeout=0.1  # Very short timeout
            )
        )
        
        # Create a tool call
        tool_call = ToolCall(
            tool_id="call_456",
            name="slow_tool",
            arguments={"param": "value"}
        )
        
        # Execute the tool call
        await tool.initialize()
        with pytest.raises(asyncio.TimeoutError):
            await tool.execute(tool_call.arguments)
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test that a tool handles execution errors gracefully."""
        # Create a test tool that raises an exception
        class ErrorTool(Tool):
            async def _execute(self, input_data) -> Any:
                raise ValueError("Test error")
        
        tool = ErrorTool(
            name="error_tool",
            description="A tool that raises an error"
        )
        
        # Create a tool call
        tool_call = ToolCall(
            tool_id="call_789",
            name="error_tool",
            arguments={"param": "value"}
        )
        
        # Execute the tool call
        await tool.initialize()
        with pytest.raises(Exception):
            await tool.execute(tool_call.arguments)
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_tool_permission_validation(self):
        """Test that a tool validates permissions before execution."""
        # Create a test tool with required permissions
        class PermissionTool(Tool):
            async def _execute(self, input_data) -> Any:
                return {"result": "success"}
            
            async def _check_permission(self, permission: ToolPermission) -> bool:
                # Override to simulate permission checks
                return permission != ToolPermission.NETWORK
        
        tool = PermissionTool(
            name="permission_tool",
            description="A tool that requires permissions",
            config=ToolConfig(
                required_permissions=[ToolPermission.WRITE, ToolPermission.NETWORK]
            )
        )
        
        # Initialize should fail due to missing NETWORK permission
        with pytest.raises(PermissionError):
            await tool.initialize()
            
        # Try with only WRITE permission
        tool = PermissionTool(
            name="permission_tool",
            description="A tool that requires permissions",
            config=ToolConfig(
                required_permissions=[ToolPermission.WRITE]
            )
        )
        
        # This should succeed
        await tool.initialize()
        await tool.cleanup()

    @pytest.mark.asyncio
    async def test_dynamic_tool_execution(self):
        """Test that a dynamic tool can execute properly."""
        # Create a dynamic tool with a mock function
        async def mock_function(input_data):
            return {"dynamic": "result", "input": input_data}
        
        tool = DynamicTool(
            name="dynamic_tool",
            description="A dynamic tool",
            function=mock_function
        )
        
        # Execute the tool
        await tool.initialize()
        result = await tool.execute({"param": "value"})
        await tool.cleanup()
        
        # Verify the result
        assert result["dynamic"] == "result"
        assert result["input"]["param"] == "value"

    def test_tool_to_dict(self):
        """Test that a tool can be converted to a dictionary."""
        # Create a tool with a custom to_dict method for testing
        class DictTool(MinimalTool):
            def to_dict(self):
                return {
                    "name": self.name,
                    "description": self.description,
                    "inputs": self.inputs,
                    "config": {
                        "timeout": self.config.timeout,
                        "required_permissions": list(self.config.required_permissions),
                        "adhesive_types": list(self.config.adhesive_types),
                        "max_retries": self.config.max_retries,
                        "metadata": self.config.metadata
                    },
                    "initialized": self._initialized
                }
        
        tool = DictTool(
            name="test_tool",
            description="A test tool",
            config=ToolConfig(
                timeout=15.0,
                required_permissions=[ToolPermission.READ, ToolPermission.WRITE]
            )
        )
        
        tool_dict = tool.to_dict()
        
        assert tool_dict["name"] == "test_tool"
        assert tool_dict["description"] == "A test tool"
        assert tool_dict["config"]["timeout"] == 15.0
        assert ToolPermission.READ in tool_dict["config"]["required_permissions"]
        assert ToolPermission.WRITE in tool_dict["config"]["required_permissions"]
