"""
Unit tests for the tool registry in the GLUE framework.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from typing import Any, Dict, Optional, List

from glue.tools.tool_base import Tool, ToolConfig, ToolPermission
from glue.tools.tool_registry import ToolRegistry


# Create a minimal concrete implementation for testing
class MockTool(Tool):
    async def _execute(self, input_data) -> Any:
        return {"result": "test implementation"}


class AnotherMockTool(Tool):
    async def _execute(self, input_data) -> Any:
        return {"result": "another test implementation"}


class TestToolRegistry:
    """Test suite for the ToolRegistry class."""

    def test_registry_initialization(self):
        """Test that a tool registry can be initialized."""
        registry = ToolRegistry()
        assert registry is not None
        assert len(registry.get_all_tools()) == 0

    def test_tool_registration(self):
        """Test that tools can be registered."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="test_tool",
            description="A test tool"
        )
        
        tool2 = AnotherMockTool(
            name="another_tool",
            description="Another test tool"
        )
        
        registry.register(tool1)
        registry.register(tool2)
        
        assert len(registry.get_all_tools()) == 2
        assert registry.get_tool("test_tool") == tool1
        assert registry.get_tool("another_tool") == tool2

    def test_tool_registration_duplicate(self):
        """Test that registering a tool with a duplicate name raises an error."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="test_tool",
            description="A test tool"
        )
        
        tool2 = AnotherMockTool(
            name="test_tool",  # Same name as tool1
            description="Another test tool"
        )
        
        registry.register(tool1)
        
        with pytest.raises(ValueError):
            registry.register(tool2)

    def test_tool_unregistration(self):
        """Test that tools can be unregistered."""
        registry = ToolRegistry()
        
        tool = MockTool(
            name="test_tool",
            description="A test tool"
        )
        
        registry.register(tool)
        assert len(registry.get_all_tools()) == 1
        
        registry.unregister("test_tool")
        assert len(registry.get_all_tools()) == 0
        
        # Unregistering a non-existent tool should not raise an error
        registry.unregister("non_existent_tool")

    def test_get_tool_by_name(self):
        """Test that tools can be retrieved by name."""
        registry = ToolRegistry()
        
        tool = MockTool(
            name="test_tool",
            description="A test tool"
        )
        
        registry.register(tool)
        
        retrieved_tool = registry.get_tool("test_tool")
        assert retrieved_tool == tool
        
        # Getting a non-existent tool should return None
        assert registry.get_tool("non_existent_tool") is None

    def test_get_tools_by_permission(self):
        """Test that tools can be filtered by required permissions."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="read_tool",
            description="A tool requiring read permission",
            config=ToolConfig(
                required_permissions=[ToolPermission.READ]
            )
        )
        
        tool2 = AnotherMockTool(
            name="write_tool",
            description="A tool requiring write permission",
            config=ToolConfig(
                required_permissions=[ToolPermission.WRITE]
            )
        )
        
        tool3 = MockTool(
            name="network_tool",
            description="A tool requiring network permission",
            config=ToolConfig(
                required_permissions=[ToolPermission.NETWORK]
            )
        )
        
        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)
        
        read_tools = registry.get_tools_by_permission(ToolPermission.READ)
        assert len(read_tools) == 1
        assert read_tools[0].name == "read_tool"
        
        write_tools = registry.get_tools_by_permission(ToolPermission.WRITE)
        assert len(write_tools) == 1
        assert write_tools[0].name == "write_tool"
        
        # Test getting tools with multiple permissions
        tool4 = AnotherMockTool(
            name="multi_tool",
            description="A tool requiring multiple permissions",
            config=ToolConfig(
                required_permissions=[ToolPermission.READ, ToolPermission.WRITE]
            )
        )
        
        registry.register(tool4)
        
        read_tools = registry.get_tools_by_permission(ToolPermission.READ)
        assert len(read_tools) == 2
        assert any(tool.name == "read_tool" for tool in read_tools)
        assert any(tool.name == "multi_tool" for tool in read_tools)

    @pytest.mark.asyncio
    async def test_initialize_all_tools(self):
        """Test that all tools can be initialized at once."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="tool1",
            description="Tool 1"
        )
        
        tool2 = MockTool(
            name="tool2",
            description="Tool 2"
        )
        
        registry.register(tool1)
        registry.register(tool2)
        
        await registry.initialize_all()
        
        assert tool1._initialized
        assert tool2._initialized
        
        await registry.cleanup_all()
        
        assert not tool1._initialized
        assert not tool2._initialized

    @pytest.mark.asyncio
    async def test_initialize_specific_tools(self):
        """Test that specific tools can be initialized by name."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="tool1",
            description="Tool 1"
        )
        
        tool2 = MockTool(
            name="tool2",
            description="Tool 2"
        )
        
        registry.register(tool1)
        registry.register(tool2)
        
        await registry.initialize_tools(["tool1"])
        
        assert tool1._initialized
        assert not tool2._initialized
        
        await registry.cleanup_tools(["tool1"])
        
        assert not tool1._initialized
        assert not tool2._initialized

    def test_registry_to_dict(self):
        """Test that the registry can be converted to a dictionary."""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="tool1",
            description="Tool 1"
        )
        
        tool2 = MockTool(
            name="tool2",
            description="Tool 2"
        )
        
        registry.register(tool1)
        registry.register(tool2)
        
        registry_dict = registry.to_dict()
        
        assert len(registry_dict["tools"]) == 2
        assert any(tool["name"] == "tool1" for tool in registry_dict["tools"])
        assert any(tool["name"] == "tool2" for tool in registry_dict["tools"])
