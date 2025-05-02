"""
Tests for dynamic tool creation in the GLUE framework.

This module contains tests for the DynamicToolFactory class, which enables
runtime creation of tools, following the Test-Driven Development (TDD) approach.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Import the tool factory and related classes
from glue.tools.tool_factory import DynamicToolFactory, ToolSpec
from glue.tools.tool_base import Tool, ToolConfig, ToolPermission
from glue.core.types import AdhesiveType

# ==================== DynamicToolFactory Tests ====================
class TestDynamicToolFactory:
    """Tests for the DynamicToolFactory class"""
    
    @pytest.fixture
    def factory(self):
        """Create a DynamicToolFactory instance"""
        return DynamicToolFactory()
    
    @pytest.mark.asyncio
    async def test_create_tool_from_spec(self, factory):
        """Test creating a tool from a specification"""
        # Define a simple function for the tool
        def test_function(input_data):
            return {"result": f"Processed: {input_data}"}
        
        # Create a tool specification
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            function=test_function,
            permissions={ToolPermission.READ},
            adhesives={AdhesiveType.TAPE}
        )
        
        # Create the tool
        tool = await factory.create_tool(spec)
        
        # Verify the tool was created correctly
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        
        # Test the tool execution
        result = await tool.execute("test_input")
        assert result == {"result": "Processed: test_input"}
    
    @pytest.mark.asyncio
    async def test_create_tool_from_code(self, factory):
        """Test creating a tool from Python code string"""
        # Define a code string
        code = """
def code_tool(input_data):
    \"\"\"Tool created from code\"\"\"
    return {"status": "success", "input": input_data}
"""
        
        # Create the tool
        tool = await factory.create_from_code(code)
        
        # Verify the tool was created correctly
        assert tool is not None
        assert tool.name == "code_tool"
        assert "Tool created from code" in tool.description
        
        # Test the tool execution
        result = await tool.execute("code_input")
        assert result["status"] == "success"
        assert result["input"] == "code_input"
    
    @pytest.mark.asyncio
    async def test_create_async_tool(self, factory):
        """Test creating an async tool"""
        # Define an async function
        async def async_function(input_data):
            await asyncio.sleep(0.1)  # Simulate async operation
            return {"async_result": input_data}
        
        # Create a tool specification
        spec = ToolSpec(
            name="async_tool",
            description="An async test tool",
            function=async_function
        )
        
        # Create the tool
        tool = await factory.create_tool(spec)
        
        # Test the tool execution
        result = await tool.execute("async_input")
        assert result["async_result"] == "async_input"
    
    @pytest.mark.asyncio
    async def test_tool_with_permissions(self, factory):
        """Test creating a tool with specific permissions"""
        # Create a tool with network and execute permissions
        spec = ToolSpec(
            name="permission_tool",
            description="A tool with permissions",
            function=lambda x: {"result": x},
            permissions={ToolPermission.NETWORK, ToolPermission.EXECUTE}
        )
        
        # Create the tool
        tool = await factory.create_tool(spec)
        
        # Verify permissions
        assert ToolPermission.NETWORK in tool.config.required_permissions
        assert ToolPermission.EXECUTE in tool.config.required_permissions
        assert ToolPermission.WRITE not in tool.config.required_permissions
    
    @pytest.mark.asyncio
    async def test_natural_language_tool_creation(self, factory):
        """Test creating a tool from natural language"""
        # Test with a simple request
        tool = await factory.parse_natural_request(
            "Create a tool that searches for information online", 
            "research_team"
        )
        
        # Verify the tool was created
        assert tool is not None
        assert "search" in tool.name.lower()
        assert ToolPermission.NETWORK in tool.config.required_permissions
        
        # Test with a write operation
        tool = await factory.parse_natural_request(
            "Create a tool to save data to a file", 
            "data_team"
        )
        
        assert tool is not None
        assert "save" in tool.name.lower() or "data" in tool.name.lower()
        assert ToolPermission.WRITE in tool.config.required_permissions
    
    @pytest.mark.asyncio
    async def test_duplicate_tool_handling(self, factory):
        """Test handling of duplicate tool creation"""
        # Create a tool
        spec = ToolSpec(
            name="duplicate_tool",
            description="Original tool",
            function=lambda x: {"original": True}
        )
        
        original_tool = await factory.create_tool(spec)
        
        # Try to create a tool with the same name
        duplicate_spec = ToolSpec(
            name="duplicate_tool",
            description="Duplicate tool",
            function=lambda x: {"duplicate": True}
        )
        
        duplicate_tool = await factory.create_tool(duplicate_spec)
        
        # Should return the original tool
        assert duplicate_tool is original_tool
        assert duplicate_tool.description == "Original tool"
        
        # Test execution to verify it's the original
        result = await duplicate_tool.execute("test")
        assert "original" in result
    
    @pytest.mark.asyncio
    async def test_tool_cleanup(self, factory):
        """Test cleaning up created tools"""
        # Create a tool with a cleanup method
        cleanup_mock = AsyncMock()
        
        class CleanupTool(Tool):
            async def cleanup(self):
                await cleanup_mock()
                
            async def _execute(self, input_data):
                return {"cleaned": True}
        
        # Register and create the tool
        DynamicToolFactory.register_tool_class(CleanupTool)
        
        spec = ToolSpec(
            name="cleanup_tool",
            description="A tool with cleanup",
            function=lambda x: {"cleaned": True}
        )
        
        await factory.create_tool(spec)
        
        # Clean up all tools
        await factory.cleanup()
        
        # Verify cleanup was called
        cleanup_mock.assert_called_once()
        
        # Verify tools were cleared
        assert len(factory.created_tools) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, factory):
        """Test error handling in dynamic tools"""
        # Define a function that raises an error
        def error_function(input_data):
            raise ValueError("Test error")
        
        # Create a tool spec with the error function
        spec = ToolSpec(
            name="error_tool",
            description="A tool that raises an error",
            function=error_function
        )
        
        # Create the tool
        tool = await factory.create_tool(spec)
        
        # Test that the error is propagated correctly
        with pytest.raises(ValueError) as excinfo:
            await tool.execute("test")
        
        # Verify the error message
        assert str(excinfo.value) == "Test error"
    
    @pytest.mark.asyncio
    async def test_error_handling_isolated(self):
        """Test error handling in dynamic tools with an isolated factory"""
        # Create a fresh factory instance specifically for this test
        isolated_factory = DynamicToolFactory()
        
        # Create a tool that raises an exception
        def error_function(input_data):
            raise ValueError("Test error")
        
        spec = ToolSpec(
            name="error_tool",
            description="A tool that raises an error",
            function=error_function
        )
        
        tool = await isolated_factory.create_tool(spec)
        
        # Test that the error is propagated correctly
        with pytest.raises(ValueError) as excinfo:
            await tool.execute("test")
        
        # Verify the error message
        assert str(excinfo.value) == "Test error"
        
        # Clean up
        await isolated_factory.cleanup()
    
    @pytest.mark.asyncio
    async def test_direct_error_handling(self):
        """Test error handling directly without using the factory"""
        # Create a simple tool class that raises an error
        class ErrorTool(Tool):
            async def _execute(self, input_data):
                raise ValueError("Direct test error")
        
        # Create an instance
        tool = ErrorTool(
            name="direct_error_tool",
            description="A tool that directly raises an error"
        )
        
        # Initialize the tool
        await tool.initialize()
        
        # Verify the error is raised correctly
        try:
            await tool._execute("test")
            pytest.fail("Expected ValueError but no exception was raised")
        except ValueError as e:
            assert str(e) == "Direct test error"
            
        # Clean up
        await tool.cleanup()
    
# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
