"""
Tests for persistence functionality in dynamic component creation.

This module contains tests for the persistence system that allows
dynamically created tools and MCP servers to be saved and loaded,
following the Test-Driven Development (TDD) approach.
"""
import os
import json
import tempfile
import uuid
from pathlib import Path
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock

from glue.core.persistence import (
    DynamicComponentStore,
    ComponentType,
    ComponentSpec
)
from glue.core.mcp_factory import DynamicMCPFactory
from glue.core.mcp import MCPServer
from glue.tools.tool_base import Tool

# ==================== Component Store Tests ====================
class TestDynamicComponentStore:
    """Tests for the DynamicComponentStore class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create a DynamicComponentStore instance"""
        return DynamicComponentStore(storage_dir=temp_dir)
    
    @pytest.fixture
    def mcp_factory(self):
        """Create a DynamicMCPFactory instance"""
        factory = DynamicMCPFactory()
        yield factory
    
    @pytest.mark.asyncio
    async def test_save_and_load_mcp_server(self, store, mcp_factory, temp_dir):
        """Test saving and loading an MCP server"""
        # Create a server
        handler_code = """
async def handle_request(request):
    return {"message": "Hello, world!"}
"""
        server = await mcp_factory.create_from_code(
            code=handler_code,
            name="test_server",
            handler_name="handle_request",
            action_name="test_action"
        )
        
        # Save the server
        component_id = await store.save_component(
            component_type=ComponentType.MCP_SERVER,
            name="test_server",
            code=handler_code,
            metadata={"version": "1.0"}
        )
        
        # Verify the file was created in the correct subdirectory
        component_path = Path(temp_dir) / ComponentType.MCP_SERVER.value / f"{component_id}.json"
        assert component_path.exists()
        
        # Load the component
        loaded_spec = await store.load_component(component_id)
        
        # Verify the loaded component
        assert loaded_spec.component_type == ComponentType.MCP_SERVER
        assert loaded_spec.name == "test_server"
        assert loaded_spec.code == handler_code
        assert loaded_spec.metadata["version"] == "1.0"
    
    @pytest.mark.asyncio
    async def test_save_and_load_tool(self, store, mcp_factory, temp_dir):
        """Test saving and loading a dynamic tool"""
        # Create a tool
        tool_code = """
def process_text(parameters):
    text = parameters.get("text", "")
    return {"processed": text.upper()}
"""
        tool = await mcp_factory.create_tool_from_code(
            code=tool_code,
            tool_name="text_processor",
            function_name="process_text",
            description="Process text to uppercase"
        )
        
        # Save the tool
        component_id = await store.save_component(
            component_type=ComponentType.TOOL,
            name="text_processor",
            code=tool_code,
            metadata={"version": "1.0"}
        )
        
        # Verify the file was created in the correct subdirectory
        component_path = Path(temp_dir) / ComponentType.TOOL.value / f"{component_id}.json"
        assert component_path.exists()
        
        # Load the component
        loaded_spec = await store.load_component(component_id)
        
        # Verify the loaded component
        assert loaded_spec.component_type == ComponentType.TOOL
        assert loaded_spec.name == "text_processor"
        assert loaded_spec.code == tool_code
        assert loaded_spec.metadata["version"] == "1.0"
    
    @pytest.mark.asyncio
    async def test_list_components(self, store, mcp_factory, temp_dir):
        """Test listing saved components"""
        # Save multiple components
        server1_id = await store.save_component(
            component_type=ComponentType.MCP_SERVER,
            name="server1",
            code="# Server 1 code",
            metadata={"version": "1.0"}
        )
        
        tool1_id = await store.save_component(
            component_type=ComponentType.TOOL,
            name="tool1",
            code="# Tool 1 code",
            metadata={"version": "1.0"}
        )
        
        server2_id = await store.save_component(
            component_type=ComponentType.MCP_SERVER,
            name="server2",
            code="# Server 2 code",
            metadata={"version": "1.0"}
        )
        
        # List all components
        all_components = await store.list_components()
        assert len(all_components) >= 3
        
        # List only MCP servers
        servers = await store.list_components(component_type=ComponentType.MCP_SERVER)
        assert len(servers) >= 2
        
        # List only tools
        tools = await store.list_components(component_type=ComponentType.TOOL)
        assert len(tools) >= 1
    
    @pytest.mark.asyncio
    async def test_delete_component(self, store, temp_dir):
        """Test deleting a saved component"""
        # Save a component
        component_id = await store.save_component(
            component_type=ComponentType.TOOL,
            name="temp_tool",
            code="# Temporary tool code",
            metadata={"version": "1.0"}
        )
        
        # Verify the file was created in the correct subdirectory
        component_path = Path(temp_dir) / ComponentType.TOOL.value / f"{component_id}.json"
        assert component_path.exists()
        
        # Delete the component
        await store.delete_component(component_id)
        
        # Verify the file was deleted
        assert not component_path.exists()
    
    @pytest.mark.asyncio
    async def test_update_component(self, store, temp_dir):
        """Test updating a saved component"""
        # Save a component
        component_id = await store.save_component(
            component_type=ComponentType.MCP_SERVER,
            name="update_server",
            code="# Original code",
            metadata={"version": "1.0"}
        )
        
        # Verify the file was created in the correct subdirectory
        component_path = Path(temp_dir) / ComponentType.MCP_SERVER.value / f"{component_id}.json"
        assert component_path.exists()
        
        # Load the component to verify initial state
        loaded_spec = await store.load_component(component_id)
        assert loaded_spec.code == "# Original code"
        
        # Update the component
        await store.update_component(
            component_id=component_id,
            code="# Updated code",
            metadata={"version": "1.1"}
        )
        
        # Load the updated component
        loaded_spec = await store.load_component(component_id)
        
        # Verify the updates
        assert loaded_spec.code == "# Updated code"
        assert loaded_spec.metadata["version"] == "1.1"
        assert loaded_spec.name == "update_server"  # Name should not change

# ==================== Factory Integration Tests ====================
class TestFactoryPersistenceIntegration:
    """Tests for integrating the persistence store with the factory"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest_asyncio.fixture
    async def mcp_factory(self, temp_dir):
        """Create a DynamicMCPFactory instance with persistence"""
        factory = DynamicMCPFactory()
        store = DynamicComponentStore(storage_dir=temp_dir)
        factory.component_store = store
        yield factory
        await factory.cleanup()
    
    @pytest.mark.asyncio
    async def test_factory_save_and_load_server(self, mcp_factory):
        """Test saving and loading an MCP server through the factory"""
        # Create a server
        handler_code = """
async def handle_query(request):
    query = request.get("parameters", {}).get("query", "")
    return {
        "result": f"Processed: {query}",
        "length": len(query)
    }
"""
        server = await mcp_factory.create_from_code(
            code=handler_code,
            name="query_server",
            handler_name="handle_query",
            action_name="query"
        )
        
        # Save the server
        await mcp_factory.component_store.save_component(
            component_type=ComponentType.MCP_SERVER,
            name="query_server",
            code=handler_code,
            metadata={"version": "1.0"}
        )
        
        # Verify the server was created and saved
        assert server.name == "query_server"
        
        # List saved servers
        servers = await mcp_factory.component_store.list_components(
            component_type=ComponentType.MCP_SERVER
        )
        assert any(s.name == "query_server" for s in servers)
    
    @pytest.mark.asyncio
    async def test_factory_save_and_load_tool(self, mcp_factory):
        """Test saving and loading a tool through the factory"""
        # Create a tool
        tool_code = """
def calculate(parameters):
    a = parameters.get("a", 0)
    b = parameters.get("b", 0)
    operation = parameters.get("operation", "add")

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else "Error: Division by zero"
    else:
        result = "Error: Unknown operation"

    return {"result": result}
"""
        tool = await mcp_factory.create_tool_from_code(
            code=tool_code,
            tool_name="calculator",
            function_name="calculate",
            description="Perform basic calculations"
        )
        
        # Save the tool
        await mcp_factory.component_store.save_component(
            component_type=ComponentType.TOOL,
            name="calculator",
            code=tool_code,
            metadata={"version": "1.0"}
        )
        
        # Verify the tool was created and saved
        assert tool.name == "calculator"
        
        # List saved tools
        tools = await mcp_factory.component_store.list_components(
            component_type=ComponentType.TOOL
        )
        assert any(t.name == "calculator" for t in tools)

# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
