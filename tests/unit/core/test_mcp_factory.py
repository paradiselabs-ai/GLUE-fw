"""
Tests for dynamic MCP creation in the GLUE framework.

This module contains tests for the MCPFactory class, which enables
runtime creation of MCP servers and clients, following the Test-Driven Development (TDD) approach.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Import the MCP classes
from glue.core.mcp import MCPConfig, MCPClient, MCPTool, MCPServer
from glue.tools.tool_base import Tool, ToolConfig, ToolPermission
from glue.core.types import AdhesiveType

# This import will fail until we implement the factory - that's expected in TDD
try:
    from glue.core.mcp_factory import MCPSpec, DynamicMCPFactory, create_dynamic_mcp_server
except ImportError:
    pass  # We'll implement this later

# ==================== MCPFactory Tests ====================
class TestDynamicMCPFactory:
    """Tests for the DynamicMCPFactory class"""
    
    @pytest.fixture
    def factory(self):
        """Create a DynamicMCPFactory instance"""
        # This will fail until we implement the factory
        return DynamicMCPFactory()
    
    @pytest.mark.asyncio
    async def test_create_mcp_server_from_spec(self, factory):
        """Test creating an MCP server from a specification"""
        # Define a simple handler function
        async def test_handler(request):
            return {"status": "success", "message": "Handled by dynamic MCP"}
        
        # Create an MCP specification
        spec = MCPSpec(
            name="test_mcp",
            version="1.0.0",
            host="localhost",
            port=8080,
            handlers={"test_action": test_handler}
        )
        
        # Create the MCP server
        server = await factory.create_server(spec)
        
        # Verify the server was created correctly
        assert server is not None
        assert server.name == "test_mcp"
        assert server.version == "1.0.0"
        assert server.host == "localhost"
        assert server.port == 8080
        
        # Test the server's request handling
        request = {"action": "test_action", "parameters": {}}
        response = await server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["message"] == "Handled by dynamic MCP"
    
    @pytest.mark.asyncio
    async def test_create_mcp_from_code(self, factory):
        """Test creating an MCP server from Python code string"""
        # Define a code string
        code = """
async def handle_custom_action(request):
    \"\"\"Custom MCP action handler\"\"\"
    return {
        "status": "success", 
        "custom": True,
        "request": request
    }
"""
        
        # Create the MCP server
        server = await factory.create_from_code(
            code=code,
            name="code_mcp",
            handler_name="handle_custom_action",
            action_name="custom_action"
        )
        
        # Verify the server was created correctly
        assert server is not None
        assert server.name == "code_mcp"
        
        # Test the server's request handling
        request = {"action": "custom_action", "parameters": {"test": 123}}
        response = await server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["custom"] is True
        assert response["request"]["parameters"]["test"] == 123
    
    @pytest.mark.asyncio
    async def test_register_mcp_with_team(self, factory):
        """Test registering an MCP server with a team"""
        # Create a mock team
        mock_team = MagicMock()
        mock_team.name = "test_team"
        mock_team.register_mcp = MagicMock()
        
        # Create a simple MCP spec
        spec = MCPSpec(
            name="team_mcp",
            version="1.0.0",
            host="localhost",
            port=8081
        )
        
        # Create and register the MCP server
        server = await factory.create_server(spec)
        factory.register_with_team(server, mock_team)
        
        # Verify the server was registered with the team
        mock_team.register_mcp.assert_called_once_with(server)
    
    @pytest.mark.asyncio
    async def test_create_mcp_tool_for_server(self, factory):
        """Test creating an MCP tool that connects to a server"""
        # Create a mock server
        mock_server = MagicMock()
        mock_server.name = "tool_server"
        mock_server.host = "localhost"
        mock_server.port = 8082
        
        # Create a tool for the server
        tool = await factory.create_tool_for_server(
            server=mock_server,
            tool_name="server_tool",
            description="Tool for server"
        )
        
        # Verify the tool was created correctly
        assert tool is not None
        assert tool.name == "server_tool"
        assert tool.description == "Tool for server"
        assert "localhost:8082" in tool.mcp_client.config.endpoint
    
    @pytest.mark.asyncio
    async def test_natural_language_mcp_creation(self, factory):
        """Test creating an MCP server from natural language"""
        # Test with a simple request
        server = await factory.parse_natural_request(
            "Create an MCP server that processes data requests", 
            "data_team"
        )
        
        # Verify the server was created
        assert server is not None
        assert "data" in server.name.lower()
        
        # Test the server's capabilities
        request = {"action": "process_data", "parameters": {"data": "test"}}
        response = await server.handle_request(request)
        
        assert response["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_mcp_server_cleanup(self, factory):
        """Test cleaning up created MCP servers"""
        # Create a server with a cleanup method
        cleanup_mock = AsyncMock()
        
        class CleanupServer(MCPServer):
            async def cleanup(self):
                await cleanup_mock()
        
        # Create a mock server instance
        mock_server = CleanupServer("cleanup_server")
        factory.created_servers = {"cleanup_server": mock_server}
        
        # Clean up all servers
        await factory.cleanup()
        
        # Verify cleanup was called
        cleanup_mock.assert_called_once()
        
        # Verify servers were cleared
        assert len(factory.created_servers) == 0

# ==================== Factory Function Tests ====================
class TestMCPFactoryFunctions:
    """Tests for MCP factory functions"""
    
    @pytest.mark.asyncio
    async def test_create_dynamic_mcp_server(self):
        """Test creating an MCP server with the factory function"""
        # Define a handler function
        async def test_handler(request):
            return {"dynamic": True}
        
        # Create a server with the factory function
        server = await create_dynamic_mcp_server(
            name="factory_server",
            version="1.0.0",
            handlers={"test_action": test_handler}
        )
        
        # Verify the server was created correctly
        assert server is not None
        assert server.name == "factory_server"
        assert server.version == "1.0.0"
        
        # Test the server's request handling
        request = {"action": "test_action", "parameters": {}}
        response = await server.handle_request(request)
        
        assert response["dynamic"] is True

# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
