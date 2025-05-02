"""
Tests for sandboxing functionality in dynamic component creation.

This module contains tests for the sandboxing system that provides security
for dynamically created tools and MCP servers, following the Test-Driven Development (TDD) approach.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import importlib

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Import necessary modules
from glue.core.mcp import MCPServer
from glue.tools.tool_base import Tool

# This import will fail until we implement the sandbox - that's expected in TDD
try:
    from glue.core.sandbox import CodeSandbox, SandboxConfig, SandboxViolation
except ImportError:
    pass  # We'll implement this later

# ==================== CodeSandbox Tests ====================
class TestCodeSandbox:
    """Tests for the CodeSandbox class"""
    
    @pytest.fixture
    def sandbox(self):
        """Create a CodeSandbox instance"""
        # This will fail until we implement the sandbox
        config = SandboxConfig(
            allowed_modules=["math", "json", "re"],
            forbidden_modules=["os", "sys", "subprocess"],
            memory_limit_mb=100,
            execution_timeout_seconds=5,
            allow_network=False
        )
        return CodeSandbox(config)
    
    @pytest.mark.asyncio
    async def test_safe_code_execution(self, sandbox):
        """Test executing safe code in the sandbox"""
        # Define safe code that uses allowed modules
        safe_code = """
import math
import json

def calculate_area(radius):
    \"\"\"Calculate the area of a circle\"\"\"
    return math.pi * radius ** 2

def process_data(data_str):
    \"\"\"Process JSON data\"\"\"
    data = json.loads(data_str)
    return {
        "processed": True,
        "count": len(data),
        "keys": list(data.keys())
    }
"""
        
        # Execute the code in the sandbox
        module = await sandbox.execute_code(safe_code, "safe_module")
        
        # Verify the functions work as expected
        assert hasattr(module, "calculate_area")
        assert hasattr(module, "process_data")
        
        # Test the functions
        area = module.calculate_area(5)
        assert round(area, 2) == round(78.54, 2)
        
        result = module.process_data('{"a": 1, "b": 2}')
        assert result["processed"] is True
        assert result["count"] == 2
        assert set(result["keys"]) == {"a", "b"}
    
    @pytest.mark.asyncio
    async def test_unsafe_code_detection(self, sandbox):
        """Test detecting unsafe code that tries to import forbidden modules"""
        # Define unsafe code that tries to import os
        unsafe_code = """
import os
import math

def get_files():
    \"\"\"Get files in current directory\"\"\"
    return os.listdir('.')
"""
        
        # Attempt to execute the code in the sandbox
        with pytest.raises(SandboxViolation) as excinfo:
            await sandbox.execute_code(unsafe_code, "unsafe_module")
        
        # Verify the error message
        assert "Forbidden module" in str(excinfo.value)
        assert "os" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_code_with_system_calls(self, sandbox):
        """Test detecting code that tries to make system calls"""
        # Define code that tries to execute system commands
        system_call_code = """
def execute_command(cmd):
    \"\"\"Execute a system command\"\"\"
    import subprocess
    return subprocess.check_output(cmd, shell=True)
"""
        
        # Attempt to execute the code in the sandbox
        with pytest.raises(SandboxViolation) as excinfo:
            await sandbox.execute_code(system_call_code, "system_call_module")
        
        # Verify the error message
        assert "Forbidden module" in str(excinfo.value)
        assert "subprocess" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_code_with_file_operations(self, sandbox):
        """Test detecting code that tries to perform file operations"""
        # Define code that tries to read/write files
        file_op_code = """
def write_to_file(content):
    \"\"\"Write content to a file\"\"\"
    with open('test.txt', 'w') as f:
        f.write(content)
    return True
"""
        
        # Attempt to execute the code in the sandbox
        with pytest.raises(SandboxViolation) as excinfo:
            await sandbox.execute_code(file_op_code, "file_op_module")
        
        # Verify the error message
        assert "File operation" in str(excinfo.value) or "Forbidden operation" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, sandbox):
        """Test enforcing memory limits in the sandbox"""
        # Define code that tries to allocate a large amount of memory
        memory_hog_code = """
def allocate_large_memory():
    \"\"\"Allocate a large amount of memory\"\"\"
    # Try to allocate 200MB (exceeding the 100MB limit)
    large_list = [0] * (200 * 1024 * 1024 // 8)  # 200MB worth of integers
    return len(large_list)
"""
        
        # Attempt to execute the code in the sandbox
        with pytest.raises(SandboxViolation) as excinfo:
            module = await sandbox.execute_code(memory_hog_code, "memory_hog_module")
            # This should fail when the function is called
            module.allocate_large_memory()
        
        # Verify the error message
        assert "Memory limit" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self, sandbox):
        """Test enforcing execution timeouts in the sandbox"""
        # Define code with an infinite loop
        infinite_loop_code = """
import time

def run_forever():
    \"\"\"Function that runs for a long time\"\"\"
    while True:
        time.sleep(0.1)
"""
        
        # Attempt to execute the code in the sandbox
        module = await sandbox.execute_code(infinite_loop_code, "timeout_module")
        
        # Call the function that should timeout
        with pytest.raises(SandboxViolation) as excinfo:
            await sandbox.execute_function(module.run_forever)
        
        # Verify the error message
        assert "Execution timeout" in str(excinfo.value)

# ==================== Sandbox Integration Tests ====================
class TestSandboxIntegration:
    """Tests for integrating the sandbox with dynamic components"""
    
    @pytest.fixture
    def mcp_factory(self):
        """Create a factory with sandbox integration"""
        from glue.core.mcp_factory import DynamicMCPFactory
        factory = DynamicMCPFactory()
        return factory
    
    @pytest.mark.asyncio
    async def test_sandbox_with_mcp_creation(self, mcp_factory):
        """Test creating an MCP server with sandboxed handlers"""
        # Define a handler with safe code
        handler_code = """
import json

async def process_data(request):
    \"\"\"Process data in a sandboxed environment\"\"\"
    data = request.get("parameters", {})
    result = {
        "processed": True,
        "item_count": len(data),
        "has_id": "id" in data
    }
    return result
"""
        
        # Create the MCP server with sandboxing enabled
        server = await mcp_factory.create_from_code(
            code=handler_code,
            name="sandboxed_mcp",
            handler_name="process_data",
            action_name="process",
            sandbox_enabled=True
        )
        
        # Test the server with a request
        request = {"action": "process", "parameters": {"id": 123, "name": "test"}}
        response = await server.handle_request(request)
        
        # Verify the response
        assert response["processed"] is True
        assert response["item_count"] == 2
        assert response["has_id"] is True
    
    @pytest.mark.asyncio
    async def test_sandbox_with_unsafe_mcp_creation(self, mcp_factory):
        """Test creating an MCP server with unsafe handlers"""
        # Define a handler with unsafe code
        unsafe_handler_code = """
import os

async def get_files(request):
    \"\"\"Try to access the file system\"\"\"
    directory = request.get("parameters", {}).get("dir", ".")
    files = os.listdir(directory)
    return {"files": files}
"""
        
        # Attempt to create the MCP server with sandboxing enabled
        with pytest.raises(SandboxViolation) as excinfo:
            await mcp_factory.create_from_code(
                code=unsafe_handler_code,
                name="unsafe_mcp",
                handler_name="get_files",
                action_name="list_files",
                sandbox_enabled=True
            )
        
        # Verify the error message
        assert "Forbidden module" in str(excinfo.value)
        assert "os" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_sandbox_with_tool_creation(self, mcp_factory):
        """Test creating a tool with sandboxed execution"""
        # Define tool execution code
        tool_code = """
async def execute_tool(parameters):
    \"\"\"Execute a tool in a sandboxed environment\"\"\"
    text = parameters.get("text", "")
    return {
        "result": text.upper(),
        "length": len(text),
        "words": len(text.split())
    }
"""
        
        # Create the tool with sandboxing enabled
        tool = await mcp_factory.create_tool_from_code(
            code=tool_code,
            tool_name="text_processor",
            function_name="execute_tool",
            description="Process text in various ways",
            sandbox_enabled=True
        )
        
        # Execute the tool
        result = await tool.execute({"text": "hello world"})
        
        # Verify the result
        assert result["result"] == "HELLO WORLD"
        assert result["length"] == 11
        assert result["words"] == 2

# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
