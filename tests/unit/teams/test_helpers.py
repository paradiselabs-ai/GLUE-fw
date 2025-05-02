"""
Helper classes for team tests.
"""
import asyncio
from typing import Dict, List, Any, Optional

from glue.core.model import Model


class SimpleTool:
    """A simple tool implementation for unit tests."""
    
    def __init__(self, name: str = "test_tool", result: Any = None):
        self.name = name
        self.default_result = result or {"result": "success"}
        self.initialized = False
        self.cleaned_up = False
        self.tools = {}
        
    async def execute(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        # Return the default result or a custom result based on params
        if params and "custom_result" in params:
            return params["custom_result"]
        return self.default_result
        
    async def initialize(self) -> None:
        """Initialize the tool."""
        self.initialized = True
        
    async def cleanup(self) -> None:
        """Clean up tool resources."""
        self.cleaned_up = True


class SimpleModel(Model):
    """A simple model implementation for unit tests."""
    
    def __init__(self, name: str = "test_model"):
        # Initialize with basic config to match Model class requirements
        self.config = {"provider": "test", "model": "test"}
        # Ensure name is set as an attribute
        self.name = name
        super().__init__(config=self.config, name=name)
        self.tools = {}
        self.team = None
        self.generate_called = False
        self.generate_response = "Test response"
        self.adhesives = set()
        
    async def generate(self, prompt: str, tools: List[Dict[str, Any]] = None) -> str:
        """Generate a response from the model."""
        self.generate_called = True
        self.last_prompt = prompt
        return self.generate_response
        
    async def add_tool(self, tool_name: str, tool: Any) -> None:
        """Add a tool to the model."""
        self.tools[tool_name] = tool
        
    async def initialize(self) -> None:
        """Initialize the model."""
        self._initialized = True
        
    async def cleanup(self) -> None:
        """Clean up model resources."""
        self._initialized = False
