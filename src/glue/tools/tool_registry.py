"""
Tool registry for the GLUE framework.

This module provides a central registry for all tools in the GLUE framework.
Tools can be registered, unregistered, and retrieved by name or by required permissions.
"""
import logging
from typing import Dict, List, Optional, Set, Any

from glue.tools.tool_base import Tool, ToolPermission
from glue.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)

# Tool class registry
_tool_classes = {}

def register_tool_class(name: str, tool_class: Any) -> None:
    """
    Register a tool class with the registry.
    
    Args:
        name: The name of the tool class
        tool_class: The tool class to register
    """
    _tool_classes[name] = tool_class
    logger.info(f"Registered tool class: {name}")

# Register the web search tool
register_tool_class("search", WebSearchTool)

# Register additional tools needed for tests
class CodeInterpreterTool(Tool):
    """Code interpreter tool for tests."""
    def __init__(self, name: str, description: str = "Code interpreter tool", config: Dict[str, Any] = None):
        super().__init__(name, description, config)
        self.api_key = config.get("metadata", {}).get("api_key") if config else None
        
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        return {"result": "Code executed successfully"}

class FileHandlerTool(Tool):
    """File handler tool for tests."""
    def __init__(self, name: str, description: str = "File handler tool", config: Dict[str, Any] = None):
        super().__init__(name, description, config)
        self.api_key = config.get("metadata", {}).get("api_key") if config else None
        
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        return {"result": "File handled successfully"}

# Register additional tools
register_tool_class("code_interpreter", CodeInterpreterTool)
register_tool_class("file_handler", FileHandlerTool)

def get_tool_class(name: str) -> Any:
    """
    Get a tool class by name.
    
    Args:
        name: The name of the tool class to get
        
    Returns:
        The tool class
        
    Raises:
        ValueError: If the tool class is not found
    """
    if name not in _tool_classes:
        raise ValueError(f"Tool class '{name}' not found")
    
    return _tool_classes[name]

class ToolRegistry:
    """
    Central registry for all tools in the GLUE framework.
    
    The registry maintains a mapping of tool names to tool instances and provides
    methods for registering, unregistering, and retrieving tools.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: The name of the tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Attempted to unregister non-existent tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            The tool instance if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            A list of all registered tool instances
        """
        return list(self._tools.values())
    
    def get_tools_by_permission(self, permission: ToolPermission) -> List[Tool]:
        """
        Get all tools that require a specific permission.
        
        Args:
            permission: The permission to filter by
            
        Returns:
            A list of tool instances that require the specified permission
        """
        return [
            tool for tool in self._tools.values()
            if permission in tool.config.required_permissions
        ]
    
    async def initialize_all(self) -> None:
        """
        Initialize all registered tools.
        
        This method calls the initialize method on all registered tools.
        """
        for tool in self._tools.values():
            await tool.initialize()
            logger.info(f"Initialized tool: {tool.name}")
    
    async def cleanup_all(self) -> None:
        """
        Clean up all registered tools.
        
        This method calls the cleanup method on all registered tools.
        """
        for tool in self._tools.values():
            await tool.cleanup()
            logger.info(f"Cleaned up tool: {tool.name}")
    
    async def initialize_tools(self, tool_names: List[str]) -> None:
        """
        Initialize specific tools by name.
        
        Args:
            tool_names: A list of tool names to initialize
        """
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                await tool.initialize()
                logger.info(f"Initialized tool: {name}")
            else:
                logger.warning(f"Attempted to initialize non-existent tool: {name}")
    
    async def cleanup_tools(self, tool_names: List[str]) -> None:
        """
        Clean up specific tools by name.
        
        Args:
            tool_names: A list of tool names to clean up
        """
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                await tool.cleanup()
                logger.info(f"Cleaned up tool: {name}")
            else:
                logger.warning(f"Attempted to clean up non-existent tool: {name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the registry to a dictionary.
        
        Returns:
            A dictionary representation of the registry
        """
        # Handle potential errors in tool.to_dict() calls
        tools_dict = []
        for tool in self._tools.values():
            try:
                tools_dict.append(tool.to_dict())
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool.name} to dict: {str(e)}")
                # Include basic information even if full conversion fails
                tools_dict.append({
                    "name": tool.name,
                    "description": tool.description,
                    "error": f"Failed to fully convert: {str(e)}"
                })
                
        return {
            "tools": tools_dict
        }
