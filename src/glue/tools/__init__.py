"""GLUE Tools Module

This module provides the base classes and registration mechanisms for tools
that can be used by GLUE applications.
"""

from typing import Dict, Any, Callable, List, Optional
import logging
import inspect

# Global registry of tools
_tool_registry = {}
logger = logging.getLogger("glue.tools")

# Import tool implementations
# We need to be careful with the import order to avoid circular imports
from .tool_base import Tool, ToolConfig, ToolPermission
from .file_handler_tool import FileHandlerTool, FileOperation
from .code_interpreter_tool import CodeInterpreterTool, CodeLanguage

class SimpleBaseTool:
    """Base class for simple tools that can be used in GLUE applications.
    
    A simple tool has a single function that takes input and returns output.
    """
    
    def __init__(self, name: str, description: str, func: Callable):
        """Initialize a new SimpleBaseTool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            func: Function that implements the tool's functionality
        """
        self.name = name
        self.description = description
        self.func = func
        
        # Extract parameter information from the function
        sig = inspect.signature(func)
        self.params = {}
        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue
                
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else None
            required = param.default == inspect.Parameter.empty
            
            self.params[param_name] = {
                'type': param_type,
                'required': required,
                'default': default
            }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the provided arguments.
        
        Args:
            **kwargs: Arguments to pass to the tool function
            
        Returns:
            Result of the tool execution
        """
        logger.info(f"Executing tool: {self.name}")
        
        # Validate arguments
        for param_name, param_info in self.params.items():
            if param_info['required'] and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        # Add default values for missing optional parameters
        for param_name, param_info in self.params.items():
            if not param_info['required'] and param_name not in kwargs:
                kwargs[param_name] = param_info['default']
        
        # Execute the function
        result = self.func(**kwargs)
        
        # Handle async functions
        if inspect.isawaitable(result):
            result = await result
            
        return result


def register_tool(name: str, description: str = None):
    """Decorator to register a function as a tool.
    
    Args:
        name: Name of the tool
        description: Description of what the tool does
    
    Returns:
        Decorator function
    """
    def decorator(func):
        nonlocal description
        if description is None:
            description = func.__doc__ or f"Tool: {name}"
            
        tool = SimpleBaseTool(name, description, func)
        _tool_registry[name] = tool
        logger.info(f"Registered tool: {name}")
        return func
    return decorator


def get_tool(name: str) -> Optional[SimpleBaseTool]:
    """Get a tool by name.
    
    Args:
        name: Name of the tool to get
        
    Returns:
        The tool, or None if not found
    """
    return _tool_registry.get(name)


def list_tools() -> List[str]:
    """List all registered tools.
    
    Returns:
        List of tool names
    """
    return list(_tool_registry.keys())
