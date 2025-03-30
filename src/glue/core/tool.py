"""
Tool creation and management for the GLUE framework.

This module provides functions for creating and managing tools in the GLUE framework.
"""

import importlib
import logging
from typing import Dict, Any, Optional, Type, Union

from ..tools.tool_base import Tool, ToolConfig
from .types import AdhesiveType

# Set up logging
logger = logging.getLogger("glue.tool")


def create_tool(config: Dict[str, Any]) -> Tool:
    """Create a tool from a configuration dictionary.
    
    Args:
        config: Tool configuration dictionary
        
    Returns:
        A Tool instance
        
    Raises:
        ValueError: If the configuration is invalid
        ImportError: If the tool class cannot be imported
    """
    if not config:
        raise ValueError("Tool configuration cannot be empty")
    
    name = config.get("name")
    if not name:
        raise ValueError("Tool name is required")
    
    # Handle the case where config contains a ToolConfig object
    tool_config = config.get("config")
    
    # Get the tool type
    tool_type = config.get("type", "search")  # Default to search if not specified
    
    # Get the tool class
    try:
        from ..tools.tool_registry import get_tool_class
        tool_class = get_tool_class(tool_type)
    except (ImportError, ValueError) as e:
        # Fallback to a basic implementation if the registry isn't available
        from ..tools.tool_base import BasicTool
        tool_class = BasicTool
    
    # Create and return the tool
    if hasattr(tool_config, "name") and hasattr(tool_config, "description"):
        # This is a ToolConfig-like object
        return tool_class(
            name=name,
            description=tool_config.description,
            config={}  # Empty config to avoid errors
        )
    else:
        # Otherwise, follow the original implementation
        description = config.get("description", "")
        if not description:
            # Use the name as description if not provided
            description = f"Tool: {name}"
        
        return tool_class(
            name=name,
            description=description,
            config=config.get("config", {})
        )
