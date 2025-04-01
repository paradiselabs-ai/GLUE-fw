"""
Utility functions for the GLUE framework.

This module contains helper functions for creating models, tools, and other
components used throughout the framework.
"""
from typing import Dict, Any, Optional, Union, List
import logging

from .model import Model, BaseModel
from ..tools.tool_base import Tool
from ..tools.tool_registry import CodeInterpreterTool, FileHandlerTool, WebSearchTool
from .schemas import ModelConfig, ToolConfig

logger = logging.getLogger("glue.utils")

def create_model(config: Dict[str, Any]) -> Model:
    """Create a model from a configuration dictionary.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_name = config.get("name", "unnamed_model")
    model_config = config.get("config")
    adhesives = config.get("adhesives", [])
    role = config.get("role", "")
    
    # Create the model
    model = Model(model_config)
    
    # Set model properties
    model.name = model_name
    model.adhesives = adhesives
    model.role = role
    
    return model

def create_tool(config: Dict[str, Any]) -> Tool:
    """Create a tool from a configuration dictionary.
    
    Args:
        config: Tool configuration dictionary
        
    Returns:
        Initialized tool
    """
    # Use the existing create_tool function from the tool module
    from .tool import create_tool as core_create_tool
    return core_create_tool(config)
