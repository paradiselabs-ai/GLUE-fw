"""
Model module for the GLUE framework.

This module provides the Model class, which is a concrete implementation of the BaseModel
that can be used in teams.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Type, AsyncIterable, Set
import inspect

from .base_model import BaseModel
from .schemas import Message, ToolCall, ToolResult, ModelConfig
from .types import AdhesiveType

# Set up logging
logger = logging.getLogger("glue.model")


class Model(BaseModel):
    """Concrete implementation of the model class that can be used in teams."""
    
    def __init__(self, config=None, **kwargs):
        """Initialize a new model.
        
        Args:
            config: Model configuration
            **kwargs: Additional keyword arguments
        """
        # Handle config conversion
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            # Attempt conversion if possible, otherwise initialize empty
            try:
                config = vars(config)
            except TypeError:
                 # Handle cases where config might be a simple type or non-dict-like object
                 print(f"Warning: Model config is not a dictionary or easily convertible: {type(config)}. Initializing provider might fail.")
                 config = {}

        # Ensure name is always set
        if 'name' not in config and 'name' in kwargs:
            config['name'] = kwargs['name']
            
        # Initialize the base model
        super().__init__(config)
        
        # Set up adhesives for compatibility
        self.adhesives = set()
        adhesives = config.get('adhesives', [])
        logger.debug(f"Model {config.get('name', 'unnamed')}: Raw adhesives from config: {adhesives}")
        
        for adhesive in adhesives:
            # Convert string adhesives to AdhesiveType enum values
            if isinstance(adhesive, str):
                try:
                    # Try to get the enum value by name (case-insensitive)
                    adhesive_type = next(
                        (at for at in AdhesiveType if at.value.lower() == adhesive.lower()),
                        None
                    )
                    if adhesive_type:
                        self.adhesives.add(adhesive_type)
                        logger.debug(f"Added adhesive {adhesive_type} from string '{adhesive}'")
                    else:
                        logger.warning(f"Unknown adhesive type: {adhesive}")
                except Exception as e:
                    logger.warning(f"Error converting adhesive '{adhesive}' to enum: {e}")
            else:
                # Assume it's already an AdhesiveType enum value
                self.adhesives.add(adhesive)
                logger.debug(f"Added adhesive {adhesive}")
            
        # Add GLUE adhesive by default if no adhesives specified
        if not self.adhesives:
            self.adhesives.add(AdhesiveType.GLUE)
            logger.debug("No adhesives specified, added default GLUE adhesive")
        
        logger.info(f"Model {config.get('name', 'unnamed')} initialized with adhesives: {self.adhesives}")
    
    def has_adhesive(self, adhesive: AdhesiveType) -> bool:
        """Check if this model supports the given adhesive type.
        
        Args:
            adhesive: Adhesive type to check
            
        Returns:
            True if the model supports the adhesive, False otherwise
        """
        return adhesive in self.adhesives
    
    def add_tool_sync(self, name: str, tool: Any) -> None:
        """Add a tool to the model synchronously.
        
        This is a synchronous version of add_tool for use during setup.
        
        Args:
            name: Name of the tool
            tool: Tool to add
        """
        # Add to tools dictionary
        if not hasattr(self, 'tools'):
            self.tools = {}
            
        self.tools[name] = tool
        
        # Format tool for provider if needed
        formatted_tool = self._format_tool_for_provider(name, tool)
        
        # Add to provider tools
        if hasattr(self, '_provider_tools'):
            self._provider_tools[name] = formatted_tool
            
        logger.debug(f"Added tool {name} to model {self.name} synchronously")

    def _format_tool_for_provider(self, name: str, tool: Any) -> Dict[str, Any]:
        """Format a tool for the provider.
        
        Args:
            name: Name of the tool
            tool: Tool to format
            
        Returns:
            Formatted tool
        """
        # Default format for tools
        formatted_tool = {
            "name": name,
            "description": getattr(tool, "description", f"Tool: {name}")
        }
        
        # Add parameters if available
        if hasattr(tool, "parameters"):
            formatted_tool["parameters"] = tool.parameters
            # Ensure 'type' is set for Gemini compatibility
            if "type" not in formatted_tool["parameters"]:
                formatted_tool["parameters"]["type"] = "object"
        elif hasattr(tool, "get_parameters") and callable(tool.get_parameters):
            try:
                formatted_tool["parameters"] = tool.get_parameters()
                # Ensure 'type' is set for Gemini compatibility
                if "type" not in formatted_tool["parameters"]:
                    formatted_tool["parameters"]["type"] = "object"
            except Exception as e:
                logger.warning(f"Error getting parameters for tool {name}: {e}")
                formatted_tool["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
        else:
            # Default parameters structure
            formatted_tool["parameters"] = {
                "type": "object",  # Required by Gemini API
                "properties": {},
                "required": []
            }
            
            # Try to infer parameters from the execute method if available
            if hasattr(tool, "execute") and callable(tool.execute):
                try:
                    sig = inspect.signature(tool.execute)
                    for param_name, param in sig.parameters.items():
                        # Skip self and kwargs
                        if param_name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
                            continue
                            
                        # Add parameter to properties
                        formatted_tool["parameters"]["properties"][param_name] = {
                            "type": "string",  # Default to string
                            "description": f"Parameter: {param_name}"
                        }
                        
                        # Add to required list if no default value
                        if param.default == inspect.Parameter.empty:
                            formatted_tool["parameters"]["required"].append(param_name)
                except Exception as e:
                    logger.warning(f"Error inferring parameters for tool {name}: {e}")
        
        return formatted_tool

    async def setup(self) -> None:
        """Set up the model by initializing any required resources.
        
        This is a placeholder implementation for test compatibility.
        In a real implementation, this would initialize any required resources.
        """
        # Nothing to do in the base implementation
        pass
