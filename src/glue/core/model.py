"""
Model module for the GLUE framework.

This module provides the Model class, which is a concrete implementation of the BaseModel
that can be used in teams.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Type, AsyncIterable, Set

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
        for adhesive in adhesives:
            self.adhesives.add(adhesive)
            
        # Add GLUE adhesive by default if no adhesives specified
        if not self.adhesives:
            self.adhesives.add(AdhesiveType.GLUE)
    
    def has_adhesive(self, adhesive: AdhesiveType) -> bool:
        """Check if this model supports the given adhesive type.
        
        Args:
            adhesive: Adhesive type to check
            
        Returns:
            True if the model supports the adhesive, False otherwise
        """
        return adhesive in self.adhesives
    
    async def setup(self) -> None:
        """Set up the model by initializing any required resources.
        
        This is a placeholder implementation for test compatibility.
        In a real implementation, this would initialize any required resources.
        """
        # Nothing to do in the base implementation
        pass
