"""
PydanticAI integration for GLUE tools.

This module provides a base class for tools that use Pydantic models
for input and output validation.
"""
import logging
from typing import Dict, Any, Optional, Type, Union, get_type_hints
from pydantic import BaseModel, ValidationError

from .tool_base import Tool, ToolConfig

logger = logging.getLogger(__name__)

class PydanticValidatedTool(Tool):
    """
    Base class for tools that use Pydantic models for input and output validation.
    
    This class extends the base Tool class to add input and output validation
    using Pydantic models. This ensures that tool inputs and outputs conform
    to the expected schema.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        output_schema: Optional[Type[BaseModel]] = None,
        config: Optional[Union[ToolConfig, Dict[str, Any]]] = None
    ):
        """
        Initialize a new PydanticValidatedTool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            input_schema: Pydantic model class for input validation
            output_schema: Optional Pydantic model class for output validation
            config: Tool configuration
        """
        super().__init__(name, description, config)
        
        self.input_schema = input_schema
        self.output_schema = output_schema
        
        # Update tool inputs based on the input schema
        self._update_inputs_from_schema()
    
    def _update_inputs_from_schema(self) -> None:
        """
        Update the tool inputs based on the input schema.
        
        This extracts field information from the Pydantic model to provide
        better documentation for the tool inputs.
        """
        if not self.input_schema:
            return
            
        # Get field information from the schema
        schema = self.input_schema.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Update inputs with schema information
        self.inputs = {}
        for name, prop in properties.items():
            self.inputs[name] = {
                "type": prop.get("type", "string"),
                "description": prop.get("description", ""),
                "required": name in required,
                "default": prop.get("default", None),
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against the input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validated input data as a dictionary
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Create a model instance to validate the input
            validated = self.input_schema(**input_data)
            # Return as a dictionary
            return validated.model_dump()
        except ValidationError as e:
            # Convert validation error to a more user-friendly message
            error_msg = f"Input validation failed for tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def validate_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output data against the output schema.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            Validated output data as a dictionary
            
        Raises:
            ValueError: If validation fails
        """
        if not self.output_schema:
            return output_data
            
        try:
            # Create a model instance to validate the output
            validated = self.output_schema(**output_data)
            # Return as a dictionary
            return validated.model_dump()
        except ValidationError as e:
            # Convert validation error to a more user-friendly message
            error_msg = f"Output validation failed for tool '{self.name}': {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with input validation.
        
        Args:
            **kwargs: Input arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If input validation fails
        """
        # Convert kwargs to a dictionary
        input_data = dict(kwargs)
        
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Execute the tool with validated input
        result = await self._execute(validated_input)
        
        # Validate output if an output schema is provided
        if self.output_schema:
            result = self.validate_output(result)
            
        return result
