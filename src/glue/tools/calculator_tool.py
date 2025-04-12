"""
Calculator tool for the GLUE framework.

This module implements a simple calculator tool that demonstrates
the use of PydanticValidatedTool for input and output validation.
"""
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

from .pydantic_validated_tool import PydanticValidatedTool
from .tool_base import ToolConfig, ToolPermission

logger = logging.getLogger(__name__)

class Operation(str, Enum):
    """Supported calculator operations"""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool"""
    operation: Operation = Field(
        description="The operation to perform (add, subtract, multiply, divide)"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class CalculatorOutput(BaseModel):
    """Output schema for the calculator tool"""
    result: float = Field(description="Calculation result")
    operation: str = Field(description="Operation performed")
    expression: str = Field(description="Expression calculated")

class CalculatorTool(PydanticValidatedTool):
    """Tool for performing basic arithmetic operations"""
    
    def __init__(
        self,
        name: str = "calculator",
        description: str = "Perform basic arithmetic operations",
        config: Optional[Union[ToolConfig, Dict[str, Any]]] = None
    ):
        """
        Initialize the calculator tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            config: Tool configuration
        """
        # Create default tool config if not provided
        if config is None:
            config = ToolConfig(
                required_permissions=set()  # No special permissions needed
            )
        
        super().__init__(
            name, 
            description, 
            input_schema=CalculatorInput,
            output_schema=CalculatorOutput,
            config=config
        )
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a calculation.
        
        Args:
            input_data: Dictionary containing 'operation', 'a', and 'b'
            
        Returns:
            Dictionary containing the calculation result
            
        Raises:
            ValueError: If division by zero is attempted
        """
        # Input is already validated by PydanticValidatedTool
        operation = input_data["operation"]
        a = input_data["a"]
        b = input_data["b"]
        
        # Perform the calculation
        if operation == Operation.ADD:
            result = a + b
            expression = f"{a} + {b}"
        elif operation == Operation.SUBTRACT:
            result = a - b
            expression = f"{a} - {b}"
        elif operation == Operation.MULTIPLY:
            result = a * b
            expression = f"{a} * {b}"
        elif operation == Operation.DIVIDE:
            if b == 0:
                raise ValueError("Division by zero is not allowed")
            result = a / b
            expression = f"{a} / {b}"
        else:
            # This should never happen due to Pydantic validation
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Return the result
        return {
            "result": result,
            "operation": operation,
            "expression": expression
        }
