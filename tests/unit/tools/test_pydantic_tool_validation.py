"""
Tests for PydanticAI tool validation integration.

This module tests the integration of PydanticAI validation with GLUE tools.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from glue.tools.tool_base import Tool, ToolConfig, ToolPermission
from glue.core.types import Message

# We'll test with a new PydanticValidatedTool class that doesn't exist yet
# This test should fail until we implement the class
class TestPydanticToolValidation:
    
    def test_pydantic_tool_validation_exists(self):
        """Test that the PydanticValidatedTool class exists."""
        try:
            from glue.tools.pydantic_validated_tool import PydanticValidatedTool
            assert PydanticValidatedTool is not None
        except ImportError:
            pytest.fail("PydanticValidatedTool class does not exist")
    
    def test_pydantic_tool_schema_validation(self):
        """Test that the PydanticValidatedTool validates input against a schema."""
        try:
            from glue.tools.pydantic_validated_tool import PydanticValidatedTool
            
            # Define input schema
            class WebSearchInput(BaseModel):
                query: str = Field(description="Search query")
                max_results: int = Field(default=5, description="Maximum number of results")
            
            # Create a concrete implementation for testing
            class TestTool(PydanticValidatedTool):
                async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                    return input_data
            
            # Create a tool with the schema
            tool = TestTool(
                name="test_tool",
                description="Test tool with Pydantic validation",
                input_schema=WebSearchInput
            )
            
            # This should pass validation
            valid_input = {"query": "test query", "max_results": 10}
            assert tool.validate_input(valid_input) == valid_input
            
            # This should fail validation (missing required field)
            invalid_input = {"max_results": 10}
            with pytest.raises(ValueError):
                tool.validate_input(invalid_input)
                
        except ImportError:
            pytest.fail("PydanticValidatedTool class does not exist")
    
    @pytest.mark.asyncio
    async def test_pydantic_tool_execution_with_validation(self):
        """Test that the PydanticValidatedTool validates input before execution."""
        try:
            from glue.tools.pydantic_validated_tool import PydanticValidatedTool
            
            # Define input and output schemas
            class CalcInput(BaseModel):
                a: int = Field(description="First number")
                b: int = Field(description="Second number")
                
            class CalcOutput(BaseModel):
                result: int = Field(description="Calculation result")
            
            # Create a test tool implementation
            class TestCalcTool(PydanticValidatedTool):
                def __init__(self):
                    super().__init__(
                        name="calc",
                        description="Simple calculator",
                        input_schema=CalcInput,
                        output_schema=CalcOutput
                    )
                
                async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                    # Input should already be validated by this point
                    a = input_data["a"]
                    b = input_data["b"]
                    return {"result": a + b}
            
            # Create and initialize the tool
            tool = TestCalcTool()
            await tool.initialize()
            
            # Test with valid input
            result = await tool.execute(a=5, b=3)
            assert result["result"] == 8
            
            # Test with invalid input (should raise error before execution)
            with pytest.raises(ValueError):
                await tool.execute(a="not a number", b=3)
                
        except ImportError:
            pytest.fail("PydanticValidatedTool class does not exist")
