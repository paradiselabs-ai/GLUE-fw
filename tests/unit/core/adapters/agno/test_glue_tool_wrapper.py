# tests/unit/core/adapters/agno/test_glue_tool_wrapper.py
import pytest
import asyncio
from glue.tools.tool_base import Tool, ToolConfig
from glue.enums import AdhesiveType, ToolPermission
from glue.core.adapters.agno.adapter import create_agno_tool_from_glue_tool # This will fail initially

# Dummy GLUE Tool for testing
class DummyGlueTool(Tool):
    """A simple dummy tool for testing the wrapper."""
    def __init__(self, name="dummy_tool", description="Dummy description"):
        config = ToolConfig(
            adhesive_types={AdhesiveType.GLUE},
            required_permissions={ToolPermission.READ}
        )
        super().__init__(name=name, description=description, config=config)
        self.executed_with = None

    async def _execute(self, *args, **kwargs) -> str:
        """Dummy execution logic."""
        self.executed_with = {"args": args, "kwargs": kwargs}
        return f"Executed {self.name} with {args} and {kwargs}"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_agno_tool_from_glue_tool():
    """
    Test that create_agno_tool_from_glue_tool correctly wraps a GLUE Tool
    into an Agno-compatible callable function.
    """
    glue_tool_instance = DummyGlueTool()
    tool_name = "wrapped_dummy_tool"

    # --- This call assumes the function exists ---
    agno_tool_func = create_agno_tool_from_glue_tool(glue_tool_instance, tool_name)

    # Assert basic properties
    assert callable(agno_tool_func)
    assert agno_tool_func.__name__ == tool_name
    assert agno_tool_func.__doc__ == glue_tool_instance.description # Use actual desc

    # Test execution
    test_kwargs = {"param": "value", "another_param": 123}
    result = await agno_tool_func(**test_kwargs)

    # Assert execution result
    expected_result = f"Executed {glue_tool_instance.name} with () and {test_kwargs}"
    assert result == expected_result

    # Assert that the original tool's execute method was called correctly
    assert glue_tool_instance.executed_with is not None
    assert glue_tool_instance.executed_with["args"] == ()
    assert glue_tool_instance.executed_with["kwargs"] == test_kwargs
