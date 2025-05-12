import pytest
from glue.core.adapters.agno.adapter import GlueModelAdapter


class DummyGlueModel:
    """
    A dummy GLUE model with a generate() coroutine to test GlueModelAdapter.
    """
    def __init__(self):
        self.name = "dummy"
        self.model_name = "dummy-model"
        self.generated_task = None

    async def generate(self, task):
        """Store the task and return a formatted response."""
        self.generated_task = task
        return f"response: {task}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_glue_model_adapter_arun_calls_generate():
    """
    The GlueModelAdapter should call the underlying GLUE model's generate() method
    and return its output.
    """
    dummy = DummyGlueModel()
    adapter = GlueModelAdapter(dummy)
    result = await adapter.arun("test-task")

    # The adapter should return the model's response
    assert result == "response: test-task"
    # The dummy model should have recorded the task
    assert dummy.generated_task == "test-task"
