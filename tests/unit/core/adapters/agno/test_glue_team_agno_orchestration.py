# tests/unit/core/adapters/agno/test_glue_team_agno_orchestration.py
import pytest
import asyncio
from unittest.mock import MagicMock # Using MagicMock for simplicity in spy behavior

from glue.core.team import Team
from glue.core.schemas import TeamConfig

# Dummy AgnoTeam for testing orchestration trigger
class DummyAgnoTeam:
    def __init__(self, name="dummy_agno_team"):
        self.name = name
        self.arun_called_with = None
        self.arun_call_count = 0
        # Using MagicMock to easily check calls for an async method
        self.arun = MagicMock(side_effect=self._async_arun_impl)

    async def _async_arun_impl(self, task_input):
        self.arun_called_with = task_input
        self.arun_call_count += 1
        return f"Agno team {self.name} processed: {task_input}"

    def reset_mock(self):
        self.arun_called_with = None
        self.arun_call_count = 0
        self.arun.reset_mock()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_agent_loops_triggers_agno_team_arun():
    """
    Test that Team.start_agent_loops, when given initial_input,
    triggers the arun method of an associated agno_team.
    (This test will fail initially as start_agent_loops is not yet adapted).
    """
    # 1. Arrange: Create a GLUE Team and a DummyAgnoTeam
    # We need to provide a minimal valid app and config for Team initialization
    mock_app = MagicMock()
    mock_app.config = MagicMock()
    mock_app.config.app_name = "TestApp"
    mock_app.adhesive_system = MagicMock()
    mock_app.tool_registry = MagicMock()
    mock_app.model_registry = MagicMock()
    mock_app.get_tool_config.return_value = None

    team_config = TeamConfig(name="TestGlueTeam", lead="lead_model")
    glue_team = Team(name="TestGlueTeam", config=team_config, app=mock_app)
    
    dummy_agno_orchestrator = DummyAgnoTeam(name="OrchestratorAgnoTeam")
    glue_team.agno_team = dummy_agno_orchestrator # Manually assign for this test

    initial_task = "Run the primary objective via Agno."

    # 2. Act: Call start_agent_loops
    # The original start_agent_loops might try to start its own loops
    # For this test, we're focusing on the new Agno path
    await glue_team.start_agent_loops(initial_input=initial_task)
    
    # Give asyncio a chance to run the created task if any
    await asyncio.sleep(0.01) 

    # 3. Assert: Check if agno_team.arun was called correctly
    try:
        dummy_agno_orchestrator.arun.assert_called_once_with(initial_task)
    except AssertionError as e:
        pytest.fail(f"AgnoTeam.arun was not called as expected. Call count: {dummy_agno_orchestrator.arun.call_count}, Called with: {dummy_agno_orchestrator.arun.call_args}. Error: {e}")

    # To make it fail explicitly for the RED phase if the above doesn't catch it due to current implementation.
    # This line will be removed when the GREEN phase implementation is ready.
    # assert dummy_agno_orchestrator.arun.called, "RED PHASE: AgnoTeam.arun was not called."

    # For the initial RED phase, the above assertion will likely fail because
    # start_agent_loops doesn't know about agno_team yet.
    # We can add an explicit fail if needed, but the assertion itself should fail.
