# tests/unit/core/adapters/agno/test_message_flow_integration.py
import pytest
import asyncio
from unittest.mock import MagicMock

from glue.core.teams import GlueTeam as Team
from glue.core.schemas import TeamConfig

# Dummy AgnoTeam for testing message flow integration
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
async def test_receive_information_uses_agno_team():
    """
    Test that Team.receive_information delegates to agno_team.arun when an agno_team is present.
    """
    # 1. Arrange: Create a GLUE Team with an agno_team
    team_config = TeamConfig(name="TestTeam", lead="test_lead")
    glue_team = Team(name="TestTeam", config=team_config)
    
    # Add an agno_team to the team
    dummy_agno = DummyAgnoTeam(name="AgnoTeam")
    glue_team.agno_team = dummy_agno
    
    # Test message content
    source_team = "SourceTeam"
    test_message = "Important information from source team"
    
    # 2. Act: Call receive_information directly
    result = await glue_team.receive_information(source_team, test_message)
    
    # Give asyncio a chance to run the created task if any
    await asyncio.sleep(0.01)
    
    # 3. Assert: Check if agno_team.arun was called with the message
    assert dummy_agno.arun.called, "AgnoTeam.arun was not called"
    assert dummy_agno.arun_called_with == test_message, f"AgnoTeam.arun was called with wrong message: {dummy_agno.arun_called_with}"
    assert result["success"] is True, f"receive_information returned failure: {result}"
