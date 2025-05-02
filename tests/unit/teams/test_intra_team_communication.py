import pytest
import asyncio
from typing import Dict, List, Any, Optional

from glue.core.teams import Team
from glue.core.types import AdhesiveType, TeamConfig, Message, ToolResult
from glue.core.model import Model
from tests.unit.teams.test_helpers import SimpleTool, SimpleModel


@pytest.fixture
def test_model():
    return SimpleModel(name="test_model")


@pytest.fixture
def lead_model():
    model = SimpleModel(name="lead_model")
    model.generate_response = "Lead response"
    return model


@pytest.fixture
def test_tool():
    return SimpleTool(name="test_tool")


@pytest.fixture
def team_config():
    return TeamConfig(
        name="test_team",
        lead="lead_model",
        members=["member1", "member2"],
        tools=["tool1", "tool2"]
    )


@pytest.fixture
def team(team_config):
    return Team(name="test_team", config=team_config)


class TestIntraTeamCommunication:
    """Tests for intra-team communication functionality."""

    @pytest.mark.asyncio
    async def test_add_member(self, team, test_model):
        """Test adding a model to the team."""
        await team.add_member(test_model, role="member")
        
        assert test_model.name in team.models
        assert test_model.team == team
        assert test_model.name in team.config.members

    @pytest.mark.asyncio
    async def test_add_lead_member(self, team, test_model):
        """Test adding a lead model to the team."""
        await team.add_member(test_model, role="lead")
        
        assert test_model.name in team.models
        assert test_model.team == team
        assert team.config.lead == test_model.name

    @pytest.mark.asyncio
    async def test_add_tool(self, team, test_model, test_tool):
        """Test adding a tool to the team."""
        await team.add_member(test_model)
        await team.add_tool("test_tool", test_tool, AdhesiveType.GLUE)
        
        assert test_tool in team.tools
        assert team.tool_bindings["test_tool"] == AdhesiveType.GLUE
        assert "test_tool" in test_model.tools

    @pytest.mark.asyncio
    async def test_share_result_glue(self, team):
        """Test sharing a tool result with GLUE adhesive."""
        result = ToolResult(
            tool_name="test_tool",
            result={"result": "success"},
            adhesive=AdhesiveType.GLUE,
            metadata={}
        )
        
        await team.share_result("test_tool", result)
        
        assert "test_tool" in team.shared_results
        assert team.shared_results["test_tool"] == result

    @pytest.mark.asyncio
    async def test_share_result_velcro(self, team):
        """Test sharing a tool result with VELCRO adhesive."""
        result = ToolResult(
            tool_name="test_tool",
            result={"result": "success"},
            adhesive=AdhesiveType.VELCRO,
            metadata={}
        )
        
        await team.share_result("test_tool", result)
        
        assert "test_tool" not in team.shared_results

    @pytest.mark.asyncio
    async def test_process_message_with_source(self, team, test_model):
        """Test processing a message with a specified source model."""
        await team.add_member(test_model)
        
        message = {"content": "Hello", "metadata": {}}
        response = await team.process_message(message, source_model=test_model.name)
        
        assert response == "Test response"
        assert test_model.generate_called
        assert len(team.conversation_history) == 2
        assert team.conversation_history[0].content == "Hello"
        assert team.conversation_history[1].content == "Test response"

    @pytest.mark.asyncio
    async def test_process_message_with_lead(self, team, lead_model):
        """Test processing a message using the lead model."""
        await team.add_member(lead_model, role="lead")
        
        message = {"content": "Hello", "metadata": {}}
        response = await team.process_message(message)
        
        assert response == "Lead response"
        assert lead_model.generate_called

    @pytest.mark.asyncio
    async def test_process_message_no_source(self, team, lead_model):
        """Test processing a message with no source model."""
        await team.add_member(lead_model, role="lead")
        
        message = {"content": "Hello", "metadata": {}}
        response = await team.process_message(message)
        
        assert response == "Lead response"
        assert lead_model.generate_called

    @pytest.mark.asyncio
    async def test_direct_communication(self, team):
        """Test direct communication between team members."""
        model1 = SimpleModel(name="model1")
        model1.generate_response = "Response from model1"
        
        model2 = SimpleModel(name="model2")
        model2.generate_response = "Response from model2"
        
        await team.add_member(model1)
        await team.add_member(model2)
        
        # Test direct communication
        message = {"content": "Hello from model1", "metadata": {}}
        response = await team.direct_communication("model1", "model2", message)
        
        assert response == "Response from model2"
        assert model2.generate_called
        
        # Verify conversation history
        assert len(team.conversation_history) == 1
        assert team.conversation_history[0].role == "model"

    @pytest.mark.asyncio
    async def test_get_model_tools(self, team, test_model, test_tool):
        """Test getting tools available to a model."""
        await team.add_member(test_model)
        await test_model.add_tool("test_tool", test_tool)
        
        tools = team.get_model_tools(test_model.name)
        
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_get_shared_results(self, team):
        """Test getting shared results."""
        result = ToolResult(
            tool_name="test_tool",
            result={"result": "success"},
            adhesive=AdhesiveType.GLUE,
            metadata={}
        )
        
        await team.share_result("test_tool", result)
        
        shared_results = team.get_shared_results()
        
        assert "test_tool" in shared_results
        assert shared_results == team.shared_results

    @pytest.mark.asyncio
    async def test_cleanup(self, team, test_model, test_tool):
        """Test team cleanup."""
        await team.add_member(test_model)
        await team.add_tool("test_tool", test_tool)
        
        result = ToolResult(
            tool_name="test_tool",
            result={"result": "success"},
            adhesive=AdhesiveType.GLUE,
            metadata={}
        )
        
        await team.share_result("test_tool", result)
        
        message = {"content": "Hello", "metadata": {}}
        await team.process_message(message, source_model=test_model.name)
        
        await team.cleanup()
        
        assert test_tool.cleaned_up
