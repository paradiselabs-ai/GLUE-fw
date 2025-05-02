"""
Unit tests for the agent loop system.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import logging

from glue.core.agent_loop import AgentLoop, TeamLoopCoordinator, AgentState
from glue.core.model import Model
from glue.core.types import AdhesiveType

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = MagicMock(spec=Model)
    model.name = "test_model"
    
    # Create a mock for the generate method that returns a string
    async def mock_generate(prompt):
        return "test response"
    
    model.generate = mock_generate
    
    return model


@pytest.fixture
def agent_loop(mock_model):
    """Create an agent loop for testing"""
    loop = AgentLoop("test_agent", "test_team", mock_model)
    
    # Register a test tool that is async
    async def test_tool(param1, param2=None):
        return {"result": f"Tool executed with {param1}, {param2}"}
    
    loop.register_tool("test_tool", test_tool)
    
    # Register a test adhesive
    loop.register_adhesive("test_adhesive", AdhesiveType.GLUE)
    
    return loop


@pytest.fixture
def team_coordinator():
    """Create a team coordinator with mock agents for testing"""
    coordinator = TeamLoopCoordinator("test_team")
    
    # Create mock agent loops
    agent1 = MagicMock(spec=AgentLoop)
    agent1.agent_id = "agent1"
    agent1.get_status.return_value = {"agent_id": "agent1", "state": AgentState.IDLE}
    
    agent2 = MagicMock(spec=AgentLoop)
    agent2.agent_id = "agent2"
    agent2.get_status.return_value = {"agent_id": "agent2", "state": AgentState.IDLE}
    
    # Add agents to the coordinator
    coordinator.add_agent(agent1)
    coordinator.add_agent(agent2)
    
    return coordinator, agent1, agent2


@pytest.mark.asyncio
async def test_state_transition_to_observing(agent_loop):
    """Test that the agent loop can transition to the observing state"""
    # Mock the internal methods to avoid actual execution
    agent_loop._observe_phase = MagicMock(return_value=asyncio.Future())
    agent_loop._observe_phase.return_value.set_result(None)
    
    # Start with initial state
    assert agent_loop.state == AgentState.IDLE
    
    # Transition to observing
    agent_loop._set_state(AgentState.OBSERVING)
    
    # Verify state change
    assert agent_loop.state == AgentState.OBSERVING


@pytest.mark.asyncio
async def test_tool_execution(agent_loop):
    """Test that the agent can execute tools"""
    # Create a test action
    action = {
        "id": "test_action",
        "type": "tool_use",
        "tool": "test_tool",
        "parameters": {"param1": "value1"}
    }
    
    # Define a proper async test tool function
    async def test_tool_func(**params):
        return {"result": f"Tool executed with {params.get('param1')}"}
    
    # Register our test tool
    agent_loop.register_tool("test_tool", test_tool_func)
    
    # Execute the action
    result = await agent_loop._execute_action(action)
    
    # Verify the result
    assert result["success"] is True
    assert "Tool executed with value1" in str(result["result"])


@pytest.mark.asyncio
async def test_error_handling(agent_loop):
    """Test that the agent handles errors gracefully"""
    # Register a failing tool
    def failing_tool():
        raise Exception("Tool failed")
    
    agent_loop.register_tool("failing_tool", failing_tool)
    
    # Create a test action
    action = {
        "id": "test_action",
        "type": "tool_use",
        "tool": "failing_tool",
        "parameters": {}
    }
    
    # Execute the action
    result = await agent_loop._execute_action(action)
    
    # Verify the error was captured
    assert "error" in str(result)
    assert "Tool failed" in str(result)


def test_status_reporting(agent_loop):
    """Test that the agent reports its status correctly"""
    # Get the status
    status = agent_loop.get_status()
    
    # Verify the status
    assert status["agent_id"] == "test_agent"
    assert status["team_id"] == "test_team"
    assert status["state"] == AgentState.IDLE


def test_agent_registration(team_coordinator):
    """Test that agents are registered correctly"""
    coordinator, agent1, agent2 = team_coordinator
    
    # Verify the agents are registered
    assert "agent1" in coordinator.agents
    assert "agent2" in coordinator.agents
    assert len(coordinator.agents) == 2


def test_status_reporting_coordinator(team_coordinator):
    """Test that the coordinator reports its status correctly"""
    coordinator, _, _ = team_coordinator
    
    # Get the status
    status = coordinator.get_status()
    
    # Verify the status
    assert status["team_id"] == "test_team"
    assert len(status["agents"]) == 2
    assert "agent1" in status["agents"]
    assert "agent2" in status["agents"]


def test_termination(team_coordinator):
    """Test that the coordinator terminates all agents"""
    coordinator, agent1, agent2 = team_coordinator
    
    # Terminate the coordinator
    coordinator.terminate("test")
    
    # Verify that all agents were terminated
    assert agent1.terminate.called
    assert agent2.terminate.called
    assert agent1.terminate.call_args[0][0] == "test"
    assert agent2.terminate.call_args[0][0] == "test"
