"""
Unit tests for the GlueAgnoAdapter class.

These tests verify that the adapter correctly bridges GLUE and Agno concepts,
allowing GLUE to use Agno as its underlying execution engine.
"""

import pytest
import os
import sys

# Create a stub module for agno to avoid import errors
class StubAgent:
    """Stub implementation of Agno Agent class for testing."""
    def __init__(self, name, provider=None, model=None, config=None):
        self.name = name
        self.provider = provider
        self.model = model
        self.config = config

class StubTeam:
    """Stub implementation of Agno Team class for testing."""
    def __init__(self, name, members=None, lead=None, config=None):
        self.name = name
        self.members = members or []
        self.lead = lead
        self.config = config

class StubWorkflow:
    """Stub implementation of Agno Workflow class for testing."""
    def __init__(self, name, teams=None, config=None, description=None):
        self.name = name
        self.teams = teams or []
        self.config = config
        self.description = description
        
    def run(self, input_text=None):
        """Stub implementation that raises NotImplementedError."""
        raise NotImplementedError("Agno Workflow.run() is not implemented")

# Create a module structure for our stubs
class StubAgnoModule:
    """Stub module for agno."""
    def __init__(self):
        self.agent = type('agent', (), {'Agent': StubAgent})
        self.team = type('team', (), {'Team': StubTeam})
        self.workflow = type('workflow', (), {'Workflow': StubWorkflow})

# Add the stub module to sys.modules
# sys.modules['agno'] = StubAgnoModule()
# sys.modules['agno.agent'] = sys.modules['agno'].agent
# sys.modules['agno.team'] = sys.modules['agno'].team
# sys.modules['agno.workflow'] = sys.modules['agno'].workflow

from glue.core.adapters.agno.adapter import GlueAgnoAdapter, AgnoFunction
from glue.core.tools.tool_base import Tool as GlueTool


@pytest.fixture
def minimal_config():
    """Fixture providing a minimal configuration for testing."""
    return {
        "workflow": {
            "name": "TestApp",
            "description": "Test application"
        },
        "agents": {
            "TestAgent": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            }
        },
        "teams": {
            "TestTeam": {
                "lead": "TestAgent",
                "members": ["TestAgent"]
            }
        },
        "tools": {},
        "flows": {}
    }


@pytest.mark.unit
def test_adapter_initialization():
    """Test that the adapter can be initialized."""
    adapter = GlueAgnoAdapter()
    assert adapter.workflow is None
    assert adapter.teams == {}
    assert adapter.agents == {}


@pytest.mark.unit
def test_adapter_setup(minimal_config):
    """Test that the adapter setup method creates Agno components."""
    adapter = GlueAgnoAdapter()
    success = adapter.setup(minimal_config)
    
    assert success is True
    assert adapter.workflow is not None
    assert adapter.workflow.name == "TestApp"
    assert len(adapter.agents) == 1
    assert "TestAgent" in adapter.agents
    assert len(adapter.teams) == 1
    assert "TestTeam" in adapter.teams


@pytest.mark.unit
async def test_adapter_run(minimal_config):
    """Test that the adapter can run an Agno workflow."""
    adapter = GlueAgnoAdapter()
    # Ensure setup is called successfully before run, as run depends on a populated self.workflow
    setup_success = adapter.setup(minimal_config)
    assert setup_success is True
    assert adapter.workflow is not None # Verify workflow is created

    result = await adapter.run(config=minimal_config, interactive=False, input_data={'initial_prompt': 'Test run'})
    
    # Assertions for a successful run using the real AgnoWorkflow via the adapter
    assert result is not None
    assert isinstance(result, dict)
    assert result.get("status") == "non_interactive_execution_completed"
    assert "workflow_name" in result
    assert result["workflow_name"] == minimal_config["workflow"]["name"]
    assert "result" in result # This would be the direct output from AgnoWorkflow.run
    assert "execution_log" in result
    # Further assertions on result['result'] and result['execution_log'] would depend
    # on the actual behavior of AgnoWorkflow.run() and what it returns.


@pytest.mark.unit
def test_adapter_handles_setup_errors():
    """Test that the adapter properly handles setup errors."""
    # Create a config with invalid team reference
    config = {
        "workflow": {
            "name": "TestApp"
        },
        "agents": {
            "TestAgent": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            }
        },
        "teams": {
            "TestTeam": {
                "lead": "NonExistentAgent",  # This agent doesn't exist
                "members": ["TestAgent"]
            }
        }
    }
    
    adapter = GlueAgnoAdapter()
    success = adapter.setup(config)
    assert success is False


@pytest.mark.unit
async def test_adapter_handles_run_errors(minimal_config): # Added minimal_config fixture
    """Test that the adapter properly handles run errors."""
    # Test with an empty config that will cause setup to fail
    empty_config = {}
    
    adapter = GlueAgnoAdapter()
    result = await adapter.run(config=empty_config, interactive=False, input_data={'initial_prompt': 'Test input'})
    assert result is None
    
    # Create a special test case where setup succeeds but run fails
    # We'll do this by creating a custom exception-raising workflow
    class ErrorWorkflow:
        def __init__(self, *args, **kwargs):
            self.name = "ErrorWorkflow"
        
        async def run(self, *args, **kwargs):
            raise Exception("Test error")
    
    # Replace the stub workflow with our error workflow
    original_workflow = sys.modules['agno.workflow'].Workflow
    sys.modules['agno.workflow'].Workflow = ErrorWorkflow
    
    try:
        # Run with minimal_config (which is valid), the patched ErrorWorkflow will raise an exception
        # The 'valid_config' variable was removed as it's no longer used.
        result = await adapter.run(config=minimal_config, interactive=False, input_data={'initial_prompt': 'Test input for error'})
        assert result is None
    finally:
        # Restore the original stub
        sys.modules['agno.workflow'].Workflow = original_workflow


# Define simple placeholder GLUE tools for testing
class MyAgentSpecificTool(GlueTool):
    name: str = "MyAgentSpecificTool"
    description: str = "A tool specifically for an agent."
    async def _execute(self, query: str):
        return f"AgentSpecificTool executed with: {query}"

class MyTeamTool(GlueTool):
    name: str = "MyTeamTool"
    description: str = "A tool provided by the team."
    async def _execute(self, command: str):
        return f"TeamTool executed with: {command}"

@pytest.mark.unit
async def test_team_tools_are_propagated_to_member_agents(minimal_config):
    """
    Test that tools defined at the team level in GLUE config are made available
    to the AgnoAgents that are members of that team, in addition to their own tools.
    """
    adapter = GlueAgnoAdapter()

    # Modify the minimal_config to include specific tools and an agent in a team
    test_config = {
        "workflow": {
            "name": "TeamToolTestApp",
            "description": "Test app for team tool propagation"
        },
        "tools": {
            "agent_tool_1": {
                "class": "MyAgentSpecificTool", 
                "config": {"param": "agent_tool_param"}
            },
            "team_tool_1": {
                "class": "MyTeamTool", 
                "config": {"param": "team_tool_param"}
            }
        },
        "agents": {
            "AgentInTeam": {
                "provider": "gemini", 
                "model_name": "gemini-1.5-pro",
                "tools": ["agent_tool_1"] 
            }
        },
        "teams": {
            "TestTeamWithTools": {
                "lead": "AgentInTeam",
                "members": ["AgentInTeam"],
                "tools": ["team_tool_1"] 
            }
        },
        "flows": {}
    }

    original_create_glue_tool = adapter._create_glue_tool_instance
    created_tool_instances = {}

    def mock_create_glue_tool_instance(tool_name: str, tool_data: dict):
        if tool_name == "agent_tool_1":
            instance = MyAgentSpecificTool()
            created_tool_instances[tool_name] = instance
            return instance
        elif tool_name == "team_tool_1":
            instance = MyTeamTool()
            created_tool_instances[tool_name] = instance
            return instance
        return original_create_glue_tool(tool_name, tool_data)

    adapter._create_glue_tool_instance = mock_create_glue_tool_instance

    await adapter.setup(test_config)

    adapter._create_glue_tool_instance = original_create_glue_tool

    assert "AgentInTeam" in adapter.agents, "AgentInTeam was not created by the adapter."
    agno_agent_in_team = adapter.agents.get("AgentInTeam")
    assert agno_agent_in_team is not None

    assert hasattr(agno_agent_in_team, 'tools'), "Created AgnoAgent does not have a 'tools' attribute."
    
    agent_tools = agno_agent_in_team.tools
    assert isinstance(agent_tools, list), "AgnoAgent 'tools' attribute is not a list."
    
    assert len(agent_tools) == 2, f"Expected 2 tools, got {len(agent_tools)}"

    tool_names_on_agent = set()
    for tool_func in agent_tools:
        assert isinstance(tool_func, AgnoFunction), "Tool in agent's list is not an AgnoFunction."
        tool_names_on_agent.add(tool_func.name) 

    assert "agent_tool_1" in tool_names_on_agent, "Agent's specific tool not found."
    assert "team_tool_1" in tool_names_on_agent, "Team's tool not found on agent."

