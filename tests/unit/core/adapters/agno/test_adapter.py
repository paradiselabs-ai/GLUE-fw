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
sys.modules['agno'] = StubAgnoModule()
sys.modules['agno.agent'] = sys.modules['agno'].agent
sys.modules['agno.team'] = sys.modules['agno'].team
sys.modules['agno.workflow'] = sys.modules['agno'].workflow

from glue.core.adapters.agno.adapter import GlueAgnoAdapter


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
def test_adapter_run(minimal_config):
    """Test that the adapter can run an Agno workflow."""
    adapter = GlueAgnoAdapter()
    result = adapter.run(minimal_config, "Test input")
    
    # Verify result - should be a placeholder since StubWorkflow.run raises NotImplementedError
    assert result is not None
    assert "status" in result
    assert result.get("status") == "success"
    assert result.get("message") == "Agno integration placeholder"


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
def test_adapter_handles_run_errors():
    """Test that the adapter properly handles run errors."""
    # Test with an empty config that will cause setup to fail
    empty_config = {}
    
    adapter = GlueAgnoAdapter()
    result = adapter.run(empty_config, "Test input")
    assert result is None
    
    # Create a special test case where setup succeeds but run fails
    # We'll do this by creating a custom exception-raising workflow
    class ErrorWorkflow:
        def __init__(self, *args, **kwargs):
            self.name = "ErrorWorkflow"
        
        def run(self, *args, **kwargs):
            raise Exception("Test error")
    
    # Replace the stub workflow with our error workflow
    original_workflow = sys.modules['agno.workflow'].Workflow
    sys.modules['agno.workflow'].Workflow = ErrorWorkflow
    
    try:
        # Run with a valid config but the workflow will raise an exception
        valid_config = {
            "workflow": {"name": "ErrorTest"},
            "agents": {"TestAgent": {"provider": "test", "model_name": "test"}},
            "teams": {"TestTeam": {"lead": "TestAgent", "members": ["TestAgent"]}}
        }
        
        result = adapter.run(valid_config, "Test input")
        assert result is None
    finally:
        # Restore the original stub
        sys.modules['agno.workflow'].Workflow = original_workflow
