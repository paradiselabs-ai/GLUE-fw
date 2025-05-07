"""
Unit tests for the team integration between GLUE and Agno.

These tests verify that GLUE team structures are properly translated to Agno team structures,
preserving team hierarchies, roles, and communication patterns.
"""

import pytest
import sys
from typing import Dict, Any

# Create stub classes for Agno components to avoid import errors
class StubAgent:
    """Stub implementation of Agno Agent class for testing."""
    def __init__(self, name, provider=None, model=None, config=None):
        self.name = name
        self.provider = provider
        self.model = model
        self.config = config
        self.system_prompt = None
        self.tools = []

class StubTeam:
    """Stub implementation of Agno Team class for testing."""
    def __init__(self, name, members=None, lead=None, config=None):
        self.name = name
        self.members = members or []
        self.lead = lead
        self.config = config
        self.tools = []
        self.communication_pattern = "hierarchical"  # Default pattern

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

# Add the stub module to sys.modules if not already present
if 'agno' not in sys.modules:
    sys.modules['agno'] = StubAgnoModule()
    sys.modules['agno.agent'] = sys.modules['agno'].agent
    sys.modules['agno.team'] = sys.modules['agno'].team
    sys.modules['agno.workflow'] = sys.modules['agno'].workflow

from glue.core.adapters.agno.adapter import GlueAgnoAdapter
from glue.core.adapters.agno.dsl_translator import GlueDSLAgnoTranslator


@pytest.fixture
def basic_team_config():
    """Fixture providing a basic team configuration for testing."""
    return {
        "workflow": {
            "name": "TeamTestApp",
            "description": "Test application for team integration"
        },
        "agents": {
            "LeadAgent": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "ResearchAgent": {
                "provider": "anthropic",
                "model_name": "claude-3-opus"
            },
            "WritingAgent": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            }
        },
        "teams": {
            "CoreTeam": {
                "lead": "LeadAgent",
                "members": ["LeadAgent", "ResearchAgent", "WritingAgent"],
                "communication_pattern": "hierarchical"
            }
        },
        "tools": {},
        "flows": {}
    }


@pytest.fixture
def multi_team_config():
    """Fixture providing a configuration with multiple teams for testing."""
    return {
        "workflow": {
            "name": "MultiTeamTestApp",
            "description": "Test application with multiple teams"
        },
        "agents": {
            "ProjectLead": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "ResearchLead": {
                "provider": "anthropic",
                "model_name": "claude-3-opus"
            },
            "WritingLead": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "Researcher1": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet"
            },
            "Researcher2": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            },
            "Writer1": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            },
            "Writer2": {
                "provider": "anthropic",
                "model_name": "claude-3-haiku"
            }
        },
        "teams": {
            "ProjectTeam": {
                "lead": "ProjectLead",
                "members": ["ProjectLead"],
                "communication_pattern": "hierarchical"
            },
            "ResearchTeam": {
                "lead": "ResearchLead",
                "members": ["ResearchLead", "Researcher1", "Researcher2"],
                "communication_pattern": "collaborative"
            },
            "WritingTeam": {
                "lead": "WritingLead",
                "members": ["WritingLead", "Writer1", "Writer2"],
                "communication_pattern": "hierarchical"
            }
        },
        "tools": {},
        "flows": {}
    }


@pytest.mark.unit
def test_team_creation(basic_team_config):
    """Test that teams are properly created in the Agno adapter."""
    adapter = GlueAgnoAdapter()
    success = adapter.setup(basic_team_config)
    
    assert success is True
    assert len(adapter.teams) == 1
    assert "CoreTeam" in adapter.teams
    
    # Verify team structure
    team = adapter.teams["CoreTeam"]
    assert team.name == "CoreTeam"
    assert team.lead is not None
    assert team.lead.name == "LeadAgent"
    assert len(team.members) == 3


@pytest.mark.unit
def test_team_roles(basic_team_config):
    """Test that team roles (lead, members) are properly assigned."""
    adapter = GlueAgnoAdapter()
    adapter.setup(basic_team_config)
    
    team = adapter.teams["CoreTeam"]
    
    # Verify lead is correctly assigned
    assert team.lead is not None
    assert team.lead.name == "LeadAgent"
    
    # Verify all members are present
    member_names = [member.name for member in team.members]
    assert "LeadAgent" in member_names
    assert "ResearchAgent" in member_names
    assert "WritingAgent" in member_names


@pytest.mark.unit
def test_multi_team_structure(multi_team_config):
    """Test that multiple teams are properly created with correct structures."""
    adapter = GlueAgnoAdapter()
    success = adapter.setup(multi_team_config)
    
    assert success is True
    assert len(adapter.teams) == 3
    assert "ProjectTeam" in adapter.teams
    assert "ResearchTeam" in adapter.teams
    assert "WritingTeam" in adapter.teams
    
    # Verify Project Team
    project_team = adapter.teams["ProjectTeam"]
    assert project_team.lead.name == "ProjectLead"
    assert len(project_team.members) == 1
    
    # Verify Research Team
    research_team = adapter.teams["ResearchTeam"]
    assert research_team.lead.name == "ResearchLead"
    assert len(research_team.members) == 3
    
    # Verify Writing Team
    writing_team = adapter.teams["WritingTeam"]
    assert writing_team.lead.name == "WritingLead"
    assert len(writing_team.members) == 3


@pytest.mark.unit
def test_team_communication_patterns(multi_team_config):
    """Test that team communication patterns are properly configured."""
    adapter = GlueAgnoAdapter()
    adapter.setup(multi_team_config)
    
    # Verify communication patterns
    assert adapter.teams["ProjectTeam"].communication_pattern == "hierarchical"
    assert adapter.teams["ResearchTeam"].communication_pattern == "collaborative"
    assert adapter.teams["WritingTeam"].communication_pattern == "hierarchical"


@pytest.mark.unit
def test_translator_team_mapping():
    """Test that the DSL translator correctly maps GLUE team structures to Agno team structures."""
    # Create a simple GLUE AST with team definitions
    glue_ast = {
        "app": {
            "app_name": "TeamMappingTest",
            "engine": "agno"
        },
        "models": {
            "LeadModel": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "AssistantModel": {
                "provider": "anthropic",
                "model_name": "claude-3-opus"
            }
        },
        "teams": [
            {
                "name": "MainTeam",
                "lead": "LeadModel",
                "models": ["LeadModel", "AssistantModel"],
                "communication_pattern": "hierarchical"
            }
        ],
        "tools": {},
        "flows": []
    }
    
    # Translate the AST to Agno configuration
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(glue_ast)
    
    # Verify team structure in the translated configuration
    assert "teams" in agno_config
    assert "MainTeam" in agno_config["teams"]
    
    team_config = agno_config["teams"]["MainTeam"]
    assert team_config["lead"] == "LeadModel"
    assert "LeadModel" in team_config["members"]
    assert "AssistantModel" in team_config["members"]
    assert team_config.get("communication_pattern") == "hierarchical"
