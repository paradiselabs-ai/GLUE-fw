import pytest
import logging
from unittest.mock import MagicMock, patch

from glue.core.adapters.agno.adapter import GlueAgnoAdapter
from glue.magnetic.field import MagneticField
from glue.magnetic.polarity import MagneticPolarity
from glue.core.types import FlowType

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test fixtures
@pytest.fixture
def basic_magnetic_config():
    """Basic configuration for testing magnetic flow integration."""
    return {
        "agents": {
            "MainAgent": {
                "model_name": "gpt-4",
                "provider": "openai",
                "config": {
                    "supported_adhesives": ["glue", "velcro", "tape"]
                }
            },
            "SecondaryAgent": {
                "model_name": "gpt-3.5-turbo",
                "provider": "openai",
                "config": {
                    "supported_adhesives": ["glue", "velcro", "tape"]
                }
            }
        },
        "teams": {
            "TeamA": {
                "agents": ["MainAgent"],
                "communication_pattern": "hierarchical",
                "tools": ["tool_a"]
            },
            "TeamB": {
                "agents": ["SecondaryAgent"],
                "communication_pattern": "hierarchical",
                "tools": ["tool_b"]
            }
        },
        "tools": {
            "tool_a": {
                "type": "basic",
                "description": "A basic tool for TeamA",
                "params": {
                    "input": {
                        "type": "string",
                        "description": "Input data"
                    }
                }
            },
            "tool_b": {
                "type": "basic",
                "description": "A basic tool for TeamB",
                "params": {
                    "input": {
                        "type": "string",
                        "description": "Input data"
                    }
                }
            }
        },
        "magnetic_flows": [
            {
                "source": "TeamA",
                "target": "TeamB",
                "polarity": "attract",
                "flow_type": "sequential"
            },
            {
                "source": "TeamB",
                "target": "TeamA",
                "polarity": "repel",
                "flow_type": "feedback"
            }
        ],
        "app_name": "MagneticFlowTestApp"
    }

# Stub classes for Agno components
class StubWorkflow:
    """Stub implementation of Agno Workflow class for testing."""
    def __init__(self, name):
        self.name = name
        self.teams = {}
        self.memory = MagicMock()
        self.team_connections = []
    
    def add_team(self, team):
        """Add a team to the workflow."""
        self.teams[team.name] = team
    
    def add_team_connection(self, source_team, target_team, connection_type):
        """Add a connection between teams."""
        self.team_connections.append({
            "source": source_team,
            "target": target_team,
            "type": connection_type
        })

class StubTeam:
    """Stub implementation of Agno Team class for testing."""
    def __init__(self, name, agents=None, communication_pattern=None):
        self.name = name
        self.agents = agents or []
        self.communication_pattern = communication_pattern
        self.tools = []
        self.connections = []
    
    def add_agent(self, agent):
        """Add an agent to the team."""
        self.agents.append(agent)
    
    def add_tool(self, tool):
        """Add a tool to the team."""
        self.tools.append(tool)
    
    def connect_to(self, target_team, connection_type):
        """Connect this team to another team."""
        self.connections.append({
            "target": target_team,
            "type": connection_type
        })

class StubAgent:
    """Stub implementation of Agno Agent class for testing."""
    def __init__(self, name, provider=None, model=None, config=None):
        self.name = name
        self.provider = provider
        self.model = model
        self.config = config or {}
        self.system_prompt = None
        self.tools = []
        
        # Set supported adhesives from config if provided
        if config and 'supported_adhesives' in config:
            self.supported_adhesives = config['supported_adhesives']
        else:
            # Default to all adhesive types
            self.supported_adhesives = ['glue', 'velcro', 'tape']

class StubTool:
    """Stub implementation of Agno Tool class for testing."""
    def __init__(self, name, description=None, params=None):
        self.name = name
        self.description = description
        self.params = params or {}

@pytest.mark.unit
def test_magnetic_flow_translation():
    """Test that GLUE magnetic flows are correctly translated to Agno team connections."""
    # Create a GlueAgnoAdapter instance
    adapter = GlueAgnoAdapter()
    
    # Create a magnetic field
    magnetic_field = MagneticField(name="TestField")
    
    # Add a magnetic flow from TeamA to TeamB with ATTRACT polarity
    magnetic_field.set_flow_sync(
        source_team="TeamA",
        target_team="TeamB",
        flow_type=FlowType.PUSH
    )
    
    # Add a magnetic flow from TeamB to TeamA with REPEL polarity
    magnetic_field.set_flow_sync(
        source_team="TeamB",
        target_team="TeamA",
        flow_type=FlowType.PULL
    )
    
    # Translate the magnetic field to Agno team connections
    agno_connections = adapter.translate_magnetic_flows(magnetic_field)
    
    # Verify that the correct number of connections was created
    assert len(agno_connections) == 2
    
    # Verify that the connections have the correct properties
    assert agno_connections[0]["source"] == "TeamA"
    assert agno_connections[0]["target"] == "TeamB"
    assert agno_connections[0]["type"] == "sequential"
    
    assert agno_connections[1]["source"] == "TeamB"
    assert agno_connections[1]["target"] == "TeamA"
    assert agno_connections[1]["type"] == "feedback"

@pytest.mark.unit
def test_magnetic_flow_setup(basic_magnetic_config):
    """Test that magnetic flows are properly set up in the Agno workflow."""
    # Create a GlueAgnoAdapter instance with mock Agno components
    adapter = GlueAgnoAdapter()
    
    # Set up the adapter with the basic magnetic config
    with patch.object(adapter, 'create_workflow', return_value=StubWorkflow("MagneticFlowTestApp")):
        with patch.object(adapter, 'create_agent', side_effect=lambda name, **kwargs: StubAgent(name, **kwargs)):
            with patch.object(adapter, 'create_team', side_effect=lambda name, **kwargs: StubTeam(name, **kwargs)):
                with patch.object(adapter, 'create_tool', side_effect=lambda name, **kwargs: StubTool(name, **kwargs)):
                    success = adapter.setup(basic_magnetic_config)
                    assert success is True
    
    # Verify that the workflow has the correct number of team connections
    assert len(adapter.workflow.team_connections) == 2
    
    # Verify that the connections have the correct properties
    assert adapter.workflow.team_connections[0]["source"].name == "TeamA"
    assert adapter.workflow.team_connections[0]["target"].name == "TeamB"
    assert adapter.workflow.team_connections[0]["type"] == "sequential"
    
    assert adapter.workflow.team_connections[1]["source"].name == "TeamB"
    assert adapter.workflow.team_connections[1]["target"].name == "TeamA"
    assert adapter.workflow.team_connections[1]["type"] == "feedback"

@pytest.mark.unit
def test_magnetic_polarity_mapping():
    """Test that GLUE magnetic polarities are correctly mapped to Agno connection types."""
    # Create a GlueAgnoAdapter instance
    adapter = GlueAgnoAdapter()
    
    # Test ATTRACT polarity with different flow types
    assert adapter.map_magnetic_polarity(MagneticPolarity.ATTRACT, FlowType.PUSH) == "sequential"
    assert adapter.map_magnetic_polarity(MagneticPolarity.ATTRACT, FlowType.BIDIRECTIONAL) == "parallel"
    assert adapter.map_magnetic_polarity(MagneticPolarity.ATTRACT, FlowType.PULL) == "feedback"
    
    # Test REPEL polarity with different flow types
    assert adapter.map_magnetic_polarity(MagneticPolarity.REPEL, FlowType.PUSH) == "conditional"
    assert adapter.map_magnetic_polarity(MagneticPolarity.REPEL, FlowType.BIDIRECTIONAL) == "conditional_parallel"
    assert adapter.map_magnetic_polarity(MagneticPolarity.REPEL, FlowType.PULL) == "conditional_feedback"

@pytest.mark.unit
def test_team_connection_creation():
    """Test that team connections are properly created in the Agno workflow."""
    # Create a GlueAgnoAdapter instance
    adapter = GlueAgnoAdapter()
    
    # Create mock teams
    team_a = StubTeam("TeamA")
    team_b = StubTeam("TeamB")
    
    # Create a mock workflow
    workflow = StubWorkflow("TestWorkflow")
    
    # Set the workflow and teams in the adapter
    adapter.workflow = workflow
    adapter.teams = {
        "TeamA": team_a,
        "TeamB": team_b
    }
    
    # Create a connection between TeamA and TeamB
    adapter.create_team_connection("TeamA", "TeamB", "sequential")
    
    # Verify that the connection was created in the workflow
    assert len(workflow.team_connections) == 1
    assert workflow.team_connections[0]["source"] == team_a
    assert workflow.team_connections[0]["target"] == team_b
    assert workflow.team_connections[0]["type"] == "sequential"
    
    # Verify that the connection was created in the team
    assert len(team_a.connections) == 1
    assert team_a.connections[0]["target"] == team_b
    assert team_a.connections[0]["type"] == "sequential"

@pytest.mark.unit
def test_magnetic_field_integration():
    """Test that the GLUE MagneticField can be integrated with Agno's team connections."""
    # Create a GlueAgnoAdapter instance
    adapter = GlueAgnoAdapter()
    
    # Create mock teams
    team_a = StubTeam("TeamA")
    team_b = StubTeam("TeamB")
    
    # Create a mock workflow
    workflow = StubWorkflow("TestWorkflow")
    
    # Set the workflow and teams in the adapter
    adapter.workflow = workflow
    adapter.teams = {
        "TeamA": team_a,
        "TeamB": team_b
    }
    
    # Create a magnetic field
    magnetic_field = MagneticField(name="TestField")
    
    # Add flows to the magnetic field
    magnetic_field.set_flow_sync(
        source_team="TeamA",
        target_team="TeamB",
        flow_type=FlowType.PUSH
    )
    
    magnetic_field.set_flow_sync(
        source_team="TeamB",
        target_team="TeamA",
        flow_type=FlowType.PULL
    )
    
    # Integrate the magnetic field with the adapter
    adapter.integrate_magnetic_field(magnetic_field)
    
    # Verify that the connections were created in the workflow
    assert len(workflow.team_connections) == 2
    
    # Verify that the connections have the correct properties
    assert workflow.team_connections[0]["source"] == team_a
    assert workflow.team_connections[0]["target"] == team_b
    assert workflow.team_connections[0]["type"] == "sequential"
    
    assert workflow.team_connections[1]["source"] == team_b
    assert workflow.team_connections[1]["target"] == team_a
    assert workflow.team_connections[1]["type"] == "conditional_feedback"
