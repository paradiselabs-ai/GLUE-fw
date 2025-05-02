"""
Fixtures for CLI tests, particularly for the enhanced interactive mode.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

# ==================== Mock App Fixtures ====================
@pytest.fixture
def mock_app():
    """Returns a mock GLUE app for testing interactive mode."""
    app = MagicMock()
    app.app_config = MagicMock()
    app.app_config.name = "Test App"
    app.run = AsyncMock(return_value="Response from app")
    app.clear_memory = MagicMock()
    
    # Set up team structure
    app.teams = {
        "research_team": {
            "lead": "researcher",
            "members": ["assistant", "writer"]
        }
    }
    
    # Set up tools
    app.tools = {
        "search": MagicMock(),
        "calculator": MagicMock()
    }
    
    return app

@pytest.fixture
def mock_app_with_verbose():
    """Returns a mock GLUE app that supports verbose mode."""
    app = MagicMock()
    app.app_config = MagicMock()
    app.app_config.name = "Test App"
    
    # Mock the run method to return agent interactions when verbose is enabled
    app.run = AsyncMock(return_value={
        "final_response": "Final response from app",
        "agent_interactions": [
            {"agent": "researcher", "message": "Researching the query..."},
            {"agent": "assistant", "message": "Processing research results..."},
            {"agent": "writer", "message": "Formatting the final response..."}
        ]
    })
    
    return app

@pytest.fixture
def mock_app_with_step_execution():
    """Returns a mock GLUE app that supports step-by-step execution."""
    app = MagicMock()
    app.app_config = MagicMock()
    app.app_config.name = "Test App"
    
    # Mock the step-by-step execution methods
    app.begin_step_execution = AsyncMock()
    app.next_step = AsyncMock(side_effect=[
        {"agent": "researcher", "message": "Researching the query..."},
        {"agent": "assistant", "message": "Processing research results..."},
        {"agent": "writer", "message": "Final response: Hello!"}
    ])
    app.end_step_execution = AsyncMock()
    
    return app

# ==================== Example Input Fixtures ====================
@pytest.fixture
def example_user_input():
    """Returns example user input for testing."""
    return "What is the capital of France?"

@pytest.fixture
def example_command_inputs():
    """Returns a dictionary of example command inputs for testing."""
    return {
        "help": "/help",
        "status": "/status",
        "tools": "/tools",
        "teams": "/teams",
        "clear": "/clear",
        "verbose": "/verbose",
        "step": "/step",
        "next": "/next",
        "color_on": "/color on",
        "color_off": "/color off",
        "exit": "/exit"
    }
