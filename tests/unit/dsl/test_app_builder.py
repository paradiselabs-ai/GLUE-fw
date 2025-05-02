"""
Tests for the GLUE Application Builder.

These tests focus on the builder's ability to convert runtime configurations
into actual running GLUE application instances.
"""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Import the components we'll need
from glue.dsl.app_builder import GlueAppBuilder
from glue.core.types import AdhesiveType, FlowType
from glue.core.model import Model
from glue.tools.tool_base import Tool
from glue.core.teams import Team

# ==================== Test Fixtures ====================

@pytest.fixture
def basic_config():
    """Create a basic GLUE app configuration for testing."""
    return {
        "app": {
            "name": "Test App",
            "description": "A test application",
            "version": "0.1.0",
            "development": True
        },
        "models": [
            {
                "name": "test_model",
                "provider": "openrouter",
                "role": "assistant",
                "adhesives": ["glue"],
                "config": {
                    "model": "openai/gpt-4",
                    "temperature": 0.7
                }
            }
        ],
        "tools": [
            {
                "name": "web_search",
                "type": "search",
                "config": {
                    "metadata": {
                        "api_key": "${SEARCH_API_KEY}"
                    },
                    "adhesive_types": ["glue"]
                }
            }
        ],
        "teams": [
            {
                "name": "research",
                "model": "test_model",
                "tools": ["web_search"],
                "description": "Research team"
            }
        ],
        "flows": [
            {
                "source": "research",
                "target": "research",
                "type": "bidirectional"
            }
        ]
    }

@pytest.fixture
def complex_config():
    """Create a more complex GLUE app configuration for testing."""
    return {
        "app": {
            "name": "Advanced Test App",
            "description": "A more complex test application",
            "version": "0.2.0",
            "development": True
        },
        "models": [
            {
                "name": "research_model",
                "provider": "anthropic",
                "role": "researcher",
                "adhesives": ["glue", "velcro"],
                "config": {
                    "model": "claude-3-opus",
                    "temperature": 0.5
                }
            },
            {
                "name": "coding_model",
                "provider": "openrouter",
                "role": "coder",
                "adhesives": ["glue", "tape"],
                "config": {
                    "model": "openai/gpt-4",
                    "temperature": 0.2
                }
            }
        ],
        "tools": [
            {
                "name": "web_search",
                "type": "search",
                "config": {
                    "metadata": {
                        "api_key": "${SEARCH_API_KEY}"
                    },
                    "adhesive_types": ["glue"]
                }
            },
            {
                "name": "code_interpreter",
                "type": "code_interpreter",
                "config": {
                    "timeout": 10.0,
                    "metadata": {
                        "allowed_imports": ["os", "sys", "math"]
                    }
                }
            },
            {
                "name": "file_handler",
                "type": "file_handler",
                "config": {
                    "metadata": {
                        "base_path": "/tmp",
                        "allowed_extensions": [".txt", ".py", ".json"]
                    }
                }
            }
        ],
        "teams": [
            {
                "name": "research",
                "model": "research_model",
                "tools": ["web_search"],
                "description": "Research team"
            },
            {
                "name": "development",
                "model": "coding_model",
                "tools": ["code_interpreter", "file_handler"],
                "description": "Development team"
            }
        ],
        "flows": [
            {
                "source": "research",
                "target": "development",
                "type": "push"
            },
            {
                "source": "development",
                "target": "research",
                "type": "pull"
            }
        ]
    }

# ==================== Builder Tests ====================

class TestAppBuilder:
    """Tests for the GlueAppBuilder class."""
    
    def test_builder_initialization(self):
        """Test that the builder initializes correctly."""
        builder = GlueAppBuilder()
        assert builder is not None
        
    def test_build_basic_app(self, basic_config):
        """Test building a basic app from configuration."""
        builder = GlueAppBuilder()
        app = builder.build(basic_config)
        
        # Verify app properties
        assert app.name == "Test App"
        assert app.description == "A test application"
        assert app.version == "0.1.0"
        assert app.development is True
        
        # Verify models were created
        assert len(app.models) == 1
        assert isinstance(app.models["test_model"], Model)
        assert app.models["test_model"].name == "test_model"
        
        # Verify tools were created
        assert len(app.tools) == 1
        assert isinstance(app.tools["web_search"], Tool)
        assert app.tools["web_search"].name == "web_search"
        
        # Verify teams were created
        assert len(app.teams) == 1
        assert isinstance(app.teams["research"], Team)
        assert app.teams["research"].name == "research"
        
    def test_build_complex_app(self, complex_config):
        """Test building a more complex app from configuration."""
        builder = GlueAppBuilder()
        app = builder.build(complex_config)
        
        # Verify app properties
        assert app.name == "Advanced Test App"
        assert app.version == "0.2.0"
        
        # Verify models were created
        assert len(app.models) == 2
        assert "research_model" in app.models
        assert "coding_model" in app.models
        
        # Verify tools were created
        assert len(app.tools) == 3
        assert "web_search" in app.tools
        assert "code_interpreter" in app.tools
        assert "file_handler" in app.tools
        
        # Verify teams were created
        assert len(app.teams) == 2
        assert "research" in app.teams
        assert "development" in app.teams
        
        # Verify flows were created
        assert len(app.flows) == 2
        
    def test_model_instantiation(self, basic_config):
        """Test that models are properly instantiated with correct providers."""
        # Use a real model instead of a mock
        builder = GlueAppBuilder()
        app = builder.build(basic_config)
        
        # Verify model properties
        model = app.models["test_model"]
        assert model.name == "test_model"
        assert model.provider == "openrouter"
        assert model.config["model"] == "openai/gpt-4"
        assert model.config["temperature"] == 0.7
        assert AdhesiveType.GLUE in model.adhesives
            
    def test_tool_instantiation(self, basic_config):
        """Test that tools are properly instantiated."""
        # Use a real tool instead of a mock
        builder = GlueAppBuilder()
        app = builder.build(basic_config)
        
        # Verify tool properties
        tool = app.tools["web_search"]
        assert tool.name == "web_search"
        assert AdhesiveType.GLUE in tool.config.adhesive_types
            
    def test_team_instantiation(self, basic_config):
        """Test that teams are properly instantiated with models and tools."""
        # Use a real team instead of a mock
        builder = GlueAppBuilder()
        app = builder.build(basic_config)
        
        # Verify team properties
        team = app.teams["research"]
        assert team.name == "research"
        assert team.model.name == "test_model"
        assert len(team.tools) == 1
        assert team.tools[0].name == "web_search"
            
    def test_environment_variable_substitution(self, basic_config):
        """Test that environment variables are properly substituted."""
        with patch("os.environ", {"SEARCH_API_KEY": "test_api_key"}):
            builder = GlueAppBuilder()
            app = builder.build(basic_config)
            
            # Verify environment variable was substituted
            tool = app.tools["web_search"]
            assert tool.api_key == "test_api_key"
                
    def test_error_handling_missing_model(self, basic_config):
        """Test error handling when a referenced model is missing."""
        # Remove the model but keep the reference in the team
        basic_config["models"] = []
        
        builder = GlueAppBuilder()
        with pytest.raises(ValueError) as excinfo:
            app = builder.build(basic_config)
            
        assert "Referenced model not found" in str(excinfo.value)
        
    def test_error_handling_missing_tool(self, basic_config):
        """Test error handling when a referenced tool is missing."""
        # Remove the tool but keep the reference in the team
        basic_config["tools"] = []
        
        builder = GlueAppBuilder()
        with pytest.raises(ValueError) as excinfo:
            app = builder.build(basic_config)
            
        assert "Referenced tool not found" in str(excinfo.value)
        
    def test_error_handling_invalid_flow(self, basic_config):
        """Test error handling when a flow references non-existent teams."""
        # Add a flow with non-existent teams
        basic_config["flows"].append({
            "source": "non_existent",
            "target": "also_non_existent",
            "type": "push"
        })
        
        builder = GlueAppBuilder()
        with pytest.raises(ValueError) as excinfo:
            app = builder.build(basic_config)
            
        assert "Referenced team not found" in str(excinfo.value)


# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
