"""
Unit tests for the GLUE app functionality.

These tests define the expected behavior of the GlueApp class, which
is responsible for orchestrating models, teams, and tools according to
the configuration from a .glue file.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import os
import sys
# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from glue.core.types import AdhesiveType, ModelConfig, TeamConfig, ToolConfig
from glue.magnetic.field import FlowType, MagneticField

# Import the class we will implement
try:
    from glue.core.app import GlueApp
except ImportError:
    # This will fail until we implement the class
    pass

# ==================== Basic App Tests ====================
class TestGlueAppBasics:
    """Test basic app functionality."""
    
    def test_app_initialization(self):
        """Test that the app initializes with a config file."""
        with patch('glue.dsl.parser.GlueParser.parse_file') as mock_parse:
            mock_parse.return_value = {
                "app": {
                    "name": "Test App",
                    "config": {
                        "development": True,
                        "sticky": False
                    }
                },
                "models": {},
                "tools": {},
                "magnetize": {}
            }
            
            app = GlueApp(config_file="test.glue")
            
            mock_parse.assert_called_once_with("test.glue")
            
            # Check app configuration
            assert app.app_config.name == "Test App"
            assert app.app_config.development is True
            assert app.app_config.sticky is False
    
    def test_app_initialization_with_dict(self):
        """Test that the app can initialize with a config dict."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [],
            "tools": [],
            "teams": [],
            "flows": []
        }
        
        app = GlueApp(config=config)
        
        # Check app configuration
        assert app.app_config.name == "Test App"
        assert app.app_config.development is True
    
    def test_app_initialization_error(self):
        """Test that app initialization fails with no config."""
        with pytest.raises(ValueError):
            GlueApp()
    
    @pytest.mark.asyncio
    async def test_app_setup(self):
        """Test that the app sets up models, teams, and tools."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [
                {
                    "name": "test_model",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model",
                        temperature=0.5,
                        max_tokens=1000
                    ),
                    "adhesives": ["glue", "velcro"],
                    "role": "Test model"
                }
            ],
            "tools": [
                {
                    "name": "test_tool",
                    "config": ToolConfig(
                        name="test_tool",
                        description="Test tool"
                    ),
                    "type": "search"
                }
            ],
            "teams": [
                {
                    "name": "test_team",
                    "config": TeamConfig(
                        name="test_team",
                        lead="test_model",
                        members=[],
                        tools=["test_tool"]
                    )
                }
            ],
            "flows": []
        }
        
        # Create and setup app with real implementations
        app = GlueApp(config=config)
        await app.setup()
        
        # Verify the app was set up correctly
        assert "test_model" in app.models
        assert "test_tool" in app.tools
        assert "test_team" in app.teams
        
        # Verify the team has the correct model and tool
        team = app.teams["test_team"]
        assert team.name == "test_team"
        assert team.lead == app.models["test_model"]
        assert app.tools["test_tool"] in team.tools
        
        # Verify the team is in the magnetic field
        assert team in app.field.teams
    
    @pytest.mark.asyncio
    async def test_app_run(self):
        """Test that the app can run with an input."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [
                {
                    "name": "test_model",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model",
                        temperature=0.5,
                        max_tokens=1000
                    ),
                    "adhesives": ["glue", "velcro"],
                    "role": "Test model"
                }
            ],
            "tools": [],
            "teams": [
                {
                    "name": "test_team",
                    "config": TeamConfig(
                        name="test_team",
                        lead="test_model",
                        members=[],
                        tools=[]
                    )
                }
            ],
            "flows": []
        }
        
        # Create and setup app with real implementations
        app = GlueApp(config=config)
        await app.setup()
        
        # Create a simple model for testing that returns a fixed response
        class TestModel:
            async def generate(self, prompt, **kwargs):
                return "Test response"
        
        # Replace the model with our test model
        test_model = TestModel()
        app.models["test_model"] = test_model
        
        # Get the team and replace its model
        team = app.teams["test_team"]
        team.lead = test_model
        
        # Run app
        response = await app.run("Test input")
        
        # Check that response was returned
        assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_app_run_with_adhesive_workflow(self):
        """Test that the app uses tools with appropriate adhesives."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [
                {
                    "name": "test_model",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model",
                        temperature=0.5,
                        max_tokens=1000
                    ),
                    "adhesives": ["glue", "velcro", "tape"],
                    "role": "Test model"
                }
            ],
            "tools": [
                {
                    "name": "test_tool",
                    "config": ToolConfig(
                        name="test_tool",
                        description="Test tool"
                    )
                }
            ],
            "teams": [
                {
                    "name": "test_team",
                    "config": TeamConfig(
                        name="test_team",
                        lead="test_model",
                        members=[],
                        tools=["test_tool"]
                    )
                }
            ],
            "flows": []
        }
        
        with patch('glue.core.app.create_model') as mock_create_model, \
             patch('glue.core.app.create_tool') as mock_create_tool, \
             patch('glue.core.app.Team') as mock_team, \
             patch('glue.core.app.MagneticField') as mock_field:
            
            # Mock model with tool use capability
            mock_model = AsyncMock()
            mock_model.generate.return_value = "Test response"
            mock_model.use_tool = AsyncMock()
            mock_model.use_tool.return_value = {"result": "Tool result"}
            mock_create_model.return_value = mock_model
            
            # Mock tool
            mock_tool = MagicMock()
            mock_create_tool.return_value = mock_tool
            
            # Mock team with process_message
            mock_team_instance = MagicMock()
            mock_team.return_value = mock_team_instance
            mock_team_instance.add_member = AsyncMock()
            mock_team_instance.add_tool = AsyncMock()
            mock_team_instance.process_message = AsyncMock()
            mock_team_instance.process_message.return_value = "Test response"
            mock_team_instance.share_result = AsyncMock()
            
            # Mock magnetic field
            mock_field_instance = MagicMock()
            mock_field.return_value = mock_field_instance
            mock_field_instance.add_team = AsyncMock()
            
            # Create and setup app
            app = GlueApp(config=config)
            await app.setup()
            
            # For direct access by the app.run method, add the mock model to the app's models
            app.models = {"test_model": mock_model}
            
            # Run app with model that uses a tool
            input_text = "Use tool with glue adhesive"
            await app.run(input_text)
            
            # Check that model use_tool was called
            assert mock_model.use_tool.called
    
    @pytest.mark.asyncio
    async def test_app_run_with_team_interaction(self):
        """Test that the app handles team interactions."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [
                {
                    "name": "model1",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model1"
                    ),
                    "adhesives": ["glue"],
                    "role": "Model 1"
                },
                {
                    "name": "model2",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model2"
                    ),
                    "adhesives": ["glue"],
                    "role": "Model 2"
                }
            ],
            "tools": [],
            "teams": [
                {
                    "name": "team1",
                    "config": TeamConfig(
                        name="team1",
                        lead="model1",
                        members=[],
                        tools=[]
                    )
                },
                {
                    "name": "team2",
                    "config": TeamConfig(
                        name="team2",
                        lead="model2",
                        members=[],
                        tools=[]
                    )
                }
            ],
            "flows": [
                {
                    "source": "team1",
                    "target": "team2",
                    "type": FlowType.PUSH
                }
            ]
        }
        
        with patch('glue.core.app.create_model') as mock_create_model, \
             patch('glue.core.app.Team') as mock_team, \
             patch('glue.core.app.MagneticField') as mock_field:
            
            # Mock models
            mock_model1 = AsyncMock()
            mock_model1.generate.return_value = "Model 1 response"
            
            mock_model2 = AsyncMock()
            mock_model2.generate.return_value = "Model 2 response"
            
            # Create different models based on name
            def create_model_side_effect(config, **kwargs):
                if config.get("name") == "model1":
                    return mock_model1
                elif config.get("name") == "model2":
                    return mock_model2
            
            mock_create_model.side_effect = create_model_side_effect
            
            # Mock teams
            mock_team1 = MagicMock()
            mock_team1.add_member = AsyncMock()
            mock_team1.process_message = AsyncMock()
            mock_team1.process_message.return_value = "Team 1 response"
            
            mock_team2 = MagicMock()
            mock_team2.add_member = AsyncMock()
            mock_team2.process_message = AsyncMock()
            mock_team2.process_message.return_value = "Team 2 response"
            
            # Create different teams based on name
            def create_team_side_effect(name, config, lead=None, members=None):
                if name == "team1":
                    return mock_team1
                elif name == "team2":
                    return mock_team2
            
            mock_team.side_effect = create_team_side_effect
            
            # Mock magnetic field
            mock_field_instance = MagicMock()
            mock_field.return_value = mock_field_instance
            mock_field_instance.add_team = AsyncMock()
            mock_field_instance.set_flow = AsyncMock()
            mock_field_instance.transfer_information = AsyncMock()
            mock_field_instance.transfer_information.return_value = True
            
            # Create and setup app
            app = GlueApp(config=config)
            await app.setup()
            
            # Set mock teams
            app.teams = {
                "team1": mock_team1,
                "team2": mock_team2
            }
            
            # Set mock magnetic field
            app.magnetic_field = mock_field_instance
            
            # Run app
            input_text = "Test input for team interaction"
            await app.run(input_text)
            
            # Check that team1 process_message was called
            mock_team1.process_message.assert_called_once()
            
            # Check that magnetic field transfer_information was called
            mock_field_instance.transfer_information.assert_called_once()

# ==================== Communication Tests ====================
class TestGlueAppCommunication:
    """Test app communication functionality."""
    
    @pytest.mark.asyncio
    async def test_app_run(self):
        pass

# ==================== Cleanup and Resource Management Tests ====================
class TestGlueAppCleanup:
    """Test app cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_app_close(self):
        """Test that the app properly closes and cleans up resources."""
        config = {
            "app": {
                "name": "Test App",
                "config": {
                    "development": True
                }
            },
            "models": [
                {
                    "name": "test_model",
                    "config": ModelConfig(
                        provider="openrouter",
                        model_id="test/model"
                    ),
                    "adhesives": ["glue"],
                    "role": "Test model"
                }
            ],
            "tools": [
                {
                    "name": "test_tool",
                    "config": ToolConfig(
                        name="test_tool",
                        description="Test tool"
                    )
                }
            ],
            "teams": [
                {
                    "name": "test_team",
                    "config": TeamConfig(
                        name="test_team",
                        lead="test_model",
                        members=[],
                        tools=["test_tool"]
                    )
                }
            ],
            "flows": []
        }
        
        with patch('glue.core.app.create_model') as mock_create_model, \
             patch('glue.core.app.create_tool') as mock_create_tool, \
             patch('glue.core.app.Team') as mock_team, \
             patch('glue.core.app.MagneticField') as mock_field:
            
            # Create mocks with cleanup methods
            mock_model = AsyncMock()
            mock_model.cleanup = AsyncMock()
            mock_create_model.return_value = mock_model
            
            mock_tool = MagicMock()
            mock_tool.cleanup = AsyncMock()
            mock_create_tool.return_value = mock_tool
            
            mock_team_instance = MagicMock()
            mock_team_instance.add_member = AsyncMock()
            mock_team_instance.add_tool = AsyncMock()
            mock_team_instance.cleanup = AsyncMock()
            mock_team.return_value = mock_team_instance
            
            mock_field_instance = MagicMock()
            mock_field_instance.add_team = AsyncMock()
            mock_field_instance.cleanup = AsyncMock()
            mock_field.return_value = mock_field_instance
            
            # Create and setup app
            app = GlueApp(config=config)
            await app.setup()
            
            # Close app
            await app.close()
            
            # Check that cleanup methods were called
            mock_team_instance.cleanup.assert_called_once()
            mock_field_instance.cleanup.assert_called_once()
