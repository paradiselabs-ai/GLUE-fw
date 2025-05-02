"""
Tests for the configuration validation utilities.

This module contains tests for the utilities that convert ConfigGenerator output
to Pydantic models and validate the configuration.
"""
import pytest
from typing import Dict, Any

from glue.core.schemas import AppConfig
from glue.core.config_validation import config_to_pydantic


class TestConfigToPydantic:
    """Tests for the config_to_pydantic function"""

    def test_valid_config_conversion(self):
        """Test that a valid configuration is properly converted to Pydantic models"""
        # Sample configuration from ConfigGenerator
        config = {
            "name": "Test App",
            "description": "A test application",
            "version": "0.1.0",
            "development": True,
            "log_level": "info",
            "models": [
                {
                    "provider": "openai",
                    "model_id": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            ],
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search the web for information"
                }
            ],
            "teams": [
                {
                    "name": "research_team",
                    "lead": "gpt4_model",
                    "members": ["assistant_model"],
                    "tools": ["web_search"]
                }
            ],
            "magnets": [
                {
                    "source": "research_team",
                    "target": "research_team",  # This should cause validation to fail
                    "flow_type": "bidirectional"
                }
            ]
        }
        
        # This should raise a ValidationError because source and target teams are the same
        with pytest.raises(ValueError):
            config_to_pydantic(config)
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        # Configuration missing required fields
        config = {
            "description": "A test application",
            "models": []
        }
        
        # This should raise a ValidationError because 'name' is missing
        with pytest.raises(ValueError):
            config_to_pydantic(config)
    
    def test_invalid_model_config(self):
        """Test that invalid model configurations raise validation errors"""
        # Configuration with invalid model configuration
        config = {
            "name": "Test App",
            "models": [
                {
                    "provider": "invalid_provider",  # Invalid provider
                    "model_id": "model-1"
                }
            ]
        }
        
        # This should raise a ValidationError because the provider is invalid
        with pytest.raises(ValueError):
            config_to_pydantic(config)
    
    def test_valid_minimal_config(self):
        """Test that a valid minimal configuration is properly converted"""
        # Minimal valid configuration
        config = {
            "name": "Minimal App"
        }
        
        # This should not raise any exceptions
        app_config = config_to_pydantic(config)
        
        # Verify that the conversion was successful
        assert isinstance(app_config, AppConfig)
        assert app_config.name == "Minimal App"
        assert app_config.description == ""
        assert app_config.version == "0.1.0"
        assert app_config.development is True
        assert app_config.log_level == "info"
        assert len(app_config.models) == 0
        assert len(app_config.tools) == 0
        assert len(app_config.teams) == 0
        assert len(app_config.magnets) == 0
