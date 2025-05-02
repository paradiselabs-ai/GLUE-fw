import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from enum import Enum

# Mock classes to isolate the tests from the actual implementation
class MockTokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    EQUALS = "EQUALS"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    ARROW = "ARROW"
    SEMICOLON = "SEMICOLON"
    COMMENT = "COMMENT"
    EOF = "EOF"

@dataclass
class MockToken:
    type: MockTokenType
    value: str
    line: int

class MockParser:
    """Mock parser for testing the config generator in isolation"""
    
    def __init__(self, ast: Dict[str, Any]):
        self.ast = ast
        
    def parse(self) -> Dict[str, Any]:
        return self.ast

# Import the actual ConfigGenerator class (to be implemented)
from glue.dsl.config_generator import ConfigGenerator

class TestConfigGeneratorInitialization:
    """Test the initialization of the ConfigGenerator"""
    
    def test_config_generator_initialization(self):
        """Test that the ConfigGenerator can be initialized with a parser"""
        mock_parser = MockParser({})
        config_generator = ConfigGenerator(mock_parser)
        
        assert config_generator.parser == mock_parser
        assert isinstance(config_generator.config, dict)


class TestBasicConfigGeneration:
    """Test basic configuration generation"""
    
    def test_generate_empty_config(self):
        """Test generating configuration from an empty AST"""
        mock_parser = MockParser({
            "app": {},
            "teams": [],
            "models": [],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert isinstance(config, dict)
        assert "app" in config
        assert "teams" in config
        assert "models" in config
        assert "tools" in config
        assert "flows" in config
        
    def test_generate_app_config(self):
        """Test generating app configuration"""
        mock_parser = MockParser({
            "app": {
                "name": "Test App",
                "description": "A test application",
                "version": "1.0.0",
                "config": {
                    "development": True,
                    "log_level": "debug"
                }
            },
            "teams": [],
            "models": [],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert config["app"]["name"] == "Test App"
        assert config["app"]["description"] == "A test application"
        assert config["app"]["version"] == "1.0.0"
        assert config["app"]["config"]["development"] is True
        assert config["app"]["config"]["log_level"] == "debug"


class TestModelConfigGeneration:
    """Test model configuration generation"""
    
    def test_generate_model_config(self):
        """Test generating model configuration"""
        mock_parser = MockParser({
            "app": {},
            "teams": [],
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            ],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "gpt4"
        assert config["models"][0]["provider"] == "openai"
        assert config["models"][0]["model"] == "gpt-4"
        assert config["models"][0]["temperature"] == 0.7
        assert config["models"][0]["max_tokens"] == 2048
        
    def test_generate_multiple_models(self):
        """Test generating multiple model configurations"""
        mock_parser = MockParser({
            "app": {},
            "teams": [],
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                {
                    "name": "claude",
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "temperature": 0.5,
                    "max_tokens": 4096
                }
            ],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert len(config["models"]) == 2
        assert config["models"][0]["name"] == "gpt4"
        assert config["models"][1]["name"] == "claude"


class TestToolConfigGeneration:
    """Test tool configuration generation"""
    
    def test_generate_tool_config(self):
        """Test generating tool configuration"""
        mock_parser = MockParser({
            "app": {},
            "teams": [],
            "models": [],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "provider": "google",
                    "config": {
                        "api_key": "API_KEY",
                        "engine": "google"
                    }
                }
            ],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert len(config["tools"]) == 1
        assert config["tools"][0]["name"] == "search"
        assert config["tools"][0]["description"] == "Search the web"
        assert config["tools"][0]["provider"] == "google"
        assert config["tools"][0]["config"]["api_key"] == "API_KEY"
        assert config["tools"][0]["config"]["engine"] == "google"


class TestTeamConfigGeneration:
    """Test team configuration generation"""
    
    def test_generate_team_config(self):
        """Test generating team configuration"""
        mock_parser = MockParser({
            "app": {},
            "teams": [
                {
                    "name": "research",
                    "lead": "researcher",
                    "members": ["analyst", "writer"],
                    "tools": ["search", "summarize"]
                }
            ],
            "models": [],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert len(config["teams"]) == 1
        assert config["teams"][0]["name"] == "research"
        assert config["teams"][0]["lead"] == "researcher"
        assert "analyst" in config["teams"][0]["members"]
        assert "writer" in config["teams"][0]["members"]
        assert "search" in config["teams"][0]["tools"]
        assert "summarize" in config["teams"][0]["tools"]


class TestDefaultValueApplication:
    """Test default value application"""
    
    def test_default_model_values(self):
        """Test applying default values to model configuration"""
        mock_parser = MockParser({
            "app": {},
            "teams": [],
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4"
                    # Missing temperature and max_tokens
                }
            ],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert config["models"][0]["temperature"] == 0.7  # Default value
        assert config["models"][0]["max_tokens"] == 2048  # Default value
        
    def test_default_app_values(self):
        """Test applying default values to app configuration"""
        mock_parser = MockParser({
            "app": {
                "name": "Test App"
                # Missing description, version, and config
            },
            "teams": [],
            "models": [],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        assert config["app"]["description"] == ""  # Default value
        assert config["app"]["version"] == "0.1.0"  # Default value
        assert "config" in config["app"]
        assert config["app"]["config"]["development"] is True  # Default value


class TestSemanticValidation:
    """Test semantic validation of configurations"""
    
    def test_validate_required_fields(self):
        """Test validation of required fields"""
        mock_parser = MockParser({
            "app": {
                # Missing name
            },
            "teams": [],
            "models": [],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        errors = config_generator.validate()
        
        # Check that validation detects missing required fields
        assert len(errors) > 0
    
    def test_validate_model_references(self):
        """Test validation of model references"""
        # Create a configuration with a team referencing a non-existent model
        mock_parser = MockParser({
            "app": {
                "name": "Test App"
            },
            "teams": [
                {
                    "name": "research",
                    "lead": "researcher",
                    "members": ["analyst", "writer"],
                    "tools": ["search"],
                    "model": "gpt4"  # Reference to a non-existent model
                }
            ],
            "models": [
                {
                    "name": "claude",  # Different model name
                    "provider": "anthropic",
                    "model": "claude-3-opus"
                }
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web"
                }
            ],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        errors = config_generator.validate()
        
        # Check that validation detects the invalid model reference
        assert len(errors) > 0
    
    def test_validate_tool_references(self):
        """Test validation of tool references"""
        # Create a configuration with a team referencing a non-existent tool
        mock_parser = MockParser({
            "app": {
                "name": "Test App"
            },
            "teams": [
                {
                    "name": "research",
                    "lead": "researcher",
                    "members": ["analyst", "writer"],
                    "tools": ["nonexistent_tool"]  # Reference to a non-existent tool
                }
            ],
            "models": [],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web"
                }
            ],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        errors = config_generator.validate()
        
        # Check that validation detects the invalid tool reference
        assert len(errors) > 0

class TestErrorHandling:
    """Test error handling in configuration generation"""
    
    def test_error_messages_for_missing_fields(self):
        """Test error messages for missing required fields"""
        mock_parser = MockParser({
            "app": {
                # Missing name
            },
            "teams": [],
            "models": [
                {
                    "name": "gpt4"
                    # Missing provider and model
                }
            ],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        errors = config_generator.validate()
        
        # Print errors for debugging
        print("\nErrors generated:")
        for error in errors:
            print(f"  - {error}")
        
        # Convert errors to strings for easier checking
        error_strings = [str(error) for error in errors]
        
        # Check for app name error
        app_name_missing = any("name" in error.lower() and "app" in error.lower() for error in error_strings)
        
        # Check for model provider error
        model_provider_missing = any("provider" in error.lower() for error in error_strings)
        
        # Check for model model error
        model_model_missing = any("model" in error.lower() and not "provider" in error.lower() for error in error_strings)
        
        # Assert that all expected errors are present
        assert app_name_missing, "Missing app name error not found"
        assert model_provider_missing, "Missing model provider error not found"
        assert model_model_missing, "Missing model model error not found"
    
    def test_error_suggestions(self):
        """Test that error messages include suggestions for fixes"""
        mock_parser = MockParser({
            "app": {
                "name": "Test App"
            },
            "teams": [],
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-5"  # Non-existent model
                }
            ],
            "tools": [],
            "flows": []
        })
        
        config_generator = ConfigGenerator(mock_parser)
        errors = config_generator.validate()
        
        # Check that validation detects the invalid model
        assert len(errors) > 0


class TestCompleteConfigGeneration:
    """Test complete configuration generation"""
    
    def test_generate_complete_config(self):
        """Test generating a complete configuration"""
        mock_parser = MockParser({
            "app": {
                "name": "Research Assistant",
                "description": "An AI research assistant",
                "version": "1.0.0",
                "config": {
                    "development": True,
                    "log_level": "info"
                }
            },
            "teams": [
                {
                    "name": "research",
                    "lead": "researcher",
                    "members": ["analyst", "writer"],
                    "tools": ["search", "summarize"]
                }
            ],
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "provider": "google",
                    "config": {
                        "api_key": "API_KEY",
                        "engine": "google"
                    }
                },
                {
                    "name": "summarize",
                    "description": "Summarize text",
                    "provider": "internal",
                    "config": {}
                }
            ],
            "flows": [
                {
                    "source": "research",
                    "target": "output",
                    "type": "PUSH"
                }
            ]
        })
        
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        # Verify app config
        assert config["app"]["name"] == "Research Assistant"
        assert config["app"]["description"] == "An AI research assistant"
        
        # Verify teams
        assert len(config["teams"]) == 1
        assert config["teams"][0]["name"] == "research"
        assert len(config["teams"][0]["members"]) == 2
        
        # Verify models
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "gpt4"
        
        # Verify tools
        assert len(config["tools"]) == 2
        assert config["tools"][0]["name"] == "search"
        assert config["tools"][1]["name"] == "summarize"
        
        # Verify flows
        assert len(config["flows"]) == 1
        assert config["flows"][0]["source"] == "research"
        assert config["flows"][0]["target"] == "output"


class TestParserIntegration:
    """Test integration with the parser"""
    
    def test_parser_config_generator_integration(self):
        """Test integration between parser and config generator"""
        # Create a mock parser with a simple AST
        mock_ast = {
            "app": {
                "name": "TestApp",
                "description": "Test application",
                "version": "1.0.0"
            },
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4"
                }
            ]
        }
        mock_parser = MockParser(mock_ast)
        
        # Generate configuration
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        
        # Verify that the configuration was generated correctly
        assert config["app"]["name"] == "TestApp"
        assert config["app"]["description"] == "Test application"
        assert config["app"]["version"] == "1.0.0"
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "gpt4"
        assert config["models"][0]["provider"] == "openai"
        assert config["models"][0]["model"] == "gpt-4"


class TestPydanticValidation:
    """Test Pydantic validation integration with ConfigGenerator"""
    
    def test_pydantic_validation_success(self):
        """Test successful validation using Pydantic models"""
        # Create a mock parser with a valid AST
        mock_ast = {
            "app": {
                "name": "TestApp",
                "description": "Test application",
                "version": "1.0.0"
            },
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search tool",
                    "provider": "google"
                }
            ],
            "teams": [
                {
                    "name": "research",
                    "lead": "researcher",
                    "members": ["assistant1", "assistant2"],
                    "tools": ["search"],
                    "model": "gpt4"
                }
            ]
        }
        mock_parser = MockParser(mock_ast)
        
        # Generate and validate configuration
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        errors = config_generator.validate()
        
        # Verify that there are no validation errors
        assert len(errors) == 0
    
    def test_pydantic_validation_type_error(self):
        """Test validation error due to incorrect type"""
        # Create a mock parser with an AST containing type errors
        mock_ast = {
            "app": {
                "name": "TestApp",
                "description": "Test application",
                "version": "1.0.0"
            },
            "models": [
                {
                    "name": "gpt4",
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": "not a float",  # Type error: should be float
                    "max_tokens": "not an integer"  # Type error: should be integer
                }
            ]
        }
        mock_parser = MockParser(mock_ast)
        
        # Generate and validate configuration
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        errors = config_generator.validate()
        
        # Verify that there are validation errors
        assert len(errors) > 0
        # Check that the errors contain information about the type errors
        error_string = "\n".join(errors)
        assert "temperature" in error_string
        assert "max_tokens" in error_string
    
    def test_pydantic_validation_missing_required_field(self):
        """Test validation error due to missing required field"""
        # Create a mock parser with an AST missing required fields
        mock_ast = {
            "app": {
                "description": "Test application",
                "version": "1.0.0"
                # Missing 'name' field
            },
            "models": [
                {
                    # Missing 'name' field
                    "provider": "openai",
                    "model": "gpt-4"
                }
            ]
        }
        mock_parser = MockParser(mock_ast)
        
        # Generate and validate configuration
        config_generator = ConfigGenerator(mock_parser)
        config = config_generator.generate()
        errors = config_generator.validate()
        
        # Verify that there are validation errors
        assert len(errors) > 0
        # Check that the errors contain information about the missing fields
        error_string = "\n".join(errors)
        assert "name" in error_string
