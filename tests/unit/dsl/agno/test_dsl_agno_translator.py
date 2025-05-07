"""
Unit tests for translating GLUE DSL AST into Agno configuration.

These tests verify that the GLUE DSL parser can correctly translate
GLUE DSL syntax into Agno Workflow/Team/Agent configurations.
"""

import pytest
from typing import Dict, Any

from glue.core.adapters.agno.dsl_translator import GlueDSLAgnoTranslator


@pytest.fixture
def minimal_ast() -> Dict[str, Any]:
    """Fixture providing a minimal AST for testing."""
    return {
        "app": {
            "app_name": "MinimalTestApp",
            "engine": "agno"
        },
        "models": {
            "OpenAI": {
                "provider": "openrouter",
                "model_name": "openai/gpt-3.5-turbo"
            }
        },
        "teams": [
            {
                "name": "CoreTeam",
                "lead": "OpenAI",
                "models": ["OpenAI"]
            }
        ],
        "tools": {},
        "flows": []
    }


@pytest.fixture
def ast_with_tools() -> Dict[str, Any]:
    """Fixture providing an AST with tools for testing."""
    return {
        "app": {
            "app_name": "ToolTestApp",
            "engine": "agno"
        },
        "models": {
            "OpenAI": {
                "provider": "openrouter",
                "model_name": "openai/gpt-3.5-turbo"
            }
        },
        "tools": {
            "WebSearch": {
                "description": "Search the web for information",
                "params": {
                    "query": "string"
                },
                "adhesive": "GLUE"
            }
        },
        "teams": [
            {
                "name": "CoreTeam",
                "lead": "OpenAI",
                "models": ["OpenAI"],
                "tools": ["WebSearch"]
            }
        ],
        "flows": []
    }


@pytest.fixture
def ast_with_flows() -> Dict[str, Any]:
    """Fixture providing an AST with flows for testing."""
    return {
        "app": {
            "app_name": "FlowTestApp",
            "engine": "agno"
        },
        "models": {
            "OpenAI": {
                "provider": "openrouter",
                "model_name": "openai/gpt-3.5-turbo"
            },
            "Claude": {
                "provider": "anthropic",
                "model_name": "claude-3-opus-20240229"
            }
        },
        "teams": [
            {
                "name": "ResearchTeam",
                "lead": "OpenAI",
                "models": ["OpenAI"]
            },
            {
                "name": "WritingTeam",
                "lead": "Claude",
                "models": ["Claude"]
            }
        ],
        "flows": [
            {
                "name": "ResearchToWriting",
                "from": "ResearchTeam",
                "to": "WritingTeam",
                "type": "PUSH"
            }
        ]
    }


@pytest.mark.unit
def test_translator_initialization():
    """Test that the translator can be initialized."""
    translator = GlueDSLAgnoTranslator()
    assert translator is not None


@pytest.mark.unit
def test_translate_minimal_app(minimal_ast: Dict[str, Any]):
    """Test translating a minimal app configuration."""
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(minimal_ast)
    
    # Verify basic app properties
    assert agno_config is not None
    assert "workflow" in agno_config
    assert agno_config["workflow"]["name"] == "MinimalTestApp"


@pytest.mark.unit
def test_translate_models(minimal_ast: Dict[str, Any]):
    """Test translating models to Agno agents."""
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(minimal_ast)
    
    # Verify models are translated to agents
    assert "agents" in agno_config
    assert "OpenAI" in agno_config["agents"]
    assert agno_config["agents"]["OpenAI"]["provider"] == "openrouter"
    assert agno_config["agents"]["OpenAI"]["model_name"] == "openai/gpt-3.5-turbo"


@pytest.mark.unit
def test_translate_teams(minimal_ast: Dict[str, Any]):
    """Test translating teams to Agno teams."""
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(minimal_ast)
    
    # Verify teams are translated correctly
    assert "teams" in agno_config
    assert "CoreTeam" in agno_config["teams"]
    assert "lead" in agno_config["teams"]["CoreTeam"]
    assert agno_config["teams"]["CoreTeam"]["lead"] == "OpenAI"
    assert "members" in agno_config["teams"]["CoreTeam"]
    assert "OpenAI" in agno_config["teams"]["CoreTeam"]["members"]


@pytest.mark.unit
def test_translate_tools(ast_with_tools: Dict[str, Any]):
    """Test translating tools to Agno tools."""
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(ast_with_tools)
    
    # Verify tools are translated correctly
    assert "tools" in agno_config
    assert "WebSearch" in agno_config["tools"]
    assert agno_config["tools"]["WebSearch"]["description"] == "Search the web for information"
    assert "params" in agno_config["tools"]["WebSearch"]
    assert "query" in agno_config["tools"]["WebSearch"]["params"]
    
    # Verify tools are assigned to teams
    assert "teams" in agno_config
    assert "CoreTeam" in agno_config["teams"]
    assert "tools" in agno_config["teams"]["CoreTeam"]
    assert "WebSearch" in agno_config["teams"]["CoreTeam"]["tools"]


@pytest.mark.unit
def test_translate_flows(ast_with_flows: Dict[str, Any]):
    """Test translating magnetic flows to Agno team connections."""
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(ast_with_flows)
    
    # Verify flows are translated correctly
    assert "flows" in agno_config
    assert "ResearchToWriting" in agno_config["flows"]
    assert agno_config["flows"]["ResearchToWriting"]["from"] == "ResearchTeam"
    assert agno_config["flows"]["ResearchToWriting"]["to"] == "WritingTeam"
    assert agno_config["flows"]["ResearchToWriting"]["type"] == "PUSH"
