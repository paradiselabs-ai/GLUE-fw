"""
Integration tests for prompt engineering in the GLUE framework.

These tests verify that the prompt engineering system correctly generates
appropriate system prompts for different models and providers.
"""

import os
import pytest
import asyncio
from typing import Dict, Any

from glue.core.base_model import BaseModel
from glue.core.model import Model
from glue.core.teams import Team
from glue.core.types import AdhesiveType


@pytest.mark.asyncio
async def test_system_prompt_generation():
    """Test that system prompts are correctly generated."""
    # Create a model with a role
    model_config = {
        'name': 'test_model',
        'role': 'researcher',
        'provider': 'gemini',
        'model': 'gemini-1.5-pro',
        'development': True  # Use development mode to avoid API calls
    }
    
    model = Model(model_config)
    
    # Add adhesives
    model.adhesives.add(AdhesiveType.GLUE)
    model.adhesives.add(AdhesiveType.VELCRO)
    
    # Generate a system prompt
    system_prompt = model._generate_system_prompt()
    
    # Verify the prompt contains key elements
    assert "GLUE Model: test_model" in system_prompt
    assert "researcher" in system_prompt
    assert "GLUE Framework" in system_prompt
    assert "Adhesive Tool Usage" in system_prompt
    assert "Team Communication" in system_prompt
    assert "GLUE" in system_prompt
    assert "VELCRO" in system_prompt
    assert "Response Guidelines" in system_prompt
    assert "Tool Usage Instructions" in system_prompt


@pytest.mark.asyncio
async def test_team_context_in_prompt():
    """Test that team context is correctly included in the system prompt."""
    # Create models
    researcher = Model({
        'name': 'researcher',
        'role': 'Find information',
        'provider': 'gemini',
        'development': True
    })
    
    writer = Model({
        'name': 'writer',
        'role': 'Write documentation',
        'provider': 'gemini',
        'development': True
    })
    
    # Create a team
    team = Team("research_team")
    
    # Add models to the team
    await team.add_member(researcher)
    await team.add_member(writer)
    
    # Generate system prompts
    researcher_prompt = researcher._generate_system_prompt()
    writer_prompt = writer._generate_system_prompt()
    
    # Verify team context is included
    assert "research_team" in researcher_prompt
    assert "writer" in researcher_prompt
    
    assert "research_team" in writer_prompt
    assert "researcher" in writer_prompt


@pytest.mark.asyncio
async def test_provider_specific_instructions():
    """Test that provider-specific instructions are included in the system prompt."""
    # Create models with different providers
    gemini_model = Model({
        'name': 'gemini_model',
        'provider': 'gemini',
        'development': True
    })
    
    anthropic_model = Model({
        'name': 'anthropic_model',
        'provider': 'anthropic',
        'development': True
    })
    
    # Generate system prompts
    gemini_prompt = gemini_model._generate_system_prompt()
    anthropic_prompt = anthropic_model._generate_system_prompt()
    
    # Verify provider-specific instructions
    assert "Tool Usage Instructions" in gemini_prompt
    assert "function calling syntax" in gemini_prompt
    
    assert "Tool Usage Instructions" in anthropic_prompt
    assert "<tool></tool> XML tags" in anthropic_prompt


@pytest.mark.asyncio
async def test_tool_description_formatting():
    """Test that tool descriptions are correctly formatted."""
    # Create a model
    model = Model({
        'name': 'test_model',
        'provider': 'gemini',
        'development': True
    })
    
    # Create some test tools
    tools = [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    # Format tool descriptions
    tool_description = model._prepare_tools_description(tools)
    
    # Verify the tool description
    assert "Available Tools" in tool_description
    assert "web_search" in tool_description
    assert "Search the web for information" in tool_description
    assert "Parameters:" in tool_description
    assert "`query`" in tool_description
    assert "required" in tool_description
    assert "`num_results`" in tool_description


if __name__ == "__main__":
    # Run tests directly when script is executed
    asyncio.run(test_system_prompt_generation())
    asyncio.run(test_team_context_in_prompt())
    asyncio.run(test_provider_specific_instructions())
    asyncio.run(test_tool_description_formatting())
    print("All tests passed!")
