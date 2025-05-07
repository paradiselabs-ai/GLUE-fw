"""
Unit tests for the tool integration between GLUE and Agno.

These tests verify that GLUE tools are properly translated to Agno tools,
preserving tool functionality, configurations, and assignments to teams/agents.
"""

import pytest
import sys
from typing import Dict, Any, List, Callable

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
    
    def add_tool(self, tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
        return self

class StubTool:
    """Stub implementation of Agno Tool class for testing."""
    def __init__(self, name, description=None, function=None, config=None):
        self.name = name
        self.description = description
        self.function = function
        self.config = config or {}
    
    async def execute(self, **kwargs):
        """Execute the tool function if available."""
        if callable(self.function):
            return await self.function(**kwargs)
        return {"result": f"Executed {self.name}"}

class StubTeam:
    """Stub implementation of Agno Team class for testing."""
    def __init__(self, name, members=None, lead=None, config=None):
        self.name = name
        self.members = members or []
        self.lead = lead
        self.config = config
        self.tools = []
        self.communication_pattern = "hierarchical"  # Default pattern
    
    def add_tool(self, tool):
        """Add a tool to the team."""
        self.tools.append(tool)
        return self
    
    def add_tools(self, tools):
        """Add multiple tools to the team."""
        self.tools.extend(tools)
        return self

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
        self.tool = type('tool', (), {'Tool': StubTool})

# Add the stub module to sys.modules if not already present
if 'agno' not in sys.modules:
    sys.modules['agno'] = StubAgnoModule()
    sys.modules['agno.agent'] = sys.modules['agno'].agent
    sys.modules['agno.team'] = sys.modules['agno'].team
    sys.modules['agno.workflow'] = sys.modules['agno'].workflow
    sys.modules['agno.tool'] = sys.modules['agno'].tool

from glue.core.adapters.agno.adapter import GlueAgnoAdapter
from glue.core.adapters.agno.dsl_translator import GlueDSLAgnoTranslator


@pytest.fixture
def basic_tool_config():
    """Fixture providing a basic configuration with tools for testing."""
    return {
        "workflow": {
            "name": "ToolTestApp",
            "description": "Test application for tool integration"
        },
        "agents": {
            "MainAgent": {
                "provider": "openai",
                "model_name": "gpt-4"
            }
        },
        "teams": {
            "MainTeam": {
                "lead": "MainAgent",
                "members": ["MainAgent"],
                "tools": ["search_tool", "file_tool"],
                "communication_pattern": "hierarchical"
            }
        },
        "tools": {
            "search_tool": {
                "description": "Search the web for information",
                "params": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "adhesive": "GLUE"
            },
            "file_tool": {
                "description": "Read and write files",
                "params": {
                    "path": {
                        "type": "string",
                        "description": "The file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content",
                        "optional": True
                    }
                },
                "adhesive": "VELCRO"
            }
        },
        "flows": {}
    }


@pytest.fixture
def complex_tool_config():
    """Fixture providing a more complex configuration with multiple teams and tools."""
    return {
        "workflow": {
            "name": "ComplexToolTestApp",
            "description": "Test application with multiple teams and tools"
        },
        "agents": {
            "ResearchAgent": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "CodeAgent": {
                "provider": "anthropic",
                "model_name": "claude-3-opus"
            },
            "WritingAgent": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo"
            }
        },
        "teams": {
            "ResearchTeam": {
                "lead": "ResearchAgent",
                "members": ["ResearchAgent"],
                "tools": ["search_tool", "knowledge_tool"],
                "communication_pattern": "hierarchical"
            },
            "CodeTeam": {
                "lead": "CodeAgent",
                "members": ["CodeAgent"],
                "tools": ["code_tool", "file_tool"],
                "communication_pattern": "hierarchical"
            },
            "WritingTeam": {
                "lead": "WritingAgent",
                "members": ["WritingAgent"],
                "tools": ["file_tool", "grammar_tool"],
                "communication_pattern": "hierarchical"
            }
        },
        "tools": {
            "search_tool": {
                "description": "Search the web for information",
                "params": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "adhesive": "GLUE"
            },
            "knowledge_tool": {
                "description": "Access knowledge base",
                "params": {
                    "query": {
                        "type": "string",
                        "description": "The knowledge query"
                    }
                },
                "adhesive": "VELCRO"
            },
            "code_tool": {
                "description": "Generate and analyze code",
                "params": {
                    "language": {
                        "type": "string",
                        "description": "The programming language"
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to analyze",
                        "optional": True
                    }
                },
                "adhesive": "TAPE"
            },
            "file_tool": {
                "description": "Read and write files",
                "params": {
                    "path": {
                        "type": "string",
                        "description": "The file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content",
                        "optional": True
                    }
                },
                "adhesive": "VELCRO"
            },
            "grammar_tool": {
                "description": "Check grammar and style",
                "params": {
                    "text": {
                        "type": "string",
                        "description": "The text to check"
                    }
                },
                "adhesive": "GLUE"
            }
        },
        "flows": {}
    }


@pytest.mark.unit
def test_tool_translation():
    """Test that the DSL translator correctly translates GLUE tools to Agno tools."""
    # Create a simple GLUE AST with tool definitions
    glue_ast = {
        "app": {
            "app_name": "ToolTranslationTest",
            "engine": "agno"
        },
        "models": {
            "TestModel": {
                "provider": "openai",
                "model_name": "gpt-4"
            }
        },
        "teams": [
            {
                "name": "TestTeam",
                "lead": "TestModel",
                "models": ["TestModel"],
                "tools": ["search_tool", "file_tool"]
            }
        ],
        "tools": {
            "search_tool": {
                "description": "Search the web for information",
                "params": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "adhesive": "GLUE"
            },
            "file_tool": {
                "description": "Read and write files",
                "params": {
                    "path": {
                        "type": "string",
                        "description": "The file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content",
                        "optional": True
                    }
                },
                "adhesive": "VELCRO"
            }
        },
        "flows": []
    }
    
    # Translate the AST to Agno configuration
    translator = GlueDSLAgnoTranslator()
    agno_config = translator.translate(glue_ast)
    
    # Verify tools in the translated configuration
    assert "tools" in agno_config
    assert "search_tool" in agno_config["tools"]
    assert "file_tool" in agno_config["tools"]
    
    # Verify tool properties
    search_tool = agno_config["tools"]["search_tool"]
    assert search_tool["description"] == "Search the web for information"
    assert "params" in search_tool
    assert "query" in search_tool["params"]
    assert search_tool["adhesive"] == "GLUE"
    
    file_tool = agno_config["tools"]["file_tool"]
    assert file_tool["description"] == "Read and write files"
    assert "params" in file_tool
    assert "path" in file_tool["params"]
    assert "content" in file_tool["params"]
    assert file_tool["adhesive"] == "VELCRO"


@pytest.mark.unit
def test_tool_setup(basic_tool_config):
    """Test that tools are properly set up in the Agno adapter."""
    adapter = GlueAgnoAdapter()
    
    # The setup method should be updated to handle tools
    # This test will fail until we implement tool setup in the adapter
    success = adapter.setup(basic_tool_config)
    
    assert success is True
    assert "MainTeam" in adapter.teams
    
    # Verify team has tools
    team = adapter.teams["MainTeam"]
    assert len(team.tools) == 2
    
    # Verify tool properties
    tool_names = [tool.name for tool in team.tools]
    assert "search_tool" in tool_names
    assert "file_tool" in tool_names


@pytest.mark.unit
def test_multi_team_tool_assignment(complex_tool_config):
    """Test that tools are properly assigned to multiple teams."""
    adapter = GlueAgnoAdapter()
    
    # This test will fail until we implement tool setup in the adapter
    success = adapter.setup(complex_tool_config)
    
    assert success is True
    assert "ResearchTeam" in adapter.teams
    assert "CodeTeam" in adapter.teams
    assert "WritingTeam" in adapter.teams
    
    # Verify Research Team tools
    research_team = adapter.teams["ResearchTeam"]
    research_tool_names = [tool.name for tool in research_team.tools]
    assert "search_tool" in research_tool_names
    assert "knowledge_tool" in research_tool_names
    
    # Verify Code Team tools
    code_team = adapter.teams["CodeTeam"]
    code_tool_names = [tool.name for tool in code_team.tools]
    assert "code_tool" in code_tool_names
    assert "file_tool" in code_tool_names
    
    # Verify Writing Team tools
    writing_team = adapter.teams["WritingTeam"]
    writing_tool_names = [tool.name for tool in writing_team.tools]
    assert "file_tool" in writing_tool_names
    assert "grammar_tool" in writing_tool_names


@pytest.mark.unit
def test_shared_tools(complex_tool_config):
    """Test that tools can be shared between teams."""
    adapter = GlueAgnoAdapter()
    
    # This test will fail until we implement tool setup in the adapter
    success = adapter.setup(complex_tool_config)
    
    assert success is True
    
    # Verify file_tool is shared between CodeTeam and WritingTeam
    code_team = adapter.teams["CodeTeam"]
    writing_team = adapter.teams["WritingTeam"]
    
    code_tool_names = [tool.name for tool in code_team.tools]
    writing_tool_names = [tool.name for tool in writing_team.tools]
    
    assert "file_tool" in code_tool_names
    assert "file_tool" in writing_tool_names
