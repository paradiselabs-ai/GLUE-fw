# tests/unit/dsl/test_parser.py

import pytest
from glue.dsl.parser import StickyScriptParser # Assuming parser class will be StickyScriptParser

@pytest.mark.unit
def test_parse_minimal_app_declaration():
    script = "app MyApp"
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "MyApp"},
        "agents": {},
        "teams": {},
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_app_with_description():
    script = 'app MyApp { description: "This is a test app." }'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "MyApp", "description": "This is a test app."},
        "agents": {},
        "teams": {},
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_minimal_agent_declaration():
    # In the current grammar, agent options are not optional, so a 'minimal' agent needs all its defined options.
    script = 'agent MyAgent { model: "gpt-4", provider: "openai", instructions: "Be helpful." }'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"}, # Default app name if no app_decl
        "agents": {
            "MyAgent": {"model": "gpt-4", "provider": "openai", "instructions": "Be helpful."}
        },
        "teams": {},
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_agent_with_all_options(): # Same as minimal due to grammar
    script = 'agent MyAgent { model: "claude-2", provider: "anthropic", instructions: "Be concise." }'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {
            "MyAgent": {"model": "claude-2", "provider": "anthropic", "instructions": "Be concise."}
        },
        "teams": {},
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_minimal_tool_declaration():
    script = 'tool MyTool'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {},
        "teams": {},
        "tools": {"MyTool": {}},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_tool_with_description():
    script = 'tool MyTool { description: "A useful tool." }'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {},
        "teams": {},
        "tools": {"MyTool": {"description": "A useful tool."}},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_minimal_team_declaration():
    # Team options (lead, members, tools, instructions) are mandatory in the current grammar block for team_decl
    script = 'team MyTeam { lead: LeadAgent, members: [], tools: [], instructions: "Work together." }'
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {},
        "teams": {
            "MyTeam": {"lead": "LeadAgent", "members": [], "tools": [], "instructions": "Work together."}
        },
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_team_with_all_options():
    script = """
    team MyTeam {
        lead: Supervisor,
        members: [AgentOne, AgentTwo],
        tools: [ToolA, ToolB],
        instructions: "Collaborate effectively."
    }
    """
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {},
        "teams": {
            "MyTeam": {
                "lead": "Supervisor",
                "members": ["AgentOne", "AgentTwo"],
                "tools": ["ToolA", "ToolB"],
                "instructions": "Collaborate effectively."
            }
        },
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_team_with_empty_members_tools_lists():
    script = """
    team MyTeam {
        lead: Supervisor,
        members: [],
        tools: [],
        instructions: "Collaborate effectively."
    }
    """
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "DefaultGlueApp"},
        "agents": {},
        "teams": {
            "MyTeam": {
                "lead": "Supervisor",
                "members": [],
                "tools": [],
                "instructions": "Collaborate effectively."
            }
        },
        "tools": {},
        "flows": {}
    }
    assert parser.parse(script) == expected_config

@pytest.mark.unit
def test_parse_basic_flow_declaration():
    script = "flow TeamA -> TeamB"
    parser = StickyScriptParser()
    # The flow name is auto-generated (flow_0, flow_1, etc.)
    # So we check for the presence and structure of the first flow.
    parsed_config = parser.parse(script)
    assert "flow_0" in parsed_config["flows"]
    assert parsed_config["flows"]["flow_0"] == {"from": "TeamA", "to": "TeamB", "type": "basic"}
    assert parsed_config["workflow"]["name"] == "DefaultGlueApp" # Ensure default app name

@pytest.mark.unit
def test_parse_multiple_declarations():
    script = """
    app TestSuiteApp { description: "Comprehensive test."}
    agent Coder { model: "gpt-4", provider: "openai", instructions: "Write code." }
    tool Search { description: "Web search tool." }
    team DevTeam {
        lead: Coder,
        members: [Coder],
        tools: [Search],
        instructions: "Develop the feature."
    }
    flow DevTeam -> QA_Team
    """
    parser = StickyScriptParser()
    expected_config = {
        "workflow": {"name": "TestSuiteApp", "description": "Comprehensive test."},
        "agents": {
            "Coder": {"model": "gpt-4", "provider": "openai", "instructions": "Write code."}
        },
        "teams": {
            "DevTeam": {
                "lead": "Coder",
                "members": ["Coder"],
                "tools": ["Search"],
                "instructions": "Develop the feature."
            }
        },
        "tools": {
            "Search": {"description": "Web search tool."}
        },
        "flows": {
            "flow_0": {"from": "DevTeam", "to": "QA_Team", "type": "basic"}
        }
    }
    assert parser.parse(script) == expected_config
