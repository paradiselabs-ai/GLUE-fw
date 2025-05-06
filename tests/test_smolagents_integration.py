import os
import tempfile
import pytest
from smolagents import InferenceClientModel

from glue.core.glue_smoltool import GlueSmolTool
from glue.core.glue_memory_adapters import \
    GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from glue.core.glue_smolagent import GlueSmolAgent
from glue.core.glue_smolteam import GlueSmolTeam
from glue.dsl.app_builder import GlueAppBuilder


class DummyTool:
    """
    A simple dummy GLUE tool stub for testing GlueSmolTool.
    """
    def __init__(self):
        self.name = "dummy"
        self.description = "dummy tool"
        self.inputs = {"param": "type"}
        self.output_type = "string"

    def execute(self, **kwargs):
        return {"result": True, "kwargs": kwargs}


def test_glue_smoltool_execute_calls_underlying_tool():
    tool = DummyTool()
    wrapper = GlueSmolTool(tool)
    output = wrapper(param=123)
    assert isinstance(output, dict)
    assert output["result"]
    assert output["kwargs"]["param"] == 123


def test_memory_adapters_persistence_and_clear(tmp_path):
    # GLUE persistent adapter
    glue_adapter = GLUEPersistentAdapter(team_id="team1", memory_dir=str(tmp_path))
    entry = {"msg": "hello"}
    glue_adapter.add(entry)
    assert glue_adapter.get_all() == [entry]

    # VELCRO session adapter
    velcro = VELCROSessionAdapter()
    velcro.add(1, "hi", "tool1")
    entries = velcro.get_all()
    assert len(entries) == 1
    assert entries[0]["content"] == "hi"
    velcro.clear()
    assert velcro.get_all() == []

    # TAPE ephemeral adapter
    tape = TAPEEphemeralAdapter()
    tape.add("key1", "val1")
    assert tape.get("key1") == "val1"
    # After retrieval, value should be removed
    assert tape.get("key1") is None


def test_glue_smolagent_placeholder_injection():
    # Instantiate agent with no tools for prompt template test
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")
    agent = GlueSmolAgent(model=model, tools=[], planning_interval=1)
    system_prompt = agent.prompt_templates.get("system_prompt", "")
    # Check required placeholders are present
    assert "{{tool_descriptions}}" in system_prompt
    assert "{{managed_agents_description}}" in system_prompt
    assert "{{authorized_imports}}" in system_prompt


def test_glue_smolteam_setup_with_lead_only():
    # Define a minimal dummy Team
    class DummyConfig:
        lead = "lead1"
        members = []

    class DummyTeam:
        def __init__(self):
            self.name = "teamX"
            self.config = DummyConfig()
            self.tools = []

    dummy_team = DummyTeam()
    clients = {"lead1": InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")}
    smol_team = GlueSmolTeam(team=dummy_team, model_clients=clients)
    # Should not raise
    smol_team.setup()
    assert hasattr(smol_team, "lead_agent")
    assert isinstance(smol_team.lead_agent, GlueSmolAgent)
    assert smol_team.member_agents == {}


def test_dsl_builder_extracts_smol_config():
    builder = GlueAppBuilder()
    config = {
        "app": {"name": "test-app", "description": "desc"},
        "models": [
            {
                "name": "m1",
                "provider": "openrouter",
                "config": {
                    "model": "meta-llama/test:free",
                    "planning_interval": 5,
                    "system_prompt": "Test prompt"
                }
            }
        ],
        "tools": [],
        "teams": [],
        "flows": []
    }
    app = builder.build(config)
    model = app.models.get("m1")
    assert hasattr(model, "smol_config")
    assert model.smol_config.get("planning_interval") == 5
    assert model.smol_config.get("system_prompt") == "Test prompt" 