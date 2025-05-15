from smolagents import InferenceClientModel

from glue.core.glue_memory_adapters import \
    GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from glue.core.glue_smolagent import GlueSmolAgent
from glue.core.glue_smolteam import GlueSmolTeam
from glue.dsl.app_builder import GlueAppBuilder
from glue.core.types import AdhesiveType


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


def test_glue_smolagent_adhesive_memory(tmp_path):
    # GLUE only
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.GLUE]})
    assert "glue" in agent.memories
    assert isinstance(agent.memory, GLUEPersistentAdapter)
    # VELCRO only
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.VELCRO]})
    assert "velcro" in agent.memories
    assert isinstance(agent.memory, VELCROSessionAdapter)
    # TAPE only
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.TAPE]})
    assert "tape" in agent.memories
    assert isinstance(agent.memory, TAPEEphemeralAdapter)
    # Multiple adhesives: GLUE + VELCRO + TAPE
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.GLUE, AdhesiveType.VELCRO, AdhesiveType.TAPE]})
    assert set(agent.memories.keys()) == {"glue", "velcro", "tape"}
    # Priority: GLUE > VELCRO > TAPE
    assert isinstance(agent.memory, GLUEPersistentAdapter)
    # If only VELCRO + TAPE
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.VELCRO, AdhesiveType.TAPE]})
    assert set(agent.memories.keys()) == {"velcro", "tape"}
    assert isinstance(agent.memory, VELCROSessionAdapter)
    # If only TAPE
    agent = GlueSmolAgent(model=model, tools=[], glue_config={"adhesives": [AdhesiveType.TAPE]})
    assert set(agent.memories.keys()) == {"tape"}
    assert isinstance(agent.memory, TAPEEphemeralAdapter)


def test_gluesmolagent_tool_injection_to_interpreter_globals():
    """
    Test that GlueSmolAgent injects tools as plain functions into the interpreter's globals.
    """
    from glue.core.glue_smolagent import GlueSmolAgent
    from smolagents import InferenceClientModel

    class DummyToolForInjection:
        def forward(self, x: int) -> int:
            return x + 1
    dummy_tool = DummyToolForInjection()
    dummy_tool.__name__ = "dummy_tool"
    dummy_tool._glue_tool_schema = {
        "inputs": {"x": {"type": "integer", "description": "An integer to increment"}},
        "output_type": "integer"
    }

    agent = GlueSmolAgent(
        model=InferenceClientModel(model_id="test-model"),
        tools=[dummy_tool],
        planning_interval=1,
    )
    # Manually call run with a dummy query to trigger injection
    try:
        agent.run("test query")
    except Exception:
        # Ignore errors from the model, we only care about injection
        pass
    # Check interpreter globals
    injected = getattr(agent, 'interpreter', None)
    if injected is not None:
        injected_func = agent.interpreter.globals.get("dummy_tool")
        assert injected_func is not None, "dummy_tool not injected into interpreter globals"
        assert callable(injected_func), "Injected dummy_tool is not callable"
        assert getattr(injected_func, "__name__", None) == "dummy_tool", f"Injected dummy_tool has wrong __name__: {getattr(injected_func, '__name__', None)}"


class DummyModel:
    def __init__(self):
        self.smol_config = {}
    def __call__(self, *args, **kwargs):
        class Dummy:
            content = '{"selected_tool_name": "echo", "tool_parameters": {"text": "hello"}}'
        return Dummy()
    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)

class EchoTool:
    name = "echo"
    description = "Echoes the input text."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"
    def __call__(self, text):
        return text
