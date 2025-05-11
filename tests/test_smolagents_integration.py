import os
import tempfile
import pytest
from smolagents import InferenceClientModel, PythonInterpreterTool

from glue.core.glue_smoltool import GlueSmolTool
from glue.core.glue_memory_adapters import \
    GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from glue.core.glue_smolagent import GlueSmolAgent, GlueSmolToolCallingAgent
from glue.core.glue_smolteam import GlueSmolTeam
from glue.dsl.app_builder import GlueAppBuilder
from glue.core.types import AdhesiveType
from glue.tools.delegate_task_tool import DelegateTaskTool


class DummyTool:
    """
    A simple dummy GLUE tool stub for testing GlueSmolTool.
    """
    def __init__(self):
        self.called_with = None
        self.name = "delegate_task"
        self.description = "Test delegate task tool"
        self.inputs = {
            "target_agent_id": {"type": str, "required": True},
            "task_description": {"type": str, "required": True},
        }
        self.output_type = "Any"

    def execute(self, **kwargs):
        self.called_with = kwargs
        return kwargs

    def forward(self, x: int) -> int:
        return x + 1


def test_glue_smoltool_execute_calls_underlying_tool():
    tool = DummyTool()
    wrapper = GlueSmolTool(tool)
    output = wrapper(param=123)
    assert isinstance(output, dict)
    assert output["param"] == 123


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


def test_gluesmoltool_llm_friendly_argument_mapping():
    """
    Test that GlueSmolTool maps LLM-friendly argument names (target_name, task)
    to canonical tool argument names (target_agent_id, task_description).
    """
    dummy_tool = DummyTool()
    from glue.core.glue_smoltool import GlueSmolTool
    smol_tool = GlueSmolTool(dummy_tool)
    # Call with LLM-friendly names
    result = smol_tool(target_name="assistant_1", task="Do something")
    # The dummy tool should have received canonical names
    assert dummy_tool.called_with["target_agent_id"] == "assistant_1"
    assert dummy_tool.called_with["task_description"] == "Do something"
    # Also, the result should reflect the canonical mapping
    assert result["target_agent_id"] == "assistant_1"
    assert result["task_description"] == "Do something"


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


def test_delegate_task_injected_to_interpreter_globals():
    """
    Test that delegate_task is injected into the interpreter globals for GlueSmolAgent if present in tools.
    """
    from glue.core.glue_smolagent import GlueSmolAgent
    from smolagents import InferenceClientModel

    class DummyDelegateTool:
        def __call__(self, target_agent_id, task_description, parent_task_id):
            return f"Delegated {task_description} to {target_agent_id}"
        name = "delegate_task"
        description = "Delegate a task"
        inputs = {
            "target_agent_id": {"type": "string", "description": "Agent to delegate to"},
            "task_description": {"type": "string", "description": "Task description"},
            "parent_task_id": {"type": "string", "description": "Parent task id"},
        }
        output_type = "string"

    delegate_tool = DummyDelegateTool()
    agent = GlueSmolAgent(
        model=InferenceClientModel(model_id="test-model"),
        tools=[delegate_tool],
        planning_interval=1,
    )
    try:
        agent.run("test query")
    except Exception:
        pass
    injected = getattr(agent, 'interpreter', None)
    if injected is not None:
        injected_func = agent.interpreter.globals.get("delegate_task")
        assert injected_func is not None, "delegate_task not injected into interpreter globals"
        assert callable(injected_func), "Injected delegate_task is not callable"
        assert getattr(injected_func, "__name__", None) == "delegate_task", f"Injected delegate_task has wrong __name__: {getattr(injected_func, '__name__', None)}"


def test_gluesmolteam_always_injects_delegate_task():
    """
    Test that GlueSmolTeam always injects delegate_task into the interpreter globals of the lead agent,
    even if the team config does not explicitly include delegate_task in its tools list.
    """
    from glue.core.glue_smolteam import GlueSmolTeam
    from smolagents import InferenceClientModel, PythonInterpreterTool
    from glue.tools.delegate_task_tool import DelegateTaskTool
    class DummyConfig:
        lead = "lead1"
        members = []
    class DummyTeam:
        def __init__(self):
            self.name = "teamY"
            self.config = DummyConfig()
            # Add both delegate_task and a code interpreter tool
            self.tools = [DelegateTaskTool(app=None), PythonInterpreterTool()]
    dummy_team = DummyTeam()
    clients = {"lead1": InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")}
    smol_team = GlueSmolTeam(team=dummy_team, model_clients=clients)
    smol_team.setup()
    agent = smol_team.lead_agent
    # Force interpreter creation and tool injection for robust testing
    agent.force_interpreter()
    injected_globals = agent.interpreter.globals
    assert injected_globals is not None, "interpreter globals not available on agent"
    injected_func = injected_globals.get("delegate_task")
    assert injected_func is not None, "delegate_task not injected into interpreter globals by GlueSmolTeam"
    assert callable(injected_func), "Injected delegate_task is not callable"
    assert getattr(injected_func, "__name__", None) == "delegate_task", f"Injected delegate_task has wrong __name__: {getattr(injected_func, '__name__', None)}"


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

def test_toolcallingagent_metadata_required():
    with pytest.raises(ValueError):
        GlueSmolToolCallingAgent(model=DummyModel(), tools=[], name=None, description="desc")
    with pytest.raises(ValueError):
        GlueSmolToolCallingAgent(model=DummyModel(), tools=[], name="agent", description=None)

def test_toolcallingagent_tool_schema_validation():
    class BadTool:
        name = "bad"
        description = ""
        inputs = {}
        output_type = None
        def __call__(self):
            return "bad"
    with pytest.raises(ValueError):
        GlueSmolToolCallingAgent(model=DummyModel(), tools=[BadTool()], name="agent", description="desc")

def test_toolcallingagent_simple_run():
    agent = GlueSmolToolCallingAgent(
        model=DummyModel(),
        tools=[EchoTool()],
        name="agent",
        description="desc"
    )
    result = agent.run("Echo this")
    assert result == "hello"

def test_toolcallingagent_managed_agents():
    # Managed agent is just another GlueSmolToolCallingAgent
    managed = GlueSmolToolCallingAgent(
        model=DummyModel(),
        tools=[EchoTool()],
        name="subagent",
        description="desc"
    )
    agent = GlueSmolToolCallingAgent(
        model=DummyModel(),
        tools=[EchoTool()],
        name="agent",
        description="desc",
        managed_agents=[managed]
    )
    assert "subagent" in agent.managed_agents