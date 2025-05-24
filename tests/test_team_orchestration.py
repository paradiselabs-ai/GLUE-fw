import pytest
from smolagents import InferenceClientModel
from glue.core.glue_smolteam import GlueSmolTeam
from glue.core.glue_smolagent import GlueSmolAgent
from glue.core.types import FlowType

class DummyConfig:
    lead = "lead1"
    members = ["lead1", "member1"]
    tools = []

class DummyTeam:
    def __init__(self):
        self.name = "teamX"
        self.config = DummyConfig()
        self.tools = []
        self.models = {}
        self.subteams = {}
        self.relationships = {}

    async def send_information(self, target_team, content):
        # no-op for test
        return True

    async def receive_information(self, source_team, content):
        # no-op for test
        return True

@pytest.fixture
def model_clients():
    lead_model = InferenceClientModel(model_id="model_lead")
    lead_model.name = "lead1"
    member_model = InferenceClientModel(model_id="model_member")
    member_model.name = "member1"
    return {"lead1": lead_model, "member1": member_model}


def test_member_agents_registered(model_clients):
    team = DummyTeam()
    team.models = model_clients
    smol_team = GlueSmolTeam(team=team, model_clients=model_clients)
    # Should not raise
    smol_team.setup()
    # After setup, member_agents should contain member1
    assert "member1" in smol_team.member_agents
    assert isinstance(smol_team.lead_agent, GlueSmolAgent)
    assert isinstance(smol_team.member_agents["member1"], GlueSmolAgent)
    # managed_agents in lead_agent should include member1
    assert "member1" in smol_team.lead_agent.managed_agents


def test_flow_function_registration(model_clients):
    team = DummyTeam()
    team.models = model_clients
    team.relationships = {
        "member1": FlowType.BIDIRECTIONAL.value,
        "member2": FlowType.PUSH.value,
        "member3": FlowType.PULL.value,
        "member4": FlowType.REPEL.value,
    }
    smol_team = GlueSmolTeam(team=team, model_clients=model_clients)
    smol_team.setup()
    lg = smol_team.lead_agent.interpreter_globals
    # push_to for BIDIRECTIONAL and PUSH
    assert "push_to_member1" in lg
    assert "push_to_member2" in lg
    # pull_from for BIDIRECTIONAL and PULL
    assert "pull_from_member1" in lg
    assert "pull_from_member3" in lg
    # repel should not register flows for member4
    assert "push_to_member4" not in lg
    assert "pull_from_member4" not in lg