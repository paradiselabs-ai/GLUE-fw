import pytest
from glue.core.agent_loop import TeamMemberAgentLoop
from glue.core.schemas import ParseAnalyzeOutput, PlanPhaseOutput


class DummyReportTool:
    def __call__(self, **kwargs):
        return {"reported": True}


@pytest.fixture
def member_loop():
    # Create TeamMemberAgentLoop without LLM for stub behavior
    return TeamMemberAgentLoop(
        agent_id="agent", team_id="team", report_tool=DummyReportTool(), agent_llm=None
    )


@pytest.mark.asyncio
async def test_parse_and_analyze_stub(member_loop):
    # Without LLM, parse_and_analyze returns stub keywords
    out = await member_loop.parse_and_analyze("Compute sum of list")
    assert isinstance(out, ParseAnalyzeOutput)
    # analysis should contain 'keywords'
    assert "keywords" in out.analysis
    assert isinstance(out.analysis["keywords"], list)


@pytest.mark.asyncio
async def test_plan_phase_stub(member_loop):
    # Without LLM, plan_phase returns stub three steps
    memory = [{"description": "Task1"}]
    out = await member_loop.plan_phase(memory)
    assert isinstance(out, PlanPhaseOutput)
    assert len(out.substeps) == 3
    assert out.estimated_confidence == "high"


@pytest.mark.asyncio
async def test_select_and_invoke_tool_stub(member_loop):
    # Without LLM, selects performTaskStub and returns its simulated result
    tool, result = await member_loop.select_and_invoke_tool("do something", None)
    assert tool == "performTaskStub"
    assert result == {"success": True, "artifacts": []}
