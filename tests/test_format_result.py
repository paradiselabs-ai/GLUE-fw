import pytest
import json
from glue.core.agent_loop import TeamMemberAgentLoop


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses[:]

    async def generate(self, prompt):
        return self.responses.pop(0) if self.responses else ""


@pytest.mark.asyncio
async def test_format_result_stub_summary_and_context_on_success():
    # format_result should use generateSummaryStub and working memory entries
    loop = TeamMemberAgentLoop(
        agent_id="m1",
        team_id="t1",
        report_tool=lambda **kwargs: None,
        agent_llm=FakeLLM([]),
    )
    # Add a context entry to working memory
    loop.working_memory.add_entry(0, "entry1", "tool1")
    result_data = {"success": True}
    formatted = await loop.format_result(result_data)
    data = json.loads(formatted)
    # final_answer comes from generateSummaryStub
    assert data["final_answer"] == "Task completed successfully with artifacts: []"
    # supporting_context should include the memory entry
    assert len(data["supporting_context"]) == 1
    assert data["supporting_context"][0] == "entry1"


@pytest.mark.asyncio
async def test_format_result_stub_summary_and_context_on_error():
    # format_result should handle error results via stub
    loop = TeamMemberAgentLoop(
        agent_id="m1",
        team_id="t1",
        report_tool=lambda **kwargs: None,
        agent_llm=FakeLLM([]),
    )
    loop.working_memory.add_entry(0, "entryA", "toolA")
    result_data = {"error": "fail"}
    formatted = await loop.format_result(result_data)
    data = json.loads(formatted)
    # final_answer reflects failure stub
    assert data["final_answer"] == "Task failed: fail"
    # supporting_context should include the memory entry
    assert len(data["supporting_context"]) == 1
    assert data["supporting_context"][0] == "entryA"
