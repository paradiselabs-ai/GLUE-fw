import pytest
from glue.core.agent_loop import TeamMemberAgentLoop


@pytest.mark.asyncio
async def test_auto_report_tool_injection():
    # Prepare a stub raw_report_tool to capture arguments
    captured = {}

    async def raw_report_tool(
        *,
        task_id,
        status,
        detailed_answer,
        artifact_keys=None,
        calling_agent_id=None,
        calling_team=None,
    ):
        captured["task_id"] = task_id
        captured["status"] = status
        captured["detailed_answer"] = detailed_answer
        captured["artifact_keys"] = artifact_keys
        captured["calling_agent_id"] = calling_agent_id
        captured["calling_team"] = calling_team
        return {"success": True}

    # Instantiate the loop with stub report_tool
    loop = TeamMemberAgentLoop(
        agent_id="agent1", team_id="team1", report_tool=raw_report_tool, agent_llm=None
    )
    # Simulate assignment of a current task
    loop.state["current_task_id"] = "team1-task-0001"

    # Call the report_tool wrapper
    result = await loop.report_tool(
        status="success",
        detailed_answer="Completed successfully",
        artifact_keys=["artifact1"],
        calling_agent_id="agent1",
        calling_team="team1",
    )

    # Verify injection and return value
    assert captured["task_id"] == "team1-task-0001"
    assert captured["status"] == "success"
    assert captured["detailed_answer"] == "Completed successfully"
    assert captured["artifact_keys"] == ["artifact1"]
    assert captured["calling_agent_id"] == "agent1"
    assert captured["calling_team"] == "team1"
    assert result == {"success": True}
