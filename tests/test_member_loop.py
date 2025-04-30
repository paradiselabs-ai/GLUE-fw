import pytest
import json
from glue.core.agent_loop import TeamMemberAgentLoop

@pytest.mark.asyncio
async def test_team_member_agent_loop_completes_and_reports():
    reported = {}

    async def fake_report_tool(task_id, status, detailed_answer, artifact_keys=None, calling_agent_id=None, calling_team=None):
        reported['task_id'] = task_id
        reported['status'] = status
        reported['detailed_answer'] = detailed_answer
        reported['artifact_keys'] = artifact_keys
        reported['calling_agent_id'] = calling_agent_id
        reported['calling_team'] = calling_team

    async def fetch_task(agent_id):
        return {'task_id': 'fake-task', 'description': 'Fake description'}

    # Create loop without an LLM to use fallback stubs
    loop = TeamMemberAgentLoop(agent_id='member1', team_id='team1', report_tool=fake_report_tool, agent_llm=None)
    await loop.start(fetch_task)

    # Ensure loop finished
    assert loop.state['status'] == 'completed'
    # Ensure report_tool was called with correct parameters
    assert reported.get('task_id') == 'fake-task'
    assert reported.get('status') == 'success'
    assert reported.get('calling_agent_id') == 'member1'
    assert reported.get('calling_team') == 'team1'
    # Parse the detailed_answer JSON
    fmt = json.loads(reported.get('detailed_answer'))
    assert 'final_answer' in fmt
    assert isinstance(fmt['supporting_context'], list)
    # artifact_keys should be default empty list
    assert reported.get('artifact_keys') == [] 