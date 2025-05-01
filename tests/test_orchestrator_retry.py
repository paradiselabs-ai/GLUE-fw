import pytest
import asyncio
from types import SimpleNamespace
from glue.core.agent_loop import TeamLeadAgentLoop
from glue.core.orchestrator_schemas import ReportRecord, TaskRecord
from glue.core.types import TaskStatus


class DummyDelegateTool:
    async def __call__(self, **kwargs):
        return {"task": {"task_id": kwargs.get("target_agent_id") + "-task-00000001"}}


@pytest.fixture
async def orchestrator_event_loop(tmp_path):
    # Provide a running event loop for async tests
    loop = asyncio.get_event_loop()
    yield loop


@pytest.fixture
def dummy_team():
    # Dummy team with two agents
    return SimpleNamespace(
        config=SimpleNamespace(lead="agent1", members=["agent1", "agent2"]), name="team"
    )


@pytest.fixture
def loop(dummy_team):
    # Create orchestrator with no LLM (fallback)
    return TeamLeadAgentLoop(
        team=dummy_team, delegate_tool=DummyDelegateTool(), agent_llm=None
    )


@pytest.mark.asyncio
async def test_retry_logic_less_than_max(loop):
    # Setup single subtask with retries < max
    record = TaskRecord(
        description="desc", dependencies=[], state=TaskStatus.PENDING.value, retries=0
    )
    record.task_id = "task-1"
    loop.task_states = {"sub1": record}
    report = ReportRecord(
        task_id="task-1",
        status="failure",
        detailed_answer="error",
        artifact_keys=[],
        failure_reason=None,
    )
    await loop._evaluate_report(report)
    assert record.retries == 1
    assert record.state == TaskStatus.PENDING_RETRY.value
    assert not loop.terminated


@pytest.mark.asyncio
async def test_retry_logic_exceeds_max(loop):
    # Setup single subtask at max retries
    loop.max_retries = 1
    record = TaskRecord(
        description="desc", dependencies=[], state=TaskStatus.PENDING.value, retries=1
    )
    record.task_id = "task-2"
    loop.task_states = {"sub2": record}
    report = ReportRecord(
        task_id="task-2",
        status="failure",
        detailed_answer="error",
        artifact_keys=[],
        failure_reason=None,
    )
    await loop._evaluate_report(report)
    assert record.retries == 1  # should not increment beyond max
    assert record.state == TaskStatus.FAILED.value
    assert loop.terminated
