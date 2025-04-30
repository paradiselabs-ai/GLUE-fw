import pytest
from pydantic import ValidationError

from glue.core.orchestrator_schemas import Subtask, ReportRecord, EvaluateDecision
from glue.tools.delegate_task_tool import DelegateTaskArgs
from glue.tools.report_task_completion_tool import ReportTaskCompletionArgs

# Tests for Subtask schema


def test_subtask_valid():
    data = {"id": "sub1", "description": "Do something"}
    s = Subtask.model_validate(data)
    assert s.id == "sub1"
    assert s.description == "Do something"


@pytest.mark.parametrize("data", [{}, {"id": "sub1"}, {"description": "Missing id"}])
def test_subtask_invalid(data):
    with pytest.raises(ValidationError):
        Subtask.model_validate(data)


# Tests for ReportRecord schema


def test_reportrecord_valid():
    data = {
        "task_id": "task1",
        "status": "success",
        "detailed_answer": "All good.",
        "artifact_keys": ["a1", "a2"],
    }
    r = ReportRecord.model_validate(data)
    assert r.task_id == "task1"
    assert r.status == "success"


@pytest.mark.parametrize(
    "data",
    [
        {"task_id": "task1", "status": "invalid", "detailed_answer": "X"},
        {"status": "success", "detailed_answer": "X"},
        {"task_id": "task1", "detailed_answer": "Missing status"},
    ],
)
def test_reportrecord_invalid(data):
    with pytest.raises(ValidationError):
        ReportRecord.model_validate(data)


# Tests for EvaluateDecision schema


def test_evaluate_decision_valid():
    data = {"task_id": "task1", "action": "mark_complete", "details": "OK"}
    d = EvaluateDecision.model_validate(data)
    assert d.task_id == "task1"
    assert d.action == "mark_complete"


@pytest.mark.parametrize(
    "data",
    [
        {"task_id": "task1", "action": "unknown"},
        {"action": "mark_complete"},
        {"task_id": "task1"},
    ],
)
def test_evaluate_decision_invalid(data):
    with pytest.raises(ValidationError):
        EvaluateDecision.model_validate(data)


# Tests for DelegateTaskArgs schema


def test_delegate_task_args_valid():
    args = {
        "target_agent": "agent1",
        "task_description": "Do X",
        "parent_task_id": "parent1",
        "calling_team": "teamA",
        "context_keys": [],
        "required_artifacts": [],
    }
    dt = DelegateTaskArgs(**args)
    assert dt.target_agent == "agent1"


@pytest.mark.parametrize(
    "args",
    [
        {"task_description": "No target"},
        {"target_agent": "agent1"},
        {"target_agent": "agent1", "task_description": "desc"},
    ],
)
def test_delegate_task_args_invalid(args):
    with pytest.raises(ValidationError):
        DelegateTaskArgs(**args)


# Tests for ReportTaskCompletionArgs schema


def test_report_task_completion_args_valid():
    args = {
        "task_id": "task1",
        "status": "success",
        "detailed_answer": "Done",
        "artifact_keys": ["a1"],
    }
    rc = ReportTaskCompletionArgs(**args)
    assert rc.task_id == "task1"
    assert rc.status == "success"


@pytest.mark.parametrize(
    "args",
    [
        {"status": "success", "detailed_answer": "Done"},
        {"task_id": "task1", "detailed_answer": "Done"},
        {"task_id": "task1", "status": "completed", "detailed_answer": "Done"},
    ],
)
def test_report_task_completion_args_invalid(args):
    # 'completed' is normalized to 'success', so last case should pass
    should_fail = args.get("status") != "completed"
    if should_fail:
        with pytest.raises(ValidationError):
            ReportTaskCompletionArgs(**args)
    else:
        # 'completed' should normalize to 'success'
        rc = ReportTaskCompletionArgs(**args)
        assert rc.status == "success"
