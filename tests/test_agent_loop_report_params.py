import pytest
from pydantic import ValidationError
from glue.core.agent_loop import LLMReportParams


def test_valid_params_full_fields():
    params = {
        "status": "success",
        "detailed_answer": "All good.",
        "artifact_keys": ["key1", "key2"],
        "failure_reason": None,
    }
    model = LLMReportParams(**params)
    assert model.status == "success"
    assert model.detailed_answer == "All good."
    assert model.artifact_keys == ["key1", "key2"]
    assert model.failure_reason is None


@pytest.mark.parametrize(
    "input_status,expected_status",
    [
        ("success", "success"),
        ("failure", "failure"),
        ("escalation", "escalation"),
        ("completed", "success"),  # normalized to success
    ],
)
def test_status_variants(input_status, expected_status):
    model = LLMReportParams(status=input_status, detailed_answer="Detail.")
    assert model.status == expected_status


def test_default_artifact_keys():
    model = LLMReportParams(status="success", detailed_answer="OK")
    assert model.artifact_keys == []


def test_missing_status_raises():
    with pytest.raises(ValidationError):
        LLMReportParams(detailed_answer="Missing status")


def test_missing_detailed_answer_raises():
    with pytest.raises(ValidationError):
        LLMReportParams(status="success")


def test_extra_fields_are_ignored():
    data = {"status": "success", "detailed_answer": "OK", "unexpected": "ignore this"}
    model = LLMReportParams(**data)
    assert not hasattr(model, "unexpected")
    # Confirm required fields still present
    assert model.status == "success"
    assert model.detailed_answer == "OK"
