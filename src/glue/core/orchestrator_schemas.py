from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Subtask(BaseModel):
    id: str = Field(..., description="Subtask identifier")
    description: str = Field(..., description="Subtask description")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite subtask IDs")
    assigned_agent_id: Optional[str] = Field(None, description="Specific agent ID this subtask should ideally be assigned to, if specified.")

class TaskRecord(BaseModel):
    """Structured record for tracking subtask state."""
    description: str = Field(..., description="Subtask description")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite subtask IDs")
    state: str = Field(..., description="Current status of the subtask")
    retries: int = Field(0, description="Number of retries attempted")
    task_id: Optional[str] = Field(None, description="Assigned task ID from delegation tool")
    timestamp: Optional[str] = Field(None, description="ISO timestamp when task was delegated or updated")

class DecomposeOutput(BaseModel):
    """Wrapper model for LLM decomposition output."""
    subtasks: List[Subtask] = Field(..., description="List of decomposed subtasks with dependencies")

class ReportRecord(BaseModel):
    task_id: str = Field(..., description="ID of the task")
    status: Literal["success","failure","escalation"] = Field(..., description="Completion status")
    detailed_answer: str = Field(..., description="Detailed answer of task results")
    artifact_keys: List[str] = Field(default_factory=list, description="Optional artifact keys")
    failure_reason: Optional[str] = Field(None, description="Concise explanation for why the task failed, if applicable")

class EvaluateDecision(BaseModel):
    task_id: str = Field(..., description="ID of the task")
    action: Literal["mark_complete","retry","escalate"] = Field(..., description="Orchestration action")
    details: Optional[str] = Field(None, description="Additional details for decision")

class FinalResult(BaseModel):
    final_answer: str = Field(..., description="Synthesized final answer")
    supporting_context: List[str] = Field(default_factory=list, description="Supporting context entries") 