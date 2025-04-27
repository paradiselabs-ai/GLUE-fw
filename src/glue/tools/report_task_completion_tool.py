"""
Report task completion tool for GLUE framework.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

from .tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType, ToolResult
from pydantic import BaseModel, Field, ValidationError, model_validator, ConfigDict

logger = logging.getLogger("glue.tools.report_task_completion")

# PydanticAI schema for report_task_completion arguments
class ReportTaskCompletionArgs(BaseModel):
    task_id: str = Field(..., alias="task_id", min_length=1, description="ID of the completed task")
    status: Literal["success", "failure"] = Field(..., description="Completion status (e.g., 'success' or 'failure')")
    detailed_answer: str = Field(..., min_length=1, description="Detailed answer of task results")
    artifact_keys: List[str] = Field(default_factory=list, description="Optional keys to artifacts produced")

    @model_validator(mode="before")
    def normalize_status_and_alias(cls, data):
        # Accept alias 'task_id' and normalize 'completed' status
        # Map status 'completed' to 'success'
        status = data.get('status') or data.get('status'.lower())
        if status == 'completed':
            data['status'] = 'success'
        return data

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

class ReportTaskCompletionTool(Tool):
    """Tool for signalling task completion and stopping an agent's loop."""
    def __init__(self, app=None):
        """
        Initialize the report completion tool.
        Args:
            app: Reference to the GLUE application instance.
        """
        self.app = app
        name = "report_task_completion"
        description = "Signal completion of a task, provide results to the team lead, and terminate this agent's loop."
        config = ToolConfig(
            timeout=30.0,
            max_retries=0,
            required_permissions={ToolPermission.WRITE},
            adhesive_types={AdhesiveType.GLUE},
        )
        super().__init__(name=name, description=description, config=config)
        self.inputs = {
            "task_id": {"type": str, "description": "ID of the completed task", "required": True},
            "status": {"type": str, "description": "Completion status (e.g., 'success' or 'failure')", "required": True},
            "detailed_answer": {"type": str, "description": "Detailed answer of task results", "required": True},
            "artifact_keys": {"type": list, "description": "Optional keys to artifacts produced", "required": False, "default": []},
        }

    async def _execute(
        self,
        task_id: str,
        status: str,
        detailed_answer: str,
        artifact_keys: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Report task completion, notify the lead, and terminate the agent's loop.
        """
        # Validate app and context
        if not self.app:
            logger.error("ReportTaskCompletionTool: Application context is missing.")
            return {"success": False, "error": "Application context is missing."}

        # PydanticAI validate incoming parameters
        try:
            # Use Pydantic alias and normalization
            args = ReportTaskCompletionArgs(
                task_id=task_id,
                status=status,
                detailed_answer=detailed_answer,
                artifact_keys=artifact_keys or []
            )
        except ValidationError as e:
            logger.error(f"ReportTaskCompletionTool: argument validation error: {e}")
            return {"success": False, "error": str(e)}

        # Map validated fields back
        task_id = args.task_id
        status = args.status
        detailed_answer = args.detailed_answer
        artifact_keys = args.artifact_keys

        # Infer calling_agent and calling_team if missing
        calling_agent = kwargs.get("calling_agent_id") or kwargs.get("calling_model")
        if not calling_agent:
            error = "ReportTaskCompletionTool: Missing calling_agent_id or calling_model in kwargs."
            logger.error(error)
            return {"success": False, "error": error}
        if not kwargs.get("calling_team"):
            for team_name, team in self.app.teams.items():
                if calling_agent in getattr(team, 'models', {}):
                    kwargs["calling_team"] = team_name
                    break
        calling_team = kwargs.get("calling_team")
        if not calling_team:
            error = "ReportTaskCompletionTool: Missing calling_team in kwargs."
            logger.error(error)
            return {"success": False, "error": error}

        # Ensure the team exists
        team = self.app.teams.get(calling_team)
        if not team:
            error = f"ReportTaskCompletionTool: Team '{calling_team}' not found."
            logger.error(error)
            return {"success": False, "error": error}

        # Ensure task exists otherwise error out to catch mismatched IDs
        if task_id not in team.shared_results:
            error_msg = f"ReportTaskCompletionTool: Task '{task_id}' not found in team '{team.name}'. Cannot report mismatched id."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Record the completion details
        record = {
            "task_id": task_id,
            "status": status,
            "detailed_answer": detailed_answer,
            "artifact_keys": artifact_keys or [],
            "reported_by": calling_agent,
            "reported_at": datetime.utcnow().isoformat()
        }
        # Attach to shared_results
        team.shared_results[task_id]["completion"] = record
        # Share this completion as a tool result for the lead orchestrator to consume
        tr = ToolResult(tool_name=self.name, result=record)
        await team.share_result(self.name, tr, model_name=calling_agent)

        # Notify the team lead via internal message
        lead_id = team.config.lead
        if lead_id and lead_id in team.models:
            internal_msg = {
                "content": {"task_completed": record},
                "metadata": {
                    "source_team": team.name,
                    "target_team": team.name,
                    "source_model": calling_agent,
                    "target_model": lead_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "internal": True
                }
            }
            # enqueue for team history
            await team.message_queue.put((internal_msg, None))
            # direct notify lead model
            try:
                await team.direct_communication(calling_agent, lead_id, record)
            except Exception as e:
                logger.warning(f"ReportTaskCompletionTool: direct_communication error: {e}")
        else:
            logger.warning("ReportTaskCompletionTool: No lead to notify.")

        # Terminate this agent's loop
        if hasattr(team, 'agent_loops') and calling_agent in team.agent_loops:
            try:
                team.agent_loops[calling_agent].terminate(f"Task {task_id} completed")
            except Exception as e:
                logger.warning(f"ReportTaskCompletionTool: error terminating agent loop: {e}")

        logger.info(f"Reported completion for task {task_id} by agent {calling_agent} in team {team.name}")
        return {"success": True, "completion": record} 