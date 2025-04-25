"""
Report task completion tool for GLUE framework.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from .tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType

logger = logging.getLogger("glue.tools.report_task_completion")

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
            "completed_task_id": {"type": str, "description": "ID of the completed task", "required": True},
            "status": {"type": str, "description": "Completion status (e.g., done, partial)", "required": True},
            "result_summary": {"type": str, "description": "Summary of task results", "required": True},
            "artifact_keys": {"type": list, "description": "Optional keys to artifacts produced", "required": False, "default": []},
        }

    async def _execute(
        self,
        completed_task_id: str,
        status: str,
        result_summary: str,
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

        # Identify calling agent and team
        calling_agent = kwargs.get("calling_agent_id") or kwargs.get("calling_model")
        calling_team = kwargs.get("calling_team")
        if not calling_agent or not calling_team:
            error = "ReportTaskCompletionTool: Missing calling_agent_id or calling_team in kwargs."
            logger.error(error)
            return {"success": False, "error": error}

        team = self.app.teams.get(calling_team)
        if not team:
            error = f"ReportTaskCompletionTool: Team '{calling_team}' not found."
            logger.error(error)
            return {"success": False, "error": error}

        # Ensure task exists
        if completed_task_id not in team.shared_results:
            error = f"ReportTaskCompletionTool: Task '{completed_task_id}' not found in team '{team.name}'."
            logger.error(error)
            return {"success": False, "error": error}

        # Record the completion details
        record = {
            "task_id": completed_task_id,
            "status": status,
            "result_summary": result_summary,
            "artifact_keys": artifact_keys or [],
            "reported_by": calling_agent,
            "reported_at": datetime.utcnow().isoformat()
        }
        # Attach to shared_results
        team.shared_results[completed_task_id]["completion"] = record

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
                team.agent_loops[calling_agent].terminate(f"Task {completed_task_id} completed")
            except Exception as e:
                logger.warning(f"ReportTaskCompletionTool: error terminating agent loop: {e}")

        logger.info(f"Reported completion for task {completed_task_id} by agent {calling_agent} in team {team.name}")
        return {"success": True, "completion": record} 