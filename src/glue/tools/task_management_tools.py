"""
Task Management Tools for GLUE framework.

Provides tools for assigning tasks to agents and reporting task updates back to leads.
"""

import logging
import json
from typing import Dict, Any, Optional
from enum import Enum

# Assume Message and Team types are accessible or define placeholders
# from glue.core.schemas import Message 
# from glue.core.teams import Team

logger = logging.getLogger("glue.tools.task_management")

class TaskStatus(Enum):
    STARTED = "started"
    SESSION_COMPLETE = "session_complete"
    FINAL_VALIDATED = "final_validated"
    ERROR = "error"

class AdhesiveType(Enum):
    TAPE = "tape"
    VELCRO = "velcro"
    GLUE = "glue"
    
    @classmethod
    def _missing_(cls, value):
        for item in cls:
            if item.value == value.lower():
                return item
        return None # Or raise ValueError

class AssignTaskTool:
    """Tool for Team Leads to assign tasks to Agents within the same team."""

    def __init__(self, app=None):
        self.app = app
        self.name = "assign_task"
        self.description = "Assign a sub-task to a specific agent within your team."
        self.parameters = {
            "type": "object",
            "properties": {
                "target_agent_name": {
                    "type": "string",
                    "description": "Name of the agent model within your team to assign the task to."
                },
                "task_id": {
                    "type": "string",
                    "description": "A unique identifier for this specific sub-task."
                },
                "adhesive": {
                    "type": "string",
                    "enum": [e.value for e in AdhesiveType],
                    "description": "Adhesive policy for the task output (tape, velcro, or glue)."
                },
                "payload": {
                    "type": "object",
                    "description": "A dictionary containing task details (e.g., 'description', 'tool_hint', 'parameters').",
                    "properties": {
                         "description": {"type": "string", "description": "Detailed description of the task."},
                         # Add other expected payload fields as needed
                    },
                     "required": ["description"]
                }
            },
            "required": ["target_agent_name", "task_id", "adhesive", "payload"]
        }

    async def execute(self, target_agent_name: str, task_id: str, adhesive: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        calling_model_name = kwargs.get('calling_model')
        calling_team_name = kwargs.get('calling_team')

        if not self.app:
            logger.error(f"{self.name}: Cannot execute - Application context missing.")
            return {"success": False, "error": "Application context missing."}
        if not calling_model_name or not calling_team_name:
            logger.error(f"{self.name}: Cannot execute - Missing calling model/team context.")
            return {"success": False, "error": "Missing calling model/team context."}

        # Find the calling team
        calling_team = self.app.teams.get(calling_team_name)
        if not calling_team:
            logger.error(f"{self.name}: Calling team '{calling_team_name}' not found.")
            return {"success": False, "error": f"Calling team '{calling_team_name}' not found."}

        # Verify the calling model is the lead (optional but good practice)
        if not hasattr(calling_team, 'lead') or not calling_team.lead or calling_team.lead.name != calling_model_name:
             logger.warning(f"{self.name}: Called by non-lead model '{calling_model_name}' in team '{calling_team_name}'. Allowing for flexibility, but intended for leads.")
             # Decide if this should be an error:
             # return {"success": False, "error": "Only the team lead can assign tasks."}

        # Verify target agent exists in the same team
        if target_agent_name not in calling_team.models:
            logger.error(f"{self.name}: Target agent '{target_agent_name}' not found in team '{calling_team_name}'. Available: {list(calling_team.models.keys())}")
            return {"success": False, "error": f"Target agent '{target_agent_name}' not found in team '{calling_team_name}'."}

        # Validate adhesive type
        valid_adhesive = AdhesiveType(adhesive)
        if valid_adhesive is None:
             logger.error(f"{self.name}: Invalid adhesive type '{adhesive}'. Must be one of {[e.value for e in AdhesiveType]}.")
             return {"success": False, "error": f"Invalid adhesive type '{adhesive}'."}

        # Format the internal message for the team's queue
        internal_message = {
            # Content can be structured or use the payload directly
            "content": { 
                "type": "task_assignment",
                "task_id": task_id,
                "adhesive": valid_adhesive.value,
                "task_details": payload 
            },
            "metadata": {
                "message_type": "task_assignment", # Explicit type
                "source_team": calling_team_name,
                "target_team": calling_team_name, # Internal message
                "source_model": calling_model_name,
                "target_model": target_agent_name, # Specifically target the agent
                "internal": True
            }
        }

        try:
            if hasattr(calling_team, 'message_queue') and calling_team.message_queue is not None:
                await calling_team.message_queue.put((internal_message, None)) # Sender is None for internal
                logger.info(f"Task '{task_id}' assigned to agent '{target_agent_name}' by lead '{calling_model_name}' queued.")
                return {"success": True, "message": f"Task {task_id} assigned to {target_agent_name}."}
            else:
                logger.error(f"{self.name}: Team '{calling_team_name}' message queue not found.")
                return {"success": False, "error": f"Team '{calling_team_name}' message queue unavailable."}
        except Exception as e:
            logger.error(f"{self.name}: Error queueing task assignment for agent '{target_agent_name}': {e}")
            return {"success": False, "error": f"Error queueing task assignment: {e}"}


class ReportTaskUpdateTool:
    """Tool for Agents to report task status and results back to their Team Lead."""

    def __init__(self, app=None):
        self.app = app
        self.name = "report_task_update"
        self.description = "Report the status or results of an assigned task back to your Team Lead."
        self.parameters = {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The unique identifier of the task you are reporting on."
                },
                "status": {
                    "type": "string",
                    "enum": [e.value for e in TaskStatus],
                    "description": "Current status of the task (started, session_complete, final_validated, error)."
                },
                "payload": {
                    "type": "object",
                    "description": "A dictionary containing results, artifacts, or error details relevant to the status."
                }
            },
            "required": ["task_id", "status", "payload"]
        }

    async def execute(self, task_id: str, status: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        calling_model_name = kwargs.get('calling_model')
        calling_team_name = kwargs.get('calling_team')

        if not self.app:
            logger.error(f"{self.name}: Cannot execute - Application context missing.")
            return {"success": False, "error": "Application context missing."}
        if not calling_model_name or not calling_team_name:
             logger.error(f"{self.name}: Cannot execute - Missing calling model/team context.")
             return {"success": False, "error": "Missing calling model/team context."}

        # Find the calling team
        calling_team = self.app.teams.get(calling_team_name)
        if not calling_team:
            logger.error(f"{self.name}: Calling team '{calling_team_name}' not found.")
            return {"success": False, "error": f"Calling team '{calling_team_name}' not found."}
            
        # Find the lead model of the team
        lead_model_name = getattr(calling_team.config, 'lead', None)
        if not lead_model_name:
             logger.error(f"{self.name}: Could not find lead model for team '{calling_team_name}'.")
             return {"success": False, "error": f"Lead model not configured for team '{calling_team_name}'."}

        # Validate status
        valid_status = TaskStatus(status)
        if valid_status is None:
             logger.error(f"{self.name}: Invalid status '{status}'. Must be one of {[e.value for e in TaskStatus]}.")
             return {"success": False, "error": f"Invalid status '{status}'."}
             
        # Format the internal message for the team's queue
        internal_message = {
             # Content can be structured or use the payload directly
            "content": {
                "type": "task_update",
                "task_id": task_id,
                "status": valid_status.value,
                "update_details": payload
            },
            "metadata": {
                "message_type": "task_update", # Explicit type
                "source_team": calling_team_name,
                "target_team": calling_team_name, # Internal message
                "source_model": calling_model_name, # The agent reporting
                "target_model": lead_model_name, # Specifically target the lead
                "internal": True
            }
        }

        try:
            if hasattr(calling_team, 'message_queue') and calling_team.message_queue is not None:
                await calling_team.message_queue.put((internal_message, None)) # Sender is None for internal
                logger.info(f"Task update for '{task_id}' (status: {status}) reported by agent '{calling_model_name}' to lead '{lead_model_name}' queued.")
                return {"success": True, "message": f"Task update {task_id} (status: {status}) reported to lead."}
            else:
                logger.error(f"{self.name}: Team '{calling_team_name}' message queue not found.")
                return {"success": False, "error": f"Team '{calling_team_name}' message queue unavailable."}
        except Exception as e:
            logger.error(f"{self.name}: Error queueing task update for task '{task_id}': {e}")
            return {"success": False, "error": f"Error queueing task update: {e}"} 