"""
Delegate task tool for GLUE framework.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

from .tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType
from ..core.agent_loop import TeamMemberAgentLoop
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("glue.tools.delegate_task")

# PydanticAI schema for delegate_task arguments
class DelegateTaskArgs(BaseModel):
    target_agent: str = Field(..., min_length=1, description="Name of the agent to delegate the task to")
    task_description: str = Field(..., min_length=1, description="Detailed description of the task")
    parent_task_id: str = Field(..., min_length=1, description="ID of the parent task - sequential preferred")
    calling_team: str = Field(..., min_length=1, description="Name of the team invoking delegation")
    context_keys: List[str] = Field(default_factory=list, description="Optional context keys for background")
    required_artifacts: List[str] = Field(default_factory=list, description="Optional artifact keys required")

class DelegateTaskTool(Tool):
    """Core function for assigning work to team members."""
    def __init__(self, app=None):
        """
        Initialize delegate task tool.
        
        Args:
            app: Reference to the GLUE application instance.
        """
        self.app = app
        name = "delegate_task"
        description = "Core function for assigning work to team members. Needed for all team leads."
        # Default permissions and adhesives
        config = ToolConfig(
            timeout=30.0,
            max_retries=0,
            required_permissions={ToolPermission.WRITE},
            adhesive_types={AdhesiveType.GLUE},
        )
        super().__init__(name=name, description=description, config=config)
        # Define explicit inputs for LLM schema generation
        self.inputs = {
            "target_agent_id": {
                "type": str,
                "description": "ID of the agent to delegate the task to",
                "required": True
            },
            "task_description": {
                "type": str,
                "description": "Detailed description of the task",
                "required": True
            },
            "parent_task_id": {
                "type": str,
                "description": "ID of the parent task",
                "required": True
            },
            "context_keys": {
                "type": list,
                "description": "Optional context keys",
                "required": False,
                "default": []
            },
            "required_artifacts": {
                "type": list,
                "description": "Optional list of required artifacts",
                "required": False,
                "default": []
            }
        }

    async def _execute(
        self,
        target_agent_id: str,
        task_description: str,
        parent_task_id: str,
        context_keys: Optional[List[str]] = None,
        required_artifacts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delegate a task to a team member and persist it in the team's shared_results.
        """
        if not self.app:
            logger.error("DelegateTaskTool: Application context is missing.")
            return {"success": False, "error": "Application context is missing."}

        # Infer calling_agent_id and calling_team if not provided
        calling_agent_id = kwargs.get("calling_agent_id") or kwargs.get("calling_model")
        if not calling_agent_id:
            error_msg = "DelegateTaskTool: Missing calling_agent_id or calling_model in kwargs."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        if not kwargs.get("calling_team"):
            for team_name, team in self.app.teams.items():
                if calling_agent_id in getattr(team, 'models', {}):
                    kwargs["calling_team"] = team_name
                    break

        # PydanticAI validate incoming parameters
        try:
            args = DelegateTaskArgs(
                target_agent=target_agent_id,
                task_description=task_description,
                parent_task_id=parent_task_id,
                calling_team=kwargs.get("calling_team", ""),
                context_keys=context_keys or [],
                required_artifacts=required_artifacts or []
            )
        except ValidationError as e:
            logger.error(f"DelegateTaskTool: argument validation error: {e}")
            return {"success": False, "error": str(e)}

        # Map validated fields back
        target_agent_id = args.target_agent
        task_description = args.task_description
        parent_task_id = args.parent_task_id
        calling_team = args.calling_team
        context_keys = args.context_keys
        required_artifacts = args.required_artifacts

        # Ensure the target agent is a member of the team (allow case-insensitive match)
        if target_agent_id not in self.app.teams[calling_team].models:
            # Try case-insensitive lookup
            match = next((agent for agent in self.app.teams[calling_team].models if agent.lower() == target_agent_id.lower()), None)
            if match:
                logger.debug(f"DelegateTaskTool: Using case-insensitive match '{match}' for target_agent_id '{target_agent_id}'")
                target_agent_id = match
            else:
                error_msg = f"DelegateTaskTool: Agent '{target_agent_id}' not found in team '{calling_team}'."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

        # Generate a unique task ID
        team_slug = self.app.teams[calling_team].name.lower().replace(" ", "-")
        task_id = f"{team_slug}-task-{uuid.uuid4().hex[:8]}"

        # Build task object
        task = {
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "description": task_description,
            "assigned_to": target_agent_id,
            "context_keys": context_keys,
            "required_artifacts": required_artifacts,
            "created_by": calling_agent_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Persist the task in team.shared_results
        self.app.teams[calling_team].shared_results[task_id] = task

        # Notify the assigned agent via internal team queue so they receive the task
        internal_message = {
            "content": {"task_assigned": task},
            "metadata": {
                "source_team": self.app.teams[calling_team].name,
                "target_team": self.app.teams[calling_team].name,
                "source_model": calling_agent_id,
                "target_model": target_agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "internal": True
            }
        }
        await self.app.teams[calling_team].message_queue.put((internal_message, None))
        # Also directly notify the agent so they immediately process the assignment
        try:
            await self.app.teams[calling_team].direct_communication(calling_agent_id, target_agent_id, task)
            logger.debug(f"DelegateTaskTool: Notified agent '{target_agent_id}' directly of the assignment.")
        except Exception as e:
            logger.warning(f"DelegateTaskTool: direct_communication error: {e}")

        logger.info(f"Delegated task {task_id} to agent {target_agent_id} in team {calling_team}")
        return {"success": True, "task": task}
