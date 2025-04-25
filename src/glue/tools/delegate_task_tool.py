"""
Delegate task tool for GLUE framework.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from .tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType

logger = logging.getLogger("glue.tools.delegate_task")

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

        calling_agent_id = kwargs.get("calling_agent_id") or kwargs.get("calling_model")
        calling_team = kwargs.get("calling_team")
        if not calling_agent_id or not calling_team:
            error_msg = "DelegateTaskTool: Missing calling_agent_id or calling_team in kwargs."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        team = self.app.teams.get(calling_team)
        if not team:
            error_msg = f"DelegateTaskTool: Team '{calling_team}' not found."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Ensure the target agent is a member of the team (allow case-insensitive match)
        if target_agent_id not in team.models:
            # Try case-insensitive lookup
            match = next((agent for agent in team.models if agent.lower() == target_agent_id.lower()), None)
            if match:
                logger.debug(f"DelegateTaskTool: Using case-insensitive match '{match}' for target_agent_id '{target_agent_id}'")
                target_agent_id = match
            else:
                error_msg = f"DelegateTaskTool: Agent '{target_agent_id}' not found in team '{team.name}'."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

        # Generate a unique task ID
        team_slug = team.name.lower().replace(" ", "-")
        task_id = f"{team_slug}-task-{uuid.uuid4().hex[:8]}"

        # Build task object
        task = {
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "description": task_description,
            "assigned_to": target_agent_id,
            "context_keys": context_keys or [],
            "required_artifacts": required_artifacts or [],
            "created_by": calling_agent_id,
            "created_at": datetime.utcnow().isoformat()
        }

        # Persist the task in team.shared_results
        team.shared_results[task_id] = task

        # Notify the assigned agent via internal team queue so they receive the task
        internal_message = {
            "content": {"task_assigned": task},
            "metadata": {
                "source_team": team.name,
                "target_team": team.name,
                "source_model": calling_agent_id,
                "target_model": target_agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "internal": True
            }
        }
        await team.message_queue.put((internal_message, None))
        # Also directly notify the agent so they immediately process the assignment
        try:
            await team.direct_communication(calling_agent_id, target_agent_id, task)
            logger.debug(f"DelegateTaskTool: Notified agent '{target_agent_id}' directly of the assignment.")
        except Exception as e:
            logger.warning(f"DelegateTaskTool: direct_communication error: {e}")

        logger.info(f"Delegated task {task_id} to agent {target_agent_id} in team {team.name}")
        return {"success": True, "task": task}
