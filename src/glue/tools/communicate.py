"""
Communication tool for GLUE framework.

This module provides a tool for models to communicate with other models
and teams within the GLUE framework.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import uuid

# Import necessary types
from ..core.types import ToolResult, AdhesiveType

logger = logging.getLogger("glue.tools.communicate")

class CommunicateTool:
    """Tool for model-to-model and model-to-team communication."""
    
    def __init__(self, app=None):
        """Initialize the communication tool.
        
        Args:
            app: Optional reference to the GLUE app
        """
        self.app = app
        self.name = "communicate"
        self.description = "Communicate with other models or teams in the GLUE framework"
        self.parameters = {
            "type": "object",
            "properties": {
                "target_type": {
                    "type": "string",
                    "enum": ["model", "team"],
                    "description": "Type of target to communicate with (model or team)"
                },
                "target_name": {
                    "type": "string",
                    "description": "Name of the target model or team"
                },
                "message": {
                    "type": "string",
                    "description": "Message to send to the target"
                }
            },
            "required": ["target_type", "target_name", "message"]
        }
        
    async def execute(self, target_type: str, target_name: str, message: str, **kwargs) -> ToolResult:
        """Execute the communication tool.
        
        Args:
            target_type: Type of target to communicate with (model or team)
            target_name: Name of the target model or team
            message: Message to send to the target
            **kwargs: Expected to contain 'calling_model' (str) and 'calling_team' (str)
            
        Returns:
            ToolResult object indicating success or failure of the communication attempt.
        """
        # Default adhesive for communication results
        result_adhesive = AdhesiveType.VELCRO

        # Get the calling model and team from kwargs
        calling_model_name = kwargs.get('calling_model')
        calling_team_name = kwargs.get('calling_team')
        
        # Retrieve app context if not already set (needed to find team/model objects)
        if not self.app:
            # This is a fallback, ideally app context should be set during tool initialization
            logger.warning("CommunicateTool: App context not set during init. Should be injected by GlueApp.")
            # Return ToolResult indicating error
            return ToolResult(
                status="error",
                error="CommunicateTool failed: Application context is missing.",
                adhesive=result_adhesive
            )

        # Validate context
        if not calling_model_name or not calling_team_name:
            logger.error(f"CommunicateTool failed: Missing calling_model ({calling_model_name}) or calling_team ({calling_team_name}) in kwargs.")
            # Return ToolResult indicating error
            return ToolResult(
                status="error",
                error="Could not determine calling model or team context from arguments.",
                adhesive=result_adhesive
            )
        
        # Find the actual team and model objects using the app context
        calling_team = self.app.teams.get(calling_team_name)
        if not calling_team:
            logger.error(f"CommunicateTool failed: Calling team '{calling_team_name}' not found in app.")
            return ToolResult(status="error", error=f"Calling team '{calling_team_name}' not found.", adhesive=result_adhesive)
            
        calling_model = calling_team.models.get(calling_model_name)
        if not calling_model:
            logger.error(f"CommunicateTool failed: Calling model '{calling_model_name}' not found in team '{calling_team_name}'.")
            return ToolResult(status="error", error=f"Calling model '{calling_model_name}' not found.", adhesive=result_adhesive)

        logger.info(f"Model {calling_model.name} in team {calling_team.name} is communicating with {target_type} {target_name}")
        
        try:
            if target_type == "model":
                # Communicate with another model
                response = await calling_model.communicate_with_model(target_name, message)
                
                if response:
                    # Return ToolResult indicating success
                    return ToolResult(
                        status="success",
                        output=f"Successfully initiated communication with model {target_name}. Response snippet: {str(response)[:50]}...",
                        adhesive=result_adhesive
                    )
                else:
                    # Return ToolResult indicating error
                    return ToolResult(
                        status="error",
                        error=f"Failed to communicate with model {target_name}",
                        adhesive=result_adhesive
                    )
            elif target_type == "team":
                # Check for intra-team communication
                if target_name == calling_team.name:
                    logger.debug(f"Initiating intra-team broadcast within {calling_team.name}")
                    broadcast_id = str(uuid.uuid4())
                    future = asyncio.get_running_loop().create_future()

                    # Ensure pending_broadcasts exists on the team
                    if not hasattr(calling_team, 'pending_broadcasts'):
                        calling_team.pending_broadcasts = {}
                    calling_team.pending_broadcasts[broadcast_id] = future

                    internal_message = {
                        "content": message,
                        "metadata": {
                            "source_team": calling_team.name,
                            "target_team": calling_team.name,
                            "source_model": calling_model.name,
                            "timestamp": datetime.now().isoformat(),
                            "internal": True, # Flag for internal message
                            "broadcast_id": broadcast_id # Add broadcast ID
                        }
                    }
                    try:
                        # Use the team object from context to access the queue
                        await calling_team.message_queue.put((internal_message, None)) # None sender for internal
                        logger.info(f"Queued internal broadcast message (ID: {broadcast_id}) from {calling_model.name} within team {calling_team.name}")
                        
                        # Wait for the broadcast to be processed
                        timeout_seconds = 30.0 # Configurable timeout
                        logger.debug(f"Waiting for broadcast {broadcast_id} result (timeout: {timeout_seconds}s)")
                        result = await asyncio.wait_for(future, timeout=timeout_seconds)
                        logger.info(f"Received result for broadcast {broadcast_id}")
                        # Return ToolResult indicating success
                        return ToolResult(
                           status="success",
                           output=f"Broadcast processed. Responses: {str(result)[:100]}...", # Summarize result
                           adhesive=result_adhesive
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for broadcast {broadcast_id} result.")
                        # Return ToolResult indicating error
                        return ToolResult(
                            status="error",
                            error=f"Timeout waiting for broadcast response (ID: {broadcast_id})",
                            adhesive=result_adhesive
                        )
                    except Exception as wait_e:
                         logger.error(f"Error waiting for broadcast {broadcast_id} result: {wait_e}")
                         return ToolResult(status="error", error=f"Error waiting for broadcast: {wait_e}", adhesive=result_adhesive)
                        
                    finally:
                        # Ensure future is removed if it still exists (e.g., on timeout or error)
                        if broadcast_id in calling_team.pending_broadcasts:
                             calling_team.pending_broadcasts.pop(broadcast_id, None)
                             logger.debug(f"Cleaned up pending future {broadcast_id} in finally block.")
                else:
                    # Communicate with another (different) team
                    logger.debug(f"Initiating inter-team communication from {calling_team.name} to {target_name}")
                    if hasattr(calling_model, 'communicate_with_team'):
                        success_flag = await calling_model.communicate_with_team(target_name, message)
                        if success_flag:
                            return ToolResult(
                                status="success",
                                output=f"Communication initiated with team {target_name}.",
                                adhesive=result_adhesive
                            )
                        else:
                            logger.warning(f"calling_model.communicate_with_team returned False/None for target {target_name}")
                            return ToolResult(
                                status="error",
                                error=f"Failed to initiate communication with team {target_name}. Method returned failure.",
                                adhesive=result_adhesive
                            )
                    else:
                        logger.error(f"Model {calling_model.name} does not have method communicate_with_team")
                        return ToolResult(
                           status="error",
                           error=f"Model {calling_model.name} cannot communicate with teams.",
                           adhesive=result_adhesive
                        )
            else:
                # Invalid target type - CORRECTLY INDENTED INSIDE TRY
                logger.warning(f"Invalid target type provided: {target_type}")
                return ToolResult(
                    status="error",
                    error=f"Invalid target type: {target_type}",
                    adhesive=result_adhesive
                )
        except Exception as main_e:
            # Catch any unexpected errors during execution - CORRECTLY INDENTED
            logger.exception(f"Unexpected error in CommunicateTool.execute: {main_e}")
            return ToolResult(
                status="error",
                error=f"Unexpected error during communication: {main_e}",
                adhesive=result_adhesive
            )
            
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the current execution context.
        
        Args:
            context: Dictionary with context information
        """
        self._current_context = context