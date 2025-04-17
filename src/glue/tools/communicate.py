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
        
    async def execute(self, target_type: str, target_name: str, message: str, **kwargs) -> Dict[str, Any]:
        """Execute the communication tool.
        
        Args:
            target_type: Type of target to communicate with (model or team)
            target_name: Name of the target model or team
            message: Message to send to the target
            **kwargs: Expected to contain 'calling_model' (str) and 'calling_team' (str)
            
        Returns:
            Dictionary with the result of the communication
        """
        # Get the calling model and team from kwargs
        calling_model_name = kwargs.get('calling_model')
        calling_team_name = kwargs.get('calling_team')
        
        # Retrieve app context if not already set (needed to find team/model objects)
        if not self.app:
            # This is a fallback, ideally app context should be set during tool initialization
            logger.warning("CommunicateTool: App context not set during init. Trying to get globally.")
            # This assumes a way to get the global app instance, which might not be reliable
            # Replace with a proper context injection mechanism if possible
            try:
                # Example: Accessing a hypothetical global app instance (replace if needed)
                from .globals import get_current_app # Hypothetical
                self.app = get_current_app() 
            except ImportError:
                logger.error("Could not get app context for CommunicateTool.")
                self.app = None # Ensure it remains None if import fails

        if not self.app:
             return {
                "success": False,
                "error": "CommunicateTool failed: Application context is missing."
             }

        # Validate context
        if not calling_model_name or not calling_team_name:
            logger.error(f"CommunicateTool failed: Missing calling_model ({calling_model_name}) or calling_team ({calling_team_name}) in kwargs.")
            return {
                "success": False,
                "error": "Could not determine calling model or team context from arguments."
            }
        
        # Find the actual team and model objects using the app context
        calling_team = self.app.teams.get(calling_team_name)
        if not calling_team:
            logger.error(f"CommunicateTool failed: Calling team '{calling_team_name}' not found in app.")
            return {"success": False, "error": f"Calling team '{calling_team_name}' not found."}
            
        calling_model = calling_team.models.get(calling_model_name)
        if not calling_model:
            logger.error(f"CommunicateTool failed: Calling model '{calling_model_name}' not found in team '{calling_team_name}'.")
            return {"success": False, "error": f"Calling model '{calling_model_name}' not found."}

        logger.info(f"Model {calling_model.name} in team {calling_team.name} is communicating with {target_type} {target_name}")
        
        if target_type == "model":
            # Communicate with another model
            response = await calling_model.communicate_with_model(target_name, message)
            
            if response:
                return {
                    "success": True,
                    "response": response
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to communicate with model {target_name}"
                }
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
                    try:
                        result = await asyncio.wait_for(future, timeout=timeout_seconds)
                        logger.info(f"Received result for broadcast {broadcast_id}")
                        return {
                           "success": True,
                           "message": "Broadcast processed.",
                           "responses": result # Return the aggregated responses
                        }
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for broadcast {broadcast_id} result.")
                        # Clean up the pending future on timeout is handled in finally
                        return {
                            "success": False,
                            "error": f"Timeout waiting for broadcast response (ID: {broadcast_id})"
                        }
                    except Exception as wait_e:
                         logger.error(f"Error waiting for broadcast {broadcast_id} result: {wait_e}")
                         return {"success": False, "error": f"Error waiting for broadcast: {wait_e}"}
                        
                except Exception as e:
                    logger.error(f"Error queueing internal message in team {calling_team.name}: {e}")
                    # Ensure future is removed if queueing fails
                    if broadcast_id in calling_team.pending_broadcasts:
                         calling_team.pending_broadcasts.pop(broadcast_id, None)
                    return {
                        "success": False,
                        "error": f"Error queueing internal message: {e}"
                    }
                finally:
                    # Ensure future is removed if it still exists (e.g., on timeout or error)
                    if broadcast_id in calling_team.pending_broadcasts:
                         calling_team.pending_broadcasts.pop(broadcast_id, None)
                         logger.debug(f"Cleaned up pending future {broadcast_id} in finally block.")
            else:
                # Communicate with another (different) team
                logger.debug(f"Initiating inter-team communication from {calling_team.name} to {target_name}")
                # The original logic used communicate_with_team, let's stick to that pathway
                # Ensure communicate_with_team exists on the model
                if hasattr(calling_model, 'communicate_with_team'):
                    response = await calling_model.communicate_with_team(target_name, message)
                    if response: # Assuming response indicates success/failure or carries data
                         # Check the type of response. communicate_with_team might return bool or dict
                         if isinstance(response, dict):
                             return response # Return the dict directly if it contains success/error
                         else:
                            # Assume boolean True means success
                            return {
                                "success": True,
                                "message": f"Communication initiated with team {target_name}."
                            }
                    else:
                        logger.warning(f"calling_model.communicate_with_team returned False/None for target {target_name}")
                        return {
                            "success": False,
                            "error": f"Failed to initiate communication with team {target_name}. Method returned failure."
                        }
                else:
                    logger.error(f"Model {calling_model.name} does not have method communicate_with_team")
                    return {
                       "success": False,
                       "error": f"Model {calling_model.name} cannot communicate with teams."
                    }
        else:
            return {
                "success": False,
                "error": f"Invalid target type: {target_type}"
            }
            
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the current execution context.
        
        Args:
            context: Dictionary with context information
        """
        self._current_context = context