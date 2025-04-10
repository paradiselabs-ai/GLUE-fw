"""
Communication tool for GLUE framework.

This module provides a tool for models to communicate with other models
and teams within the GLUE framework.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

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
        
    async def execute(self, target_type: str, target_name: str, message: str) -> Dict[str, Any]:
        """Execute the communication tool.
        
        Args:
            target_type: Type of target to communicate with (model or team)
            target_name: Name of the target model or team
            message: Message to send to the target
            
        Returns:
            Dictionary with the result of the communication
        """
        # Get the calling model and team
        calling_model = None
        calling_team = None
        
        # This will be set by the Team.process_message method
        if hasattr(self, '_current_context'):
            calling_model = self._current_context.get('model')
            calling_team = self._current_context.get('team')
        
        if not calling_model or not calling_team:
            return {
                "success": False,
                "error": "Could not determine calling model or team"
            }
            
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
                logger.debug(f"Handling intra-team communication within {calling_team.name} via message queue")
                # Construct message and put into team's queue
                internal_message = {
                    "content": message,
                    "metadata": {
                        "source_team": calling_team.name,
                        "target_team": calling_team.name,
                        "source_model": calling_model.name,
                        "timestamp": datetime.now().isoformat(),
                        "internal": True # Flag for internal message
                    }
                }
                try:
                    # Use the team object from context to access the queue
                    await calling_team.message_queue.put((internal_message, None)) # None sender for internal
                    logger.info(f"Queued internal message from {calling_model.name} within team {calling_team.name}")
                    return {
                        "success": True,
                        "message": "Internal message queued successfully."
                    }
                except Exception as e:
                    logger.error(f"Error queueing internal message in team {calling_team.name}: {e}")
                    return {
                        "success": False,
                        "error": f"Error queueing internal message: {e}"
                    }
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