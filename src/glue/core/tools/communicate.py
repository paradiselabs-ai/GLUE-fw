from datetime import datetime
import logging

# Assuming V1MessagePayload, MessageType, AdhesiveType are in types
from glue.core.types import V1MessagePayload, MessageType, AdhesiveType 

logger = logging.getLogger(__name__)

class Communicate:
    def __init__(self, app):
        self.app = app

    async def execute(self, target_type, target_name, message, **kwargs):
        """Execute the communicate tool.
        
        Args:
            target_type: Type of target (team or model)
            target_name: Name of the target
            message: Message to send
            **kwargs: Additional arguments
        """
        calling_model = kwargs.get('calling_model')
        calling_team = kwargs.get('calling_team')
        
        # Get the team object
        team = self.app.teams.get(calling_team)
        if not team:
            return {"error": f"Calling team {calling_team} not found"}
        
        # Handle team-to-team communication
        if target_type == "team":
            if target_name not in self.app.teams:
                return {
                    "error": f"Target team '{target_name}' not found. Available teams: {list(self.app.teams.keys())}",
                    "suggestion": f"If you meant to communicate with a model named '{target_name}', use target_type='model' instead."
                }
                
            # Try to send the message
            result = await team.send_information(target_name, message)
            
            # Check if result is a dictionary with error information
            if isinstance(result, dict) and not result.get("success", True):
                # Return the error information directly
                return {
                    "error": result.get("error", "Unknown error occurred"),
                    "suggestion": result.get("suggestion", "Check team configurations and flows"),
                    "status": "failed"
                }
            elif result is False:  # Handle old-style boolean returns
                return {
                    "error": f"Failed to send message to team {target_name}",
                    "suggestion": "Check if teams have a proper relationship established",
                    "status": "failed"
                }
            
            # Successfully sent message
            return {
                "status": "success",
                "message": f"Message sent to team {target_name}",
                "timestamp": datetime.now().isoformat()
            }

        # Handle model-to-model communication within the same team
        elif target_type == "model":
            # Check if target model/agent exists in the calling team
            target_is_lead = team.lead and team.lead.name == target_name
            target_is_agent = target_name in team.agents

            if not target_is_lead and not target_is_agent:
                agent_names = list(team.agents.keys())
                lead_name = team.lead.name if team.lead else None
                available_targets = agent_names + ([lead_name] if lead_name else [])
                return {
                    "error": f"Target model/agent '{target_name}' not found in team '{calling_team}'. Available: {available_targets}",
                    "suggestion": f"If you meant to communicate with another team named '{target_name}', use target_type='team'."
                }

            # --- Construct the V1 Payload ---
            # TODO: Need to ensure task_id is passed reliably via kwargs from AgentLoop
            task_id = kwargs.get('task_id', f"unknown_task_{datetime.now().isoformat()}")
            if task_id.startswith("unknown_task"):
                 logger.warning(f"Communicate tool executed without a proper task_id in kwargs. Using placeholder: {task_id}")

            try:
                payload = V1MessagePayload(
                    task_id=task_id,
                    sender_agent_id=calling_model,
                    sender_team_id=calling_team,
                    timestamp=datetime.now().isoformat(),
                    message_type=MessageType.DIRECT_MESSAGE,
                    adhesive_type=AdhesiveType.TAPE, # Communication itself is transient
                    content=message, # The actual message content
                    origin_tool_id=None # Not directly from a tool result
                )

                # Determine target loop ID
                if target_is_lead:
                    target_loop_id = f"{calling_team}-lead-{target_name}"
                else: # Must be an agent
                    target_loop_id = f"{calling_team}-agent-{target_name}"

                # Package for team queue
                # Use model_dump() if V1MessagePayload is Pydantic, or adapt if dataclass
                payload_dict = payload.__dict__ if hasattr(payload, '__dict__') else vars(payload) # Basic dataclass conversion
                # Ensure enums are converted to strings if necessary for JSON serialization later
                payload_dict['message_type'] = payload_dict['message_type'].value
                payload_dict['adhesive_type'] = payload_dict['adhesive_type'].value

                routed_message = {
                    "target_loop_id": target_loop_id,
                    "content": payload_dict
                }

                # Put message onto the team's queue for processing
                await team.message_queue.put(routed_message)

                logger.info(f"Queued direct message from {calling_model} to {target_name} in team {calling_team}.")
                return {
                    "status": "success",
                    "message": f"Message queued for model/agent {target_name} within team {calling_team}.",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                 logger.error(f"Error constructing or queuing direct message: {e}", exc_info=True)
                 return {
                     "error": f"Internal error processing direct message: {e}",
                     "status": "failed"
                 }

        # Handle invalid target_type
        else:
            return {
                "error": f"Invalid target_type: '{target_type}'. Must be 'team' or 'model'.",
                "status": "failed"
            }
