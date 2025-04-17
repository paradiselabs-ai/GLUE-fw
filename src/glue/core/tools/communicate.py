from datetime import datetime

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
        
        # ... Rest of the method for model-to-model communication 