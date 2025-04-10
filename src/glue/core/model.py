"""
Model module for the GLUE framework.

This module provides the Model class, which is a concrete implementation of the BaseModel
that can be used in teams.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Type, AsyncIterable, Set
import inspect
from datetime import datetime
import functools
import asyncio

from .base_model import BaseModel, ModelProvider
from .providers import ProviderBase
from .schemas import Message, ToolCall, ToolResult, ModelConfig
from .types import AdhesiveType

# Set up logging
logger = logging.getLogger("glue.model")


class Model(BaseModel):
    """Concrete implementation of the model class that can be used in teams."""
    
    def __init__(self, config=None, **kwargs):
        """Initialize a new model.
        
        Args:
            config: Model configuration
            **kwargs: Additional keyword arguments
        """
        # Handle config conversion
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            # Attempt conversion if possible, otherwise initialize empty
            try:
                config = vars(config)
            except TypeError:
                 # Handle cases where config might be a simple type or non-dict-like object
                 print(f"Warning: Model config is not a dictionary or easily convertible: {type(config)}. Initializing provider might fail.")
                 config = {}

        # Ensure name is always set
        if 'name' not in config and 'name' in kwargs:
            config['name'] = kwargs['name']
            
        # Initialize the base model
        super().__init__(config)
        
        # Set up adhesives for compatibility
        self.adhesives = set()
        adhesives = config.get('adhesives', [])
        logger.debug(f"Model {config.get('name', 'unnamed')}: Raw adhesives from config: {adhesives}")
        
        for adhesive in adhesives:
            # Convert string adhesives to AdhesiveType enum values
            if isinstance(adhesive, str):
                try:
                    # Try to get the enum value by name (case-insensitive)
                    adhesive_type = next(
                        (at for at in AdhesiveType if at.value.lower() == adhesive.lower()),
                        None
                    )
                    if adhesive_type:
                        self.adhesives.add(adhesive_type)
                        logger.debug(f"Added adhesive {adhesive_type} from string '{adhesive}'")
                    else:
                        logger.warning(f"Unknown adhesive type: {adhesive}")
                except Exception as e:
                    logger.warning(f"Error converting adhesive '{adhesive}' to enum: {e}")
            else:
                # Assume it's already an AdhesiveType enum value
                self.adhesives.add(adhesive)
                logger.debug(f"Added adhesive {adhesive}")
            
        # Add GLUE adhesive by default if no adhesives specified
        if not self.adhesives:
            self.adhesives.add(AdhesiveType.GLUE)
            logger.debug("No adhesives specified, added default GLUE adhesive")
        
        logger.info(f"Model {config.get('name', 'unnamed')} initialized with adhesives: {self.adhesives}")
    
    def has_adhesive(self, adhesive: AdhesiveType) -> bool:
        """Check if this model supports the given adhesive type.
        
        Args:
            adhesive: Adhesive type to check
            
        Returns:
            True if the model supports the adhesive, False otherwise
        """
        return adhesive in self.adhesives
    
    def add_tool_sync(self, name: str, tool: Any) -> None:
        """Add a tool to the model synchronously.
        
        This is a synchronous version of add_tool for use during setup.
        
        Args:
            name: Name of the tool
            tool: Tool to add
        """
        # Add to tools dictionary
        if not hasattr(self, 'tools'):
            self.tools = {}
            
        # Skip if tool already exists
        if name in self.tools:
            return
            
        self.tools[name] = tool
        
        # Format tool for provider if needed
        try:
            formatted_tool = self._format_tool_for_provider(name, tool)
            
            # Add to provider tools
            if hasattr(self, '_provider_tools'):
                self._provider_tools[name] = formatted_tool
        except Exception as e:
            logger.warning(f"Error formatting tool {name} for model {self.name}: {e}")
            
        logger.debug(f"Added tool {name} to model {self.name} synchronously")

    def _format_tool_for_provider(self, name: str, tool: Any) -> Dict[str, Any]:
        """Format a tool for the provider.
        
        Args:
            name: Name of the tool
            tool: Tool to format
            
        Returns:
            Formatted tool
        """
        # Default format for tools
        formatted_tool = {
            "name": name,
            "description": getattr(tool, "description", f"Tool: {name}")
        }
        
        # Add parameters if available
        if hasattr(tool, "parameters"):
            formatted_tool["parameters"] = tool.parameters
            # Ensure 'type' is set for Gemini compatibility
            if "type" not in formatted_tool["parameters"]:
                formatted_tool["parameters"]["type"] = "object"
        elif hasattr(tool, "get_parameters") and callable(tool.get_parameters):
            try:
                formatted_tool["parameters"] = tool.get_parameters()
                # Ensure 'type' is set for Gemini compatibility
                if "type" not in formatted_tool["parameters"]:
                    formatted_tool["parameters"]["type"] = "object"
            except Exception as e:
                logger.warning(f"Error getting parameters for tool {name}: {e}")
                formatted_tool["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
        else:
            # Default parameters structure
            formatted_tool["parameters"] = {
                "type": "object",  # Required by Gemini API
                "properties": {},
                "required": []
            }
            
            # Try to infer parameters from the execute method if available
            if hasattr(tool, "execute") and callable(tool.execute):
                try:
                    sig = inspect.signature(tool.execute)
                    for param_name, param in sig.parameters.items():
                        # Skip self and kwargs
                        if param_name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
                            continue
                            
                        # Add parameter to properties
                        formatted_tool["parameters"]["properties"][param_name] = {
                            "type": "string",  # Default to string
                            "description": f"Parameter: {param_name}"
                        }
                        
                        # Add to required list if no default value
                        if param.default == inspect.Parameter.empty:
                            formatted_tool["parameters"]["required"].append(param_name)
                except Exception as e:
                    logger.warning(f"Error inferring parameters for tool {name}: {e}")
        
        return formatted_tool

    async def setup(self) -> None:
        """Set up the model by initializing any required resources.
        
        This is a placeholder implementation for test compatibility.
        In a real implementation, this would initialize any required resources.
        """
        # Nothing to do in the base implementation
        pass
        
    async def communicate_with_model(self, target_model_name: str, message: str) -> Optional[str]:
        """Communicate with another model.
        
        This method allows a model to communicate directly with another model,
        either within the same team or in a different team.
        
        Args:
            target_model_name: Name of the target model
            message: Message to send
            
        Returns:
            Response from the target model, or None if communication failed
        """
        logger.info(f"Model {self.name} attempting to communicate with model {target_model_name}")
        
        if not hasattr(self, 'team') or self.team is None:
            logger.warning(f"Model {self.name} is not part of a team, cannot communicate")
            return None
            
        logger.debug(f"Model {self.name} is in team {self.team.name}")
        logger.debug(f"Team {self.team.name} has models: {list(self.team.models.keys())}")
        
        # Check if target model is in the same team
        if target_model_name in self.team.models:
            logger.info(f"Target model {target_model_name} found in the same team {self.team.name}")
            
            # Direct communication within the same team
            target_model = self.team.models[target_model_name]
            
            # Create a message for the target model
            model_message = Message(
                role="system",  # Changed from "model" to "system" to match allowed roles
                content=f"Message from model {self.name}: {message}"
            )
            
            # Generate a response
            try:
                logger.debug(f"Generating response from target model {target_model_name}")
                response = await target_model.generate_response([model_message])
                logger.debug(f"Response generated: {response[:100]}...")
                
                # Store in team's conversation history
                self.team.conversation_history.append(Message(
                    role="system",  # Changed from "model" to "system" to match allowed roles
                    content=f"From {self.name} to {target_model_name}: {message}"
                ))
                
                self.team.conversation_history.append(Message(
                    role="assistant",  # Changed from "model" to "assistant" to match allowed roles
                    content=f"From {target_model_name} to {self.name}: {response}"
                ))
                
                logger.info(f"Communication successful between {self.name} and {target_model_name}")
                return response
            except Exception as e:
                logger.error(f"Error communicating with model {target_model_name}: {e}")
                return f"Error communicating with model {target_model_name}: {str(e)}"
        else:
            logger.info(f"Target model {target_model_name} not found in team {self.team.name}, checking other teams")
            
            # Target model is in a different team, need to use team communication
            # Find the team that contains the target model
            if not hasattr(self.team, 'outgoing_flows') or not self.team.outgoing_flows:
                logger.warning(f"Team {self.team.name} has no outgoing flows")
                return f"Team {self.team.name} has no outgoing flows to reach model {target_model_name}"
                
            logger.debug(f"Team {self.team.name} has outgoing flows to: {[flow.target.name for flow in self.team.outgoing_flows]}")
            
            for flow in self.team.outgoing_flows:
                target_team = flow.target
                logger.debug(f"Checking team {target_team.name} for model {target_model_name}")
                logger.debug(f"Team {target_team.name} has models: {list(target_team.models.keys())}")
                
                if target_model_name in target_team.models:
                    logger.info(f"Found target model {target_model_name} in team {target_team.name}")
                    
                    # Create message with specific target model
                    message_content = {
                        "content": message,
                        "metadata": {
                            "source_team": self.team.name,
                            "target_team": target_team.name,
                            "source_model": self.name,
                            "target_model": target_model_name,
                            "timestamp": datetime.now().isoformat(),
                            "initiated": True
                        }
                    }
                    
                    # Send message to target team
                    try:
                        logger.debug(f"Sending message to team {target_team.name}")
                        success = await self.team.send_information(target_team.name, message_content)
                        if success:
                            logger.info(f"Model {self.name} sent message to model {target_model_name} in team {target_team.name}")
                            
                            # Simple one-way communication
                            # Add target model to metadata
                            message_content["metadata"]["target_model"] = target_model_name
                            
                            # Return a simple acknowledgment
                            return f"Message sent to model {target_model_name} in team {target_team.name}. Check team {target_team.name}'s conversation history for any responses."
                        else:
                            logger.warning(f"Failed to send message to model {target_model_name} in team {target_team.name}")
                            return f"Failed to send message to model {target_model_name} in team {target_team.name}"
                    except Exception as e:
                        logger.error(f"Error sending message to team {target_team.name}: {e}")
                        return f"Error sending message to team {target_team.name}: {str(e)}"
                        
            logger.warning(f"Could not find model {target_model_name} in any connected team")
            return f"Could not find model {target_model_name} in any connected team"
            
    async def _sleep(self, seconds: float):
        """Sleep for the specified number of seconds without using asyncio.
        
        Args:
            seconds: Number of seconds to sleep
        """
        # Import time locally to avoid any module loading issues
        import time
        
        # Get the current time
        start_time = time.time()
        
        # Sleep in small increments to allow other tasks to run
        while time.time() - start_time < seconds:
            # Sleep for a small amount of time
            time.sleep(0.1)
            
            # Yield control to the event loop
            await self._yield_control()
    
    async def _yield_control(self):
        """Yield control to the event loop without using asyncio.sleep."""
        # This is a no-op that allows the event loop to run other tasks
        return
        
    async def communicate_with_team(self, target_team_name: str, message: str) -> Optional[str]:
        """Communicate with another team.
        
        This method allows a model to communicate with another team.
        
        Args:
            target_team_name: Name of the target team
            message: Message to send
            
        Returns:
            Indication of success, or None if communication failed
        """
        if not hasattr(self, 'team') or self.team is None:
            logger.warning(f"Model {self.name} is not part of a team, cannot communicate")
            return None

        # DEBUG: Log the names being compared
        logger.debug(f"[COMMUNICATE_WITH_TEAM CHECK] Comparing target='{target_team_name}' with self.team.name='{self.team.name if hasattr(self.team, 'name') else 'None'}'")

        # Handle intra-team communication by queueing the message
        if target_team_name == self.team.name:
            logger.debug(f"Handling intra-team communication within {self.team.name} from model {self.name}")
            internal_message = {
                "content": message,
                "metadata": {
                    "source_team": self.team.name,
                    "target_team": self.team.name,
                    "source_model": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "internal": True # Flag for internal message
                }
            }
            try:
                # Ensure message_queue exists and is usable
                if hasattr(self.team, 'message_queue') and self.team.message_queue is not None:
                    await self.team.message_queue.put((internal_message, None))
                    logger.info(f"Queued internal message from {self.name} within team {self.team.name}")
                    return "Internal message queued successfully."
                else:
                    logger.error(f"Team {self.team.name} does not have a message queue for internal communication.")
                    return "Error: Team message queue not available."
            except Exception as e:
                logger.error(f"Error queueing internal message in team {self.team.name} from model {self.name}: {e}")
                return f"Error queueing internal message: {e}"
            
        # --- Inter-Team Communication Logic ---
        # Create message for inter-team communication
        message_content = {
            "content": message,
            "metadata": {
                "source_team": self.team.name,
                "target_team": target_team_name,
                "source_model": self.name,
                "timestamp": datetime.now().isoformat(),
                "initiated": True
            }
        }
        
        # Send message to target team
        # Ensure send_information exists on the team object
        if hasattr(self.team, 'send_information') and callable(self.team.send_information):
            success = await self.team.send_information(target_team_name, message_content)
            if success:
                logger.info(f"Model {self.name} sent message to team {target_team_name}")
                return f"Message sent to team {target_team_name}. Check team {target_team_name}'s conversation history for any responses."
            else:
                logger.warning(f"Failed to send message to team {target_team_name} via send_information")
                return f"Failed to send message to team {target_team_name}"
        else:
             logger.error(f"Team {self.team.name} does not have send_information method.")
             return "Error: Team cannot send information."
