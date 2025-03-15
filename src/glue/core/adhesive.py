"""
Adhesive system for the GLUE framework.

This module implements the adhesive system, which controls how tool results
are bound and persisted within teams and models. There are three types of
adhesives:

1. GLUE: Team-wide persistent results
2. VELCRO: Session-based persistence for models
3. TAPE: One-time use with no persistence
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from glue.core.schemas import AdhesiveType, ToolResult

# Set up logging
logger = logging.getLogger("glue.core.adhesive")


class AdhesiveSystem:
    """
    System for managing tool result persistence with different adhesive types.
    
    The adhesive system stores tool results based on their adhesive type:
    - GLUE: Team-wide persistent storage, indexed by team name and tool call ID
    - VELCRO: Model-specific storage, indexed by model name and tool call ID
    - TAPE: Temporary storage, indexed by tool call ID (removed after retrieval)
    """
    
    def __init__(self):
        """Initialize the adhesive system with empty storage."""
        # GLUE storage: team_name -> {tool_call_id -> {model, result, timestamp}}
        self.glue_storage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # VELCRO storage: model_name -> {tool_call_id -> {team, result, timestamp}}
        self.velcro_storage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # TAPE storage: tool_call_id -> {team, model, result, timestamp}
        self.tape_storage: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Adhesive system initialized")
    
    def store_glue_result(self, team_name: str, model_name: str, tool_result: ToolResult) -> None:
        """
        Store a tool result with GLUE adhesive (team-wide persistence).
        
        Args:
            team_name: Name of the team using the tool
            model_name: Name of the model using the tool
            tool_result: Result from the tool execution
        """
        # Initialize team storage if it doesn't exist
        if team_name not in self.glue_storage:
            self.glue_storage[team_name] = {}
        
        # Store the result
        self.glue_storage[team_name][tool_result.tool_call_id] = {
            "model": model_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Stored GLUE result for team {team_name}, tool call {tool_result.tool_call_id}")
    
    def store_velcro_result(self, team_name: str, model_name: str, tool_result: ToolResult) -> None:
        """
        Store a tool result with VELCRO adhesive (model-level persistence).
        
        Args:
            team_name: Name of the team using the tool
            model_name: Name of the model using the tool
            tool_result: Result from the tool execution
        """
        # Initialize model storage if it doesn't exist
        if model_name not in self.velcro_storage:
            self.velcro_storage[model_name] = {}
        
        # Store the result
        self.velcro_storage[model_name][tool_result.tool_call_id] = {
            "team": team_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Stored VELCRO result for model {model_name}, tool call {tool_result.tool_call_id}")
    
    def store_tape_result(self, team_name: str, model_name: str, tool_result: ToolResult) -> None:
        """
        Store a tool result with TAPE adhesive (one-time use).
        
        Args:
            team_name: Name of the team using the tool
            model_name: Name of the model using the tool
            tool_result: Result from the tool execution
        """
        # Store the result
        self.tape_storage[tool_result.tool_call_id] = {
            "team": team_name,
            "model": model_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Stored TAPE result for tool call {tool_result.tool_call_id}")
    
    def get_tool_result(self, tool_call_id: str) -> Optional[ToolResult]:
        """
        Get a tool result by its ID, checking all storage types.
        
        For TAPE results, the result is removed after retrieval.
        
        Args:
            tool_call_id: ID of the tool call to retrieve
            
        Returns:
            The tool result if found, None otherwise
        """
        # Check TAPE storage first (and remove if found)
        if tool_call_id in self.tape_storage:
            result = self.tape_storage[tool_call_id]["result"]
            del self.tape_storage[tool_call_id]
            logger.debug(f"Retrieved and removed TAPE result for tool call {tool_call_id}")
            return result
        
        # Check GLUE storage
        for team_name, team_storage in self.glue_storage.items():
            if tool_call_id in team_storage:
                logger.debug(f"Retrieved GLUE result for tool call {tool_call_id} from team {team_name}")
                return team_storage[tool_call_id]["result"]
        
        # Check VELCRO storage
        for model_name, model_storage in self.velcro_storage.items():
            if tool_call_id in model_storage:
                logger.debug(f"Retrieved VELCRO result for tool call {tool_call_id} from model {model_name}")
                return model_storage[tool_call_id]["result"]
        
        logger.debug(f"No result found for tool call {tool_call_id}")
        return None
    
    def get_team_tool_results(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Get all tool results for a team (GLUE storage).
        
        Args:
            team_name: Name of the team
            
        Returns:
            List of tool results with metadata
        """
        if team_name not in self.glue_storage:
            return []
        
        return list(self.glue_storage[team_name].values())
    
    def get_model_tool_results(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all tool results for a model (VELCRO storage).
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of tool results with metadata
        """
        if model_name not in self.velcro_storage:
            return []
        
        return list(self.velcro_storage[model_name].values())
    
    def clear_team_storage(self, team_name: str) -> None:
        """
        Clear all tool results for a team (GLUE storage).
        
        Args:
            team_name: Name of the team
        """
        if team_name in self.glue_storage:
            del self.glue_storage[team_name]
            logger.info(f"Cleared GLUE storage for team {team_name}")
    
    def clear_model_storage(self, model_name: str) -> None:
        """
        Clear all tool results for a model (VELCRO storage).
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.velcro_storage:
            del self.velcro_storage[model_name]
            logger.info(f"Cleared VELCRO storage for model {model_name}")
    
    def clear_storage(self, adhesive_type: Optional[AdhesiveType] = None) -> None:
        """
        Clear storage for a specific adhesive type or all types.
        
        Args:
            adhesive_type: Type of adhesive to clear, or None for all types
        """
        if adhesive_type is None:
            self.glue_storage.clear()
            self.velcro_storage.clear()
            self.tape_storage.clear()
            logger.info("Cleared all adhesive storage")
        elif adhesive_type == AdhesiveType.GLUE:
            self.glue_storage.clear()
            logger.info("Cleared GLUE storage")
        elif adhesive_type == AdhesiveType.VELCRO:
            self.velcro_storage.clear()
            logger.info("Cleared VELCRO storage")
        elif adhesive_type == AdhesiveType.TAPE:
            self.tape_storage.clear()
            logger.info("Cleared TAPE storage")
    
    def reset(self) -> None:
        """Reset the adhesive system, clearing all storage."""
        self.clear_storage()
        logger.info("Adhesive system reset")


def bind_tool_result(
    system: AdhesiveSystem,
    team: Any,
    model: Any,
    tool_result: ToolResult,
    adhesive_type: AdhesiveType
) -> None:
    """
    Bind a tool result with the specified adhesive type.
    
    Args:
        system: The adhesive system to use
        team: The team using the tool
        model: The model using the tool
        tool_result: The result from the tool execution
        adhesive_type: The type of adhesive to use
    
    Raises:
        ValueError: If the model doesn't support the adhesive type
    """
    # Check if the model supports this adhesive type
    if not check_adhesive_compatibility(model, adhesive_type):
        raise ValueError(f"Model {model.name} does not support adhesive type {adhesive_type}")
    
    # Store the result based on the adhesive type
    if adhesive_type == AdhesiveType.GLUE:
        system.store_glue_result(team.name, model.name, tool_result)
    elif adhesive_type == AdhesiveType.VELCRO:
        system.store_velcro_result(team.name, model.name, tool_result)
    elif adhesive_type == AdhesiveType.TAPE:
        system.store_tape_result(team.name, model.name, tool_result)


def get_tool_result(
    system: AdhesiveSystem,
    tool_call_id: str
) -> Optional[ToolResult]:
    """
    Get a tool result by its ID.
    
    Args:
        system: The adhesive system to use
        tool_call_id: ID of the tool call to retrieve
        
    Returns:
        The tool result if found, None otherwise
    """
    return system.get_tool_result(tool_call_id)


def check_adhesive_compatibility(model: Any, adhesive_type: AdhesiveType) -> bool:
    """
    Check if a model supports a specific adhesive type.
    
    Args:
        model: The model to check
        adhesive_type: The adhesive type to check
        
    Returns:
        True if the model supports the adhesive type, False otherwise
    """
    return hasattr(model, "adhesives") and adhesive_type in model.adhesives
