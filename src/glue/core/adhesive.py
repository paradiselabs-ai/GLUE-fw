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
from typing import Dict, Any, List, Optional

from glue.core.simple_schemas import AdhesiveType, ToolResult

# Set up logging
logger = logging.getLogger(__name__)


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
        # GLUE storage: team_name -> {tool_name -> {model, result, timestamp}}
        self.glue_storage: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # VELCRO storage: model_name -> {tool_name -> {team, result, timestamp}}
        self.velcro_storage: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # TAPE storage: tool_name -> {team, model, result, timestamp}
        self.tape_storage: Dict[str, Dict[str, Any]] = {}

        logger.info("Adhesive system initialized")

    def store_glue_result(
        self, team_name: str, model_name: str, tool_result: ToolResult
    ) -> None:
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

        # For backward compatibility with tests
        tool_name = (
            tool_result.tool_name if tool_result.tool_name else tool_result.tool_call_id
        )

        # Store the result
        self.glue_storage[team_name][tool_name] = {
            "model": model_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug(f"Stored GLUE result for team {team_name}, tool {tool_name}")

    def store_velcro_result(
        self, team_name: str, model_name: str, tool_result: ToolResult
    ) -> None:
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

        # For backward compatibility with tests
        tool_name = (
            tool_result.tool_name if tool_result.tool_name else tool_result.tool_call_id
        )

        # Store the result
        self.velcro_storage[model_name][tool_name] = {
            "team": team_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug(f"Stored VELCRO result for model {model_name}, tool {tool_name}")

    def store_tape_result(
        self, team_name: str, model_name: str, tool_result: ToolResult
    ) -> None:
        """
        Store a tool result with TAPE adhesive (one-time use).

        Args:
            team_name: Name of the team using the tool
            model_name: Name of the model using the tool
            tool_result: Result from the tool execution
        """
        # For backward compatibility with tests
        tool_name = (
            tool_result.tool_name if tool_result.tool_name else tool_result.tool_call_id
        )

        # Store the result
        self.tape_storage[tool_name] = {
            "team": team_name,
            "model": model_name,
            "result": tool_result,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug(f"Stored TAPE result for tool {tool_name}")

    def get_tool_result(self, tool_name_or_id: str) -> Optional[ToolResult]:
        """
        Get a tool result by its name or ID.

        Args:
            tool_name_or_id: Name or ID of the tool to retrieve

        Returns:
            The tool result if found, None otherwise
        """
        # Check GLUE storage first (most persistent)
        for team_storage in self.glue_storage.values():
            if tool_name_or_id in team_storage:
                result = team_storage[tool_name_or_id]["result"]
                logger.debug(f"Retrieved GLUE result for tool {tool_name_or_id}")
                return result

        # Check VELCRO storage next
        for model_storage in self.velcro_storage.values():
            if tool_name_or_id in model_storage:
                result = model_storage[tool_name_or_id]["result"]
                logger.debug(f"Retrieved VELCRO result for tool {tool_name_or_id}")
                return result

        # Check TAPE storage last (least persistent)
        if tool_name_or_id in self.tape_storage:
            result = self.tape_storage[tool_name_or_id]["result"]
            # Remove from storage after retrieval (one-time use)
            del self.tape_storage[tool_name_or_id]
            logger.debug(
                f"Retrieved and removed TAPE result for tool {tool_name_or_id}"
            )
            return result

        logger.debug(f"No result found for tool {tool_name_or_id}")
        return None

    def clear_glue_storage(self) -> None:
        """Clear all GLUE storage."""
        self.glue_storage.clear()
        logger.info("Cleared GLUE storage")

    def clear_velcro_storage(self) -> None:
        """Clear all VELCRO storage."""
        self.velcro_storage.clear()
        logger.info("Cleared VELCRO storage")

    def clear_tape_storage(self) -> None:
        """Clear all TAPE storage."""
        self.tape_storage.clear()
        logger.info("Cleared TAPE storage")

    def clear_all_storage(self) -> None:
        """Clear all adhesive storage."""
        self.clear_glue_storage()
        self.clear_velcro_storage()
        self.clear_tape_storage()
        logger.info("Cleared all adhesive storage")

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

        return [
            {
                "tool_name": tool_name,
                "model": data["model"],
                "result": data["result"],
                "timestamp": data["timestamp"],
            }
            for tool_name, data in self.glue_storage[team_name].items()
        ]

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

        return [
            {
                "tool_name": tool_name,
                "team": data["team"],
                "result": data["result"],
                "timestamp": data["timestamp"],
            }
            for tool_name, data in self.velcro_storage[model_name].items()
        ]

    def clear_storage(self, adhesive_type: Optional[AdhesiveType] = None) -> None:
        """
        Clear storage for a specific adhesive type or all types.

        Args:
            adhesive_type: Type of adhesive to clear, or None for all types
        """
        if adhesive_type is None:
            self.clear_all_storage()
        elif adhesive_type == AdhesiveType.GLUE:
            self.clear_glue_storage()
        elif adhesive_type == AdhesiveType.VELCRO:
            self.clear_velcro_storage()
        elif adhesive_type == AdhesiveType.TAPE:
            self.clear_tape_storage()

    def clear_team_storage(self, team_name: str) -> None:
        """
        Clear GLUE storage for a specific team.

        Args:
            team_name: Name of the team to clear storage for
        """
        if team_name in self.glue_storage:
            del self.glue_storage[team_name]
            logger.info(f"Cleared GLUE storage for team {team_name}")

    def clear_model_storage(self, model_name: str) -> None:
        """
        Clear VELCRO storage for a specific model.

        Args:
            model_name: Name of the model to clear storage for
        """
        if model_name in self.velcro_storage:
            del self.velcro_storage[model_name]
            logger.info(f"Cleared VELCRO storage for model {model_name}")

    def reset(self) -> None:
        """Reset the adhesive system to its initial state."""
        self.glue_storage.clear()
        self.velcro_storage.clear()
        self.tape_storage.clear()
        logger.info("Adhesive system reset")


# AdhesiveManager is an alias for AdhesiveSystem to maintain backward compatibility with tests
AdhesiveManager = AdhesiveSystem


def bind_tool_result(
    system: AdhesiveSystem = None,
    team: Any = None,
    model: Any = None,
    tool_result: ToolResult = None,
    adhesive_type: AdhesiveType = None,
    # For backward compatibility with tests
    manager: Optional[AdhesiveSystem] = None,
):
    """
    Bind a tool result with the specified adhesive type.

    Args:
        system: The adhesive system to use
        team: The team using the tool
        model: The model using the tool
        tool_result: The result from the tool execution
        adhesive_type: The type of adhesive to use
        manager: Alias for system (backward compatibility)

    Raises:
        ValueError: If the model doesn't support the adhesive type
    """
    # Handle backward compatibility
    if manager is not None and system is None:
        system = manager

    if not check_adhesive_compatibility(model, adhesive_type):
        raise ValueError(
            f"Model {model.name} does not support adhesive type {adhesive_type}"
        )

    if adhesive_type == AdhesiveType.GLUE:
        system.store_glue_result(team.name, model.name, tool_result)
    elif adhesive_type == AdhesiveType.VELCRO:
        system.store_velcro_result(team.name, model.name, tool_result)
    elif adhesive_type == AdhesiveType.TAPE:
        system.store_tape_result(team.name, model.name, tool_result)
    else:
        raise ValueError(f"Unknown adhesive type: {adhesive_type}")

    logger.debug(
        f"Bound tool result {tool_result.tool_call_id} with {adhesive_type} adhesive"
    )


def get_tool_result(
    system: AdhesiveSystem = None,
    tool_call_id: str = None,
    # For backward compatibility with tests
    manager: Optional[AdhesiveSystem] = None,
    tool_name: Optional[str] = None,
) -> Optional[ToolResult]:
    """
    Get a tool result by its ID or name.

    Args:
        system: The adhesive system to use
        tool_call_id: ID of the tool call to retrieve
        manager: Alias for system (backward compatibility)
        tool_name: Name of the tool to retrieve (backward compatibility)

    Returns:
        The tool result if found, None otherwise
    """
    # Handle backward compatibility
    if manager is not None and system is None:
        system = manager

    # Use tool_name if provided (for backward compatibility)
    lookup_key = tool_name if tool_name is not None else tool_call_id

    return system.get_tool_result(lookup_key)


def check_adhesive_compatibility(model: Any, adhesive_type: AdhesiveType) -> bool:
    """
    Check if a model supports the given adhesive type.

    Args:
        model: The model to check
        adhesive_type: The adhesive type to check

    Returns:
        True if the model supports the adhesive type, False otherwise
    """
    # For backward compatibility with tests
    if hasattr(model, "adhesives") and isinstance(model.adhesives, (set, list)):
        return adhesive_type in model.adhesives

    # For models with supported_adhesives attribute
    if hasattr(model, "supported_adhesives") and isinstance(
        model.supported_adhesives, (set, list)
    ):
        # Check if the adhesive type is in the list (could be string or enum)
        return (
            adhesive_type in model.supported_adhesives
            or adhesive_type.value in model.supported_adhesives
        )

    # For models with has_adhesive method
    if hasattr(model, "has_adhesive") and callable(model.has_adhesive):
        return model.has_adhesive(adhesive_type)

    # Default to False for safety
    return False
