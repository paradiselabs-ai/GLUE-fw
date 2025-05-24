"""
Hierarchy detection and management utilities for GLUE framework.

This module provides functions to detect and manage hierarchy structures within teams,
enabling intelligent tool assignment based on organizational roles.
"""

from typing import Any, Optional, List, Dict, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Enumeration of hierarchy levels in team structure."""
    ORCHESTRATOR = 0  # Future: Team Orchestrator (highest level)
    LEAD = 1         # Team Lead (current highest level)
    MEMBER = 2       # Team Member (lowest level)


class HierarchyDetectionError(Exception):
    """Raised when hierarchy detection encounters an error."""
    pass


def get_hierarchy_structure(team: Any) -> Dict[str, int]:
    """
    Analyze team structure and return hierarchy levels for all models.
    
    Args:
        team: Team instance to analyze
        
    Returns:
        Dictionary mapping model names to their hierarchy levels (0=highest)
        
    Raises:
        AttributeError: If team is None or missing required attributes
        HierarchyDetectionError: If hierarchy cannot be determined for other reasons
    """
    if team is None:
        raise HierarchyDetectionError("Team instance is None")

    if not hasattr(team, 'config') or not hasattr(team, 'models'):
        raise HierarchyDetectionError("Team missing required config or models attributes")

    if not team.config:
        raise HierarchyDetectionError("Team config or models are improperly initialized")

    if not isinstance(team.models, dict) or not team.models:
        raise HierarchyDetectionError("No models found in team")

    hierarchy = {}

    # Identify lead model (hierarchy level 1)
    lead_name = getattr(team.config, 'lead', None)
    if lead_name and lead_name in team.models:
        hierarchy[lead_name] = HierarchyLevel.LEAD.value
        logger.debug(f"Identified lead model: {lead_name} at level {HierarchyLevel.LEAD.value}")

    # Identify member models (hierarchy level 2)
    for model_name in team.models:
        if model_name not in hierarchy:
            hierarchy[model_name] = HierarchyLevel.MEMBER.value
            logger.debug(f"Identified member model: {model_name} at level {HierarchyLevel.MEMBER.value}")

    # Future extensibility: Check for orchestrator level (level 0)
    orchestrator_name = getattr(team.config, 'orchestrator', None)
    if orchestrator_name and orchestrator_name in team.models:
        hierarchy[orchestrator_name] = HierarchyLevel.ORCHESTRATOR.value
        logger.debug(f"Identified orchestrator model: {orchestrator_name} at level {HierarchyLevel.ORCHESTRATOR.value}")

    return hierarchy


def get_highest_ranking_model(team: Any) -> Optional[str]:
    """
    Identify the highest-ranking model in the team hierarchy.
    
    Args:
        team: Team instance to analyze
        
    Returns:
        Name of the highest-ranking model
        
    Raises:
        HierarchyDetectionError: If hierarchy cannot be determined or no models found
    """
    try:
        hierarchy = get_hierarchy_structure(team)
        
        if not hierarchy:
            logger.warning(f"No models found in team {getattr(team, 'name', 'unknown')}")
            raise HierarchyDetectionError("No models found in team")
        
        # Find model with lowest hierarchy level (highest rank)
        highest_ranking = min(hierarchy.items(), key=lambda x: x[1])
        model_name, level = highest_ranking
        
        logger.info(f"Highest-ranking model in team {getattr(team, 'name', 'unknown')}: {model_name} (level {level})")
        return model_name
        
    except Exception as e:
        if isinstance(e, HierarchyDetectionError):
            raise
        raise HierarchyDetectionError(f"Failed to determine highest-ranking model: {str(e)}")


def get_hierarchy_level(team: Any, model_name: str) -> Optional[int]:
    """
    Get the hierarchy level of a specific model.
    
    Args:
        team: Team instance
        model_name: Name of the model to check
        
    Returns:
        Hierarchy level (0=highest), or None if model not found
    """
    try:
        hierarchy = get_hierarchy_structure(team)
        return hierarchy.get(model_name)
    except HierarchyDetectionError:
        logger.warning(f"Could not determine hierarchy level for model {model_name}")
        return None


def is_top_hierarchy(team: Any, model_name: str) -> bool:
    """
    Check if a model is at the top of the hierarchy.
    
    Args:
        team: Team instance
        model_name: Name of the model to check
        
    Returns:
        True if model is at the top of hierarchy, False otherwise
        
    Raises:
        HierarchyDetectionError: If model not found or hierarchy cannot be determined
    """
    hierarchy = get_hierarchy_structure(team)
    
    # Check if the model exists in the team
    if model_name not in hierarchy:
        raise HierarchyDetectionError(f"Model '{model_name}' not found")
        
    try:
        highest_ranking = get_highest_ranking_model(team)
        return highest_ranking == model_name
    except HierarchyDetectionError as e:
        # Re-raise the exception to maintain error context
        raise e


def get_models_at_level(team: Any, level: int) -> List[str]:
    """
    Get all models at a specific hierarchy level.
    
    Args:
        team: Team instance
        level: Hierarchy level to query
        
    Returns:
        List of model names at the specified level
    """
    try:
        hierarchy = get_hierarchy_structure(team)
        return [name for name, model_level in hierarchy.items() if model_level == level]
    except HierarchyDetectionError:
        logger.warning(f"Could not get models at level {level}")
        return []


def set_hierarchy_attributes(team: Any) -> None:
    """
    Set hierarchy-related attributes on all models in the team.
    
    Args:
        team: Team instance to update
    """
    try:
        hierarchy = get_hierarchy_structure(team)
        highest_ranking = get_highest_ranking_model(team)
        
        for model_name, model in team.models.items():
            # Set hierarchy level
            model.hierarchy_level = hierarchy.get(model_name, HierarchyLevel.MEMBER.value)
            
            # Set top hierarchy flag
            model.is_hierarchy_top = (model_name == highest_ranking)
            
            # Set user input access flag (only top hierarchy gets access)
            model.has_user_input_access = (model_name == highest_ranking)
            
            logger.debug(f"Set hierarchy attributes for {model_name}: level={model.hierarchy_level}, "
                        f"is_top={model.is_hierarchy_top}, has_user_input={model.has_user_input_access}")
                        
    except Exception as e:
        logger.error(f"Failed to set hierarchy attributes: {str(e)}")


def validate_hierarchy_consistency(team: Any) -> Tuple[bool, List[str]]:
    """
    Validate that the team hierarchy is consistent and well-formed.
    
    Args:
        team: Team instance to validate
        
    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors = []
    
    try:
        hierarchy = get_hierarchy_structure(team)
        
        # Check for empty hierarchy
        if not hierarchy:
            errors.append("No models found in team")
            return False, errors
        
        # Check for multiple models at level 0 (orchestrator)
        orchestrators = get_models_at_level(team, HierarchyLevel.ORCHESTRATOR.value)
        if len(orchestrators) > 1:
            errors.append(f"Multiple orchestrators found: {orchestrators}")
        
        # Check for multiple models at level 1 (lead) 
        leads = get_models_at_level(team, HierarchyLevel.LEAD.value)
        if len(leads) > 1:
            errors.append(f"Multiple team leads found: {leads}")
        
        # Check for no top-level model
        if not orchestrators and not leads:
            errors.append("No top-level model (orchestrator or lead) found")
        
        # Check that config.lead matches actual lead
        config_lead = getattr(team.config, 'lead', None)
        if config_lead and config_lead not in leads and not orchestrators:
            errors.append(f"Config lead '{config_lead}' not found in hierarchy")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Hierarchy validation failed: {str(e)}")
        return False, errors


def validate_team_structure(team: Any) -> None:
    """
    Validate that the team structure is properly initialized.

    Args:
        team: Team instance to validate

    Raises:
        HierarchyDetectionError: If team structure is invalid
    """
    if not hasattr(team, 'config') or not hasattr(team, 'models'):
        raise HierarchyDetectionError("Team is missing required attributes: 'config' or 'models'")
    if not isinstance(team.models, dict) or not team.models:
        raise HierarchyDetectionError("Team models must be a non-empty dictionary")
    if not hasattr(team.config, 'lead') or not isinstance(team.config.lead, str):
        raise HierarchyDetectionError("Team config must have a valid 'lead' attribute")


# Edge case handlers

def handle_multiple_top_models(team: Any, top_models: List[str]) -> Optional[str]:
    """
    Handle edge case where multiple models are at the same top level.
    
    Args:
        team: Team instance
        top_models: List of models at the top level
        
    Returns:
        Name of the selected top model, or None if cannot resolve
    """
    if len(top_models) == 1:
        return top_models[0]
    
    if len(top_models) == 0:
        logger.error("No top models provided to handle_multiple_top_models")
        return None
    
    # Strategy 1: Use config.lead if it's in the list
    config_lead = getattr(team.config, 'lead', None)
    if config_lead and config_lead in top_models:
        logger.info(f"Resolved conflict by selecting config lead: {config_lead}")
        return config_lead
    
    # Strategy 2: Use the first model alphabetically for consistency
    selected = sorted(top_models)[0]
    logger.warning(f"Multiple top models {top_models}, selected alphabetically: {selected}")
    return selected


def handle_no_hierarchy(team: Any) -> Optional[str]:
    """
    Handle edge case where no clear hierarchy exists.
    
    Args:
        team: Team instance
        
    Returns:
        Name of a default model to use, or None if no models exist
    """
    if not hasattr(team, 'models') or not team.models:
        logger.error("No models found in team")
        return None
    
    # Strategy: Use the first model as default
    default_model = next(iter(team.models.keys()))
    logger.warning(f"No clear hierarchy found, using default model: {default_model}")
    return default_model
