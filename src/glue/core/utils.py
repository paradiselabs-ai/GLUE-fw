"""
Utility functions for the GLUE framework.

This module contains helper functions for creating models, tools, and other
components used throughout the framework.
"""

from typing import Dict, Any
import logging

from .model import Model

logger = logging.getLogger("glue.utils")


def create_model(config: Dict[str, Any]) -> Model:
    """Create a model from a configuration dictionary.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_name = config.get("name", "unnamed_model")
    model_config = config.get("config")
    adhesives = config.get("adhesives", [])
    role = config.get("role", "")

    # Create the model
    model = Model(model_config)

    # Set model properties
    model.name = model_name
    model.adhesives = adhesives
    model.role = role

    return model
