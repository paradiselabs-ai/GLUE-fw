"""
GLUE Framework Core Module

This module provides the core functionality of the GLUE framework,
including application management, model connectors, and adhesive types.
"""

from .app import GlueApp
from .model import Model
from .base_model import BaseModel, ModelProvider
from .types import AdhesiveType

# Export public API
__all__ = ["GlueApp", "Model", "BaseModel", "ModelProvider", "AdhesiveType"]
