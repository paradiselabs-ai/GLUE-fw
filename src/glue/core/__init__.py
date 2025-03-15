"""
GLUE Framework Core Module

This module provides the core functionality of the GLUE framework,
including application management, model connectors, and adhesive types.
"""

from .app import GlueApp
from .model import BaseModel, ModelProvider

# Adhesive types for tool binding
class AdhesiveType:
    """Enumeration of adhesive types for tool binding."""
    STRONG = "strong"  # Tool is always used
    WEAK = "weak"      # Tool is suggested but optional
    STICKY = "sticky"  # Tool persists across multiple steps
