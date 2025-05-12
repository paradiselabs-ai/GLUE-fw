"""
GLUE Framework Core Module

This module provides the core functionality of the GLUE framework,
including application management, model connectors, and adhesive types.
"""

from .app import GlueApp
from .types import AdhesiveType
from .glue_smolagent import GlueSmolAgent
from .glue_smoltool import GlueSmolTool

# Export public API
__all__ = ["GlueApp", "AdhesiveType", "GlueSmolAgent", "GlueSmolTool"]
