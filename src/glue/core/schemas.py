# Schema compatibility layer - imports from simple_schemas for backward compatibility
"""
This module provides backward compatibility by re-exporting models from simple_schemas.
The GLUE framework has moved away from pydantic to simple dataclasses that align
better with smolagents' lightweight philosophy.
"""

# Re-export all models from simple_schemas for backward compatibility
from .simple_schemas import (
    AdhesiveType,
    FlowType,
    ModelConfig,
    ToolCall,
    Message,
    ToolConfig,
    ToolResult,
    create_model_config_from_dict,
    create_tool_config_from_dict,
)

__all__ = [
    "AdhesiveType",
    "FlowType", 
    "ModelConfig",
    "ToolCall",
    "Message",
    "ToolConfig",
    "ToolResult",
    "create_model_config_from_dict",
    "create_tool_config_from_dict",
]