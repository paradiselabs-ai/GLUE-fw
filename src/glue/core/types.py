# glue/core/types.py
# ==================== Imports ====================
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from ..enums import AdhesiveType, FlowType, TaskStatus

# ==================== Constants ====================
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


# ==================== Class Definitions ====================
@dataclass
class ToolResult:
    """Result from a tool execution"""

    tool_name: str
    result: Any
    adhesive: AdhesiveType = AdhesiveType.TAPE  # Default to TAPE (one-time use)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_error: bool = False  # Flag to indicate if this result represents an error


@dataclass
class Message:
    """Message for communication"""

    role: str
    content: str
    name: Optional[str] = None  # For function messages
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Configuration Classes ====================
@dataclass
class ModelConfig:
    """Configuration for an LLM model"""

    provider: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    api_params: Dict[str, Any] = field(default_factory=dict)
    supported_adhesives: List[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    """Configuration for a tool"""

    name: str
    description: str
    provider: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamConfig:
    """Team configuration"""

    name: str
    lead: str
    members: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
