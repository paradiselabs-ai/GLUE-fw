# glue/core/types.py
# ==================== Imports ====================
from enum import Enum
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ==================== Constants ====================
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3

# ==================== Type Definitions ====================
class AdhesiveType(str, Enum):
    """Types of adhesive bindings that control tool result persistence"""
    GLUE = "glue"    # Team-wide persistent results
    VELCRO = "velcro" # Session-based persistence
    TAPE = "tape"    # One-time use, no persistence

class FlowType(str, Enum):
    """Types of magnetic flows between teams"""
    BIDIRECTIONAL = "><"  # Free flowing both ways
    PUSH = "->"          # Source pushes to target
    PULL = "<-"          # Target pulls from source
    REPEL = "<>"         # No interaction allowed

# ==================== Class Definitions ====================
@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    result: Any
    adhesive: AdhesiveType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Message:
    """Message for communication"""
    role: str
    content: str
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
