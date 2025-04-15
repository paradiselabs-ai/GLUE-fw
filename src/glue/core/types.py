# glue/core/types.py
# ==================== Imports ====================
from enum import Enum
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field, asdict
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

class MessageType(str, Enum):
    """Types of messages exchanged within the GLUE system"""
    # Core types
    TOOL_RESULT = "tool_result"         # Result from a tool execution
    AGENT_FEEDBACK = "agent_feedback"     # Feedback from agent to lead (e.g., task status)
    DIRECT_MESSAGE = "direct_message"     # Direct message between agents/models
    GLUE_DATA_SHARE = "glue_data_share"   # Sharing a persisted KB entry
    # Interactive mode types
    PAUSE_QUERY = "pause_query"           # Broadcast query indicating a pause/need for refinement
    REFINEMENT_PROPOSAL = "refinement_proposal" # Proposal during lead collaboration
    REFINEMENT_ACK = "refinement_ack"       # Acknowledgement during lead collaboration
    RESUME_TASK = "resume_task"           # Instruction to resume agent task after refinement

# ==================== Class Definitions ====================

@dataclass
class V1MessagePayload:
    """Standard V1 payload structure for inter-agent/team communication"""
    task_id: str
    sender_agent_id: str
    sender_team_id: str
    timestamp: str  # ISO 8601 format string
    message_type: MessageType
    adhesive_type: AdhesiveType
    content: Any
    origin_tool_id: Optional[str] = None
    schema_version: str = "1.0.0"
    # Optional/Recommended fields for future use can be added here later
    # verification_status: Optional[str] = None
    # intended_recipients: Optional[List[str]] = None
    # priority: Optional[int] = None
    # requires_response: bool = False
    # correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the payload instance to a dictionary suitable for transmission,
        ensuring enums are converted to their string values."""
        payload_dict = asdict(self)
        # Ensure enums are strings
        if isinstance(self.message_type, MessageType):
            payload_dict['message_type'] = self.message_type.value
        if isinstance(self.adhesive_type, AdhesiveType):
            payload_dict['adhesive_type'] = self.adhesive_type.value
        return payload_dict

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
