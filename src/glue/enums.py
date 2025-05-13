# glue/enums.py
from enum import Enum

class AdhesiveType(str, Enum):
    """Types of adhesive bindings that control tool result persistence"""

    GLUE = "glue"  # Team-wide persistent results
    VELCRO = "velcro"  # Session-based persistence
    TAPE = "tape"  # One-time use, no persistence

class ToolPermission(Enum):
    """Tool permission levels"""

    READ = "read"  # Read-only operations
    WRITE = "write"  # File/state modifications
    NETWORK = "network"  # Network access
    EXECUTE = "execute"  # Code execution

class FlowType(str, Enum):
    """Types of magnetic flows between teams"""

    BIDIRECTIONAL = "><"  # Free flowing both ways
    PUSH = "->"  # Source pushes to target
    PULL = "<-"  # Target pulls from source
    REPEL = "<>"  # No interaction allowed

class TaskStatus(str, Enum):
    """States of a subtask in the orchestrator lifecycle."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    REPORTED = "reported"
    COMPLETE = "complete"
    FAILED = "failed"
    PENDING_RETRY = "pending_retry"
    NO_MEMBER = "no_member"
    ESCALATED = "escalated"
