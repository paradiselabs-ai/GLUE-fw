from typing import Any, Optional, Dict
from .working_memory import WorkingMemory, PersistentMemory

class GLUEPersistentAdapter:
    """
    Adapter for GLUE adhesive: team-wide persistent memory using file backend.
    """
    def __init__(self, team_id: str, memory_dir: Optional[str] = None):
        self.store = PersistentMemory(team_id, memory_dir)

    def add(self, entry: Dict[str, Any]):
        """Add a persistent entry to team memory."""
        self.store.add_entry(entry)

    def get_all(self) -> list:
        """Retrieve all persistent entries."""
        return self.store.get_entries()

class VELCROSessionAdapter:
    """
    Adapter for VELCRO adhesive: session-wide in-memory working memory.
    """
    def __init__(self):
        self.store = WorkingMemory()

    def add(self, turn: int, content: str, source_tool: str):
        """Add a session memory entry."""
        self.store.add_entry(turn, content, source_tool)

    def get_all(self) -> list:
        """Retrieve all session entries."""
        return self.store.get_entries()

    def clear(self):
        """Clear session memory."""
        self.store.clear()

class TAPEEphemeralAdapter:
    """
    Adapter for TAPE adhesive: ephemeral, one-time-use memory.
    """
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def add(self, key: str, value: Any):
        """Store an ephemeral entry."""
        self.store[key] = value

    def get(self, key: str) -> Any:
        """Retrieve and remove an ephemeral entry."""
        return self.store.pop(key, None) 