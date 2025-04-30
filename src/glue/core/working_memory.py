import os
import json
from typing import Optional, List

class WorkingMemory:
    """In-memory working memory for agent loops."""
    def __init__(self):
        # Internal list to store memory entries
        self._entries = []

    def add_entry(self, turn: int, content: str, source_tool: str):
        """Add a new entry with turn number, content, and source tool."""
        entry = {"turn": turn, "content": content, "source_tool": source_tool}
        self._entries.append(entry)

    def get_entries(self) -> list:
        """Retrieve a copy of all stored entries."""
        return list(self._entries)

    def clear(self):
        """Clear all stored memory entries."""
        self._entries.clear()

class PersistentMemory:
    """File-based persistent memory for team leads."""
    def __init__(self, team_id: str, memory_dir: Optional[str] = None):
        self.team_id = team_id
        # Default directory to store persistent memory files
        self.memory_dir = memory_dir or os.path.join(os.getcwd(), "persistent_memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.file_path = os.path.join(self.memory_dir, f"{self.team_id}.json")
        # Load existing memory entries or initialize empty
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._entries = []
        else:
            self._entries = []
            with open(self.file_path, "w") as f:
                json.dump(self._entries, f, indent=2)

    def add_entry(self, entry: dict):
        """Add a new entry and persist to file."""
        self._entries.append(entry)
        self._save()

    def get_entries(self) -> List[dict]:
        """Retrieve all persistent entries."""
        return list(self._entries)

    def _save(self):
        """Write all entries back to the JSON file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._entries, f, indent=2)
        except IOError:
            pass 