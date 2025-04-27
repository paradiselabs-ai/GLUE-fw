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