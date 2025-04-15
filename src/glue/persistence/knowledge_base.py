import json
import logging
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("glue.persistence.kb")

class KnowledgeBase:
    """
    Simple file-based knowledge base using JSON Lines format.
    Each team gets its own KB file.
    """
    def __init__(self, base_path: str = "kb_data", team_id: str = "default_team"):
        """
        Initialize the Knowledge Base.

        Args:
            base_path: Directory where KB files will be stored.
            team_id: Identifier for the team whose KB this is.
        """
        self.base_path = base_path
        self.team_id = team_id
        self._ensure_base_path_exists()
        self.kb_path = self._get_kb_path()
        logger.info(f"KnowledgeBase initialized for team '{self.team_id}' at {self.kb_path}")

    def _ensure_base_path_exists(self):
        """Create the base directory if it doesn't exist."""
        try:
            os.makedirs(self.base_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating KB base path '{self.base_path}': {e}")
            raise

    def _get_kb_path(self) -> str:
        """Get the full path to the team's KB file."""
        # Sanitize team_id to make it a valid filename component
        safe_team_id = "".join(c for c in self.team_id if c.isalnum() or c in ('_', '-')).rstrip()
        if not safe_team_id:
            safe_team_id = "default_team"
        return os.path.join(self.base_path, f"{safe_team_id}_kb.jsonl")

    def add_entry(self, entry_data: Dict[str, Any]) -> Optional[str]:
        """
        Add a new entry to the knowledge base.

        Args:
            entry_data: Dictionary containing the data for the new entry. 
                        Should ideally include a unique 'id'.

        Returns:
            The ID of the added entry, or None if failed.
        """
        if 'id' not in entry_data:
            entry_data['id'] = str(uuid.uuid4())
        if 'timestamp_added' not in entry_data:
             entry_data['timestamp_added'] = datetime.now().isoformat()
             
        entry_id = entry_data['id']

        try:
            json_string = json.dumps(entry_data)
            with open(self.kb_path, 'a', encoding='utf-8') as f:
                f.write(json_string + '\n')
            logger.info(f"Added entry {entry_id} to KB for team '{self.team_id}'")
            return entry_id
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Error adding entry {entry_id} to KB '{self.kb_path}': {e}")
            return None

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load all entries from the KB file."""
        entries = []
        try:
            if os.path.exists(self.kb_path):
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entries.append(entry)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line in {self.kb_path}: {line.strip()}")
        except IOError as e:
            logger.error(f"Error reading KB file '{self.kb_path}': {e}")
        return entries

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific entry by its ID.

        Args:
            entry_id: The unique ID of the entry to retrieve.

        Returns:
            The entry dictionary if found, otherwise None.
        """
        entries = self._load_entries()
        for entry in entries:
            if entry.get('id') == entry_id:
                return entry
        logger.debug(f"Entry {entry_id} not found in KB for team '{self.team_id}'")
        return None
        
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """
        Retrieve all entries from the knowledge base.

        Returns:
            A list of all entry dictionaries.
        """
        return self._load_entries()

    def delete_entry(self, entry_id: str) -> bool:
        """
        Deletes an entry by its ID. Reads all entries, filters out the target, 
        and rewrites the file. Use with caution on large files.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if an entry was deleted, False otherwise.
        """
        entries = self._load_entries()
        original_count = len(entries)
        entries_to_keep = [entry for entry in entries if entry.get('id') != entry_id]
        
        if len(entries_to_keep) < original_count:
            try:
                # Rewrite the file with the filtered entries
                with open(self.kb_path, 'w', encoding='utf-8') as f:
                    for entry in entries_to_keep:
                        json_string = json.dumps(entry)
                        f.write(json_string + '\n')
                logger.info(f"Deleted entry {entry_id} from KB for team '{self.team_id}'")
                return True
            except IOError as e:
                logger.error(f"Error rewriting KB file '{self.kb_path}' after deletion: {e}")
                # Consider restoring from backup or handling error more robustly
                return False
        else:
            logger.warning(f"Attempted to delete entry {entry_id}, but it was not found.")
            return False
            
    def clear_kb(self) -> bool:
        """Deletes the KB file, effectively clearing all entries."""
        try:
            if os.path.exists(self.kb_path):
                os.remove(self.kb_path)
                logger.info(f"Cleared knowledge base file for team '{self.team_id}' at {self.kb_path}")
                return True
            return False # File didn't exist
        except OSError as e:
            logger.error(f"Error deleting KB file '{self.kb_path}': {e}")
            return False

# Example Usage (for testing)
if __name__ == '__main__':
    kb = KnowledgeBase(team_id="test_team_123")
    kb.clear_kb() # Start fresh

    entry1_id = kb.add_entry({"tool_name": "web_search", "result": "Python is a language.", "source_agent": "agent1"})
    entry2_data = {"id": "manual_id_abc", "tool_name": "calculator", "result": 42, "source_agent": "agent2"}
    entry2_id = kb.add_entry(entry2_data)
    
    print("All entries:", kb.get_all_entries())
    print("Entry 1:", kb.get_entry(entry1_id))
    print("Entry 2:", kb.get_entry("manual_id_abc"))
    
    kb.delete_entry(entry1_id)
    print("All entries after delete:", kb.get_all_entries())
    kb.clear_kb() 