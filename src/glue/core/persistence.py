"""
Persistence module for dynamic components in the GLUE framework.

This module provides functionality to save, load, and manage dynamically created
components such as MCP servers and tools, allowing them to persist across sessions.
"""
import os
import json
import uuid
import asyncio
import inspect
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger("glue.persistence")

class ComponentType(str, Enum):
    """Types of components that can be persisted"""
    MCP_SERVER = "mcp_server"
    TOOL = "tool"
    MCP_TOOL = "mcp_tool"

class ComponentSpec(BaseModel):
    """Specification for a persisted component"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    component_type: ComponentType
    name: str
    code: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

class DynamicComponentStore:
    """
    Store for persisting dynamic components.
    
    This class provides functionality to save, load, and manage dynamically
    created components such as MCP servers and tools, allowing them to
    persist across sessions.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the component store.
        
        Args:
            storage_dir: Directory to store component data. If None, uses
                         a default directory in the user's home folder.
        """
        if storage_dir is None:
            home_dir = os.path.expanduser("~")
            storage_dir = os.path.join(home_dir, ".glue", "components")
        
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists"""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create subdirectories for each component type
        for component_type in ComponentType:
            os.makedirs(os.path.join(self.storage_dir, component_type.value), exist_ok=True)
    
    def _get_component_path(self, component_id: str, component_type: Optional[ComponentType] = None) -> str:
        """
        Get the file path for a component.
        
        Args:
            component_id: ID of the component
            component_type: Type of the component (optional if ID contains type)
            
        Returns:
            Path to the component file
        """
        if component_type:
            return os.path.join(self.storage_dir, component_type.value, f"{component_id}.json")
        
        # Search in all component type directories
        for type_value in ComponentType:
            path = os.path.join(self.storage_dir, type_value.value, f"{component_id}.json")
            if os.path.exists(path):
                return path
        
        # If not found, assume it's a new component
        return os.path.join(self.storage_dir, "unknown", f"{component_id}.json")
    
    async def save_component(
        self,
        component_type: ComponentType,
        name: str,
        code: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save a component to the store.
        
        Args:
            component_type: Type of the component
            name: Name of the component
            code: Source code of the component
            metadata: Additional metadata for the component
            
        Returns:
            ID of the saved component
        """
        # Generate a unique ID
        component_id = str(uuid.uuid4())
        
        # Create the component spec
        now = datetime.now().isoformat()
        spec = ComponentSpec(
            id=component_id,
            component_type=component_type,
            name=name,
            code=code,
            metadata=metadata,
            created_at=now,
            updated_at=now
        )
        
        # Save to file
        await self._save_spec_to_file(spec)
        
        logger.info(f"Saved {component_type.value} '{name}' with ID {component_id}")
        return component_id
    
    async def load_component(self, component_id: str) -> ComponentSpec:
        """
        Load a component from the store.
        
        Args:
            component_id: ID of the component to load
            
        Returns:
            Component specification
            
        Raises:
            FileNotFoundError: If the component is not found
        """
        # Find the component file
        path = self._get_component_path(component_id)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Component with ID {component_id} not found")
        
        # Load from file
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            spec = ComponentSpec(**data)
            logger.info(f"Loaded component {spec.name} ({spec.component_type.value})")
            return spec
            
        except Exception as e:
            logger.error(f"Failed to load component {component_id}: {str(e)}")
            raise
    
    async def update_component(
        self,
        component_id: str,
        code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing component.
        
        Args:
            component_id: ID of the component to update
            code: New source code (if None, keeps existing code)
            metadata: New metadata (if None, keeps existing metadata)
            
        Raises:
            FileNotFoundError: If the component is not found
        """
        # Load the existing component
        spec = await self.load_component(component_id)
        
        # Update fields
        if code is not None:
            spec.code = code
        
        if metadata is not None:
            spec.metadata = metadata
        
        # Update timestamp
        spec.updated_at = datetime.now().isoformat()
        
        # Save back to file
        await self._save_spec_to_file(spec)
        
        logger.info(f"Updated component {spec.name} ({spec.component_type.value})")
    
    async def delete_component(self, component_id: str) -> None:
        """
        Delete a component from the store.
        
        Args:
            component_id: ID of the component to delete
            
        Raises:
            FileNotFoundError: If the component is not found
        """
        # Find the component file
        path = self._get_component_path(component_id)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Component with ID {component_id} not found")
        
        # Delete the file
        os.remove(path)
        
        logger.info(f"Deleted component with ID {component_id}")
    
    async def list_components(
        self,
        component_type: Optional[ComponentType] = None
    ) -> List[ComponentSpec]:
        """
        List all components in the store.
        
        Args:
            component_type: Filter by component type (optional)
            
        Returns:
            List of component specifications
        """
        components = []
        
        # Determine which directories to search
        if component_type:
            type_dirs = [os.path.join(self.storage_dir, component_type.value)]
        else:
            type_dirs = [os.path.join(self.storage_dir, t.value) for t in ComponentType]
        
        # Search each directory
        for dir_path in type_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for filename in os.listdir(dir_path):
                if not filename.endswith('.json'):
                    continue
                    
                try:
                    with open(os.path.join(dir_path, filename), 'r') as f:
                        data = json.load(f)
                    
                    spec = ComponentSpec(**data)
                    components.append(spec)
                    
                except Exception as e:
                    logger.warning(f"Failed to load component {filename}: {str(e)}")
        
        return components
    
    async def _save_spec_to_file(self, spec: ComponentSpec) -> None:
        """
        Save a component specification to a file.
        
        Args:
            spec: Component specification to save
        """
        # Ensure the directory exists
        os.makedirs(os.path.join(self.storage_dir, spec.component_type.value), exist_ok=True)
        
        # Get the file path
        path = self._get_component_path(spec.id, spec.component_type)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(spec.model_dump(), f, indent=2)
