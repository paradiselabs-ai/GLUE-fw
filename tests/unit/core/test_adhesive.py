"""
Tests for the adhesive system in the GLUE framework.

This module contains tests for the adhesive system, which controls
how tool results are bound and persisted within the framework.
"""
import pytest
import asyncio
from typing import Dict, Any, List, Set, Optional

from glue.core.schemas import AdhesiveType, ToolCall, ToolResult
from glue.core.adhesive import AdhesiveManager, bind_tool_result, get_tool_result, check_adhesive_compatibility


# Simple implementation classes for testing - not actual test classes
class _TestTeam:
    """Simple team implementation for testing."""
    def __init__(self, name: str):
        self.name = name


class _TestModel:
    """Simple model implementation for testing."""
    def __init__(self, name: str, supported_adhesives: Optional[Set[AdhesiveType]] = None):
        self.name = name
        self.supported_adhesives = supported_adhesives or {
            AdhesiveType.GLUE, 
            AdhesiveType.VELCRO, 
            AdhesiveType.TAPE
        }


class TestAdhesiveSystem:
    """Tests for the adhesive system functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.team = _TestTeam(name="test_team")
        self.model = _TestModel(name="test_model")
        
        # Create a tool call and result
        self.tool_call = ToolCall(
            tool_id="call_123",
            name="test_tool",
            arguments={"query": "test query"}
        )
        
        self.tool_result = ToolResult(
            tool_name="test_tool",
            result={"result": "test result"},
            adhesive=AdhesiveType.GLUE
        )
    
    def test_adhesive_manager_initialization(self):
        """Test that the AdhesiveManager initializes correctly."""
        manager = AdhesiveManager()
        
        # Verify the manager has empty storage for each adhesive type
        assert hasattr(manager, "glue_storage")
        assert isinstance(manager.glue_storage, Dict)
        assert len(manager.glue_storage) == 0
        
        assert hasattr(manager, "velcro_storage")
        assert isinstance(manager.velcro_storage, Dict)
        assert len(manager.velcro_storage) == 0
        
        assert hasattr(manager, "tape_storage")
        assert isinstance(manager.tape_storage, Dict)
        assert len(manager.tape_storage) == 0
    
    def test_bind_tool_result_with_glue(self):
        """Test binding a tool result with GLUE adhesive type."""
        manager = AdhesiveManager()
        
        # Bind the tool result with GLUE adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Verify the result is stored in the glue storage
        assert "test_tool" in manager.glue_storage.get("test_team", {})
        stored_result = manager.glue_storage["test_team"]["test_tool"]
        assert stored_result["model"] == "test_model"
        assert stored_result["result"] == self.tool_result
    
    def test_bind_tool_result_with_velcro(self):
        """Test binding a tool result with VELCRO adhesive type."""
        manager = AdhesiveManager()
        
        # Bind the tool result with VELCRO adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Verify the result is stored in the velcro storage
        assert "test_tool" in manager.velcro_storage.get("test_model", {})
        stored_result = manager.velcro_storage["test_model"]["test_tool"]
        assert stored_result["team"] == "test_team"
        assert stored_result["result"] == self.tool_result
    
    def test_bind_tool_result_with_tape(self):
        """Test binding a tool result with TAPE adhesive type."""
        manager = AdhesiveManager()
        
        # Bind the tool result with TAPE adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Verify the result is stored in the tape storage
        assert "test_tool" in manager.tape_storage
        stored_result = manager.tape_storage["test_tool"]
        assert stored_result["team"] == "test_team"
        assert stored_result["model"] == "test_model"
        assert stored_result["result"] == self.tool_result
    
    def test_retrieve_tool_result_glue(self):
        """Test retrieving a tool result stored with GLUE adhesive."""
        manager = AdhesiveManager()
        
        # Bind the tool result with GLUE adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Retrieve the tool result
        result = get_tool_result(
            manager=manager,
            tool_name="test_tool"
        )
        
        # Verify the result
        assert result == self.tool_result
    
    def test_retrieve_tool_result_velcro(self):
        """Test retrieving a tool result stored with VELCRO adhesive."""
        manager = AdhesiveManager()
        
        # Bind the tool result with VELCRO adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Retrieve the tool result
        result = get_tool_result(
            manager=manager,
            tool_name="test_tool"
        )
        
        # Verify the result
        assert result == self.tool_result
    
    def test_retrieve_tool_result_tape(self):
        """Test retrieving a tool result stored with TAPE adhesive."""
        manager = AdhesiveManager()
        
        # Bind the tool result with TAPE adhesive
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Retrieve the tool result
        result = get_tool_result(
            manager=manager,
            tool_name="test_tool"
        )
        
        # Verify the result
        assert result == self.tool_result
        
        # Verify the result is removed from TAPE storage after retrieval
        assert "test_tool" not in manager.tape_storage
    
    def test_retrieve_nonexistent_tool_result(self):
        """Test retrieving a nonexistent tool result."""
        manager = AdhesiveManager()
        
        # Retrieve a nonexistent tool result
        result = get_tool_result(
            manager=manager,
            tool_name="nonexistent_tool"
        )
        
        # Verify the result is None
        assert result is None
    
    def test_clear_adhesive_storage(self):
        """Test clearing adhesive storage."""
        manager = AdhesiveManager()
        
        # Bind tool results with different adhesive types
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Create a second tool result
        tool_result2 = ToolResult(
            tool_name="another_tool",
            result={"result": "another result"},
            adhesive=AdhesiveType.VELCRO
        )
        
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=tool_result2,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Clear all storage
        manager.clear_all_storage()
        
        # Verify all storage is empty
        assert len(manager.glue_storage) == 0
        assert len(manager.velcro_storage) == 0
        assert len(manager.tape_storage) == 0
    
    def test_clear_specific_adhesive_storage(self):
        """Test clearing specific adhesive storage."""
        manager = AdhesiveManager()
        
        # Bind tool results with different adhesive types
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=self.tool_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Create a second tool result
        tool_result2 = ToolResult(
            tool_name="another_tool",
            result={"result": "another result"},
            adhesive=AdhesiveType.VELCRO
        )
        
        bind_tool_result(
            manager=manager,
            team=self.team,
            model=self.model,
            tool_result=tool_result2,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Clear only GLUE storage
        manager.clear_storage(AdhesiveType.GLUE)
        
        # Verify GLUE storage is empty but VELCRO storage still has content
        assert len(manager.glue_storage) == 0
        assert len(manager.velcro_storage) > 0
        assert "another_tool" in manager.velcro_storage.get("test_model", {})
    
    def test_adhesive_compatibility(self):
        """Test checking adhesive compatibility."""
        # Create a model with limited adhesive support
        limited_model = _TestModel(
            name="limited_model",
            supported_adhesives={AdhesiveType.TAPE}
        )
        
        # Check compatibility
        assert check_adhesive_compatibility(self.model, AdhesiveType.GLUE)
        assert check_adhesive_compatibility(self.model, AdhesiveType.VELCRO)
        assert check_adhesive_compatibility(self.model, AdhesiveType.TAPE)
        
        assert not check_adhesive_compatibility(limited_model, AdhesiveType.GLUE)
        assert not check_adhesive_compatibility(limited_model, AdhesiveType.VELCRO)
        assert check_adhesive_compatibility(limited_model, AdhesiveType.TAPE)
