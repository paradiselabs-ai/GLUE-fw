"""
Tests for the adhesive system in the GLUE framework.

This module contains tests for the adhesive system, which controls
how tool results are bound and persisted within teams and models.
"""
import pytest
from typing import Dict, Any, List, Optional

from glue.core.schemas import AdhesiveType, ToolCall, ToolResult, Message
from glue.core.model import Model
from glue.core.team import Team


class TestAdhesiveSystem:
    """Tests for the adhesive system functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create actual models and teams
        self.researcher_model = Model(
            name="researcher",
            provider="custom",
            role="researcher",
            adhesives={AdhesiveType.GLUE, AdhesiveType.VELCRO}
        )
        
        self.assistant_model = Model(
            name="assistant",
            provider="custom",
            role="assistant",
            adhesives={AdhesiveType.GLUE, AdhesiveType.VELCRO}
        )
        
        self.writer_model = Model(
            name="writer",
            provider="custom",
            role="writer",
            adhesives={AdhesiveType.TAPE}
        )
        
        self.research_team = Team(
            name="researchers",
            lead=self.researcher_model,
            members=[self.assistant_model]
        )
        
        self.docs_team = Team(
            name="docs",
            lead=self.writer_model,
            members=[]
        )
        
        # Create actual tool calls and results
        self.web_search_call = ToolCall(
            tool_id="search_123",
            name="web_search",
            arguments={"query": "GLUE framework"}
        )
        
        self.web_search_result = ToolResult(
            tool_name="web_search",
            result={"results": ["Result 1", "Result 2"]},
            adhesive=AdhesiveType.GLUE
        )
        
        self.code_call = ToolCall(
            tool_id="code_456",
            name="code_interpreter",
            arguments={"code": "print('Hello GLUE')"}
        )
        
        self.code_result = ToolResult(
            tool_name="code_interpreter",
            result={"output": "Hello GLUE", "error": None},
            adhesive=AdhesiveType.VELCRO
        )
        
        self.file_call = ToolCall(
            tool_id="file_789",
            name="file_handler",
            arguments={"path": "/tmp/test.txt", "content": "Test content"}
        )
        
        self.file_result = ToolResult(
            tool_name="file_handler",
            result={"success": True, "bytes_written": 12},
            adhesive=AdhesiveType.TAPE
        )
    
    def test_adhesive_system_import(self):
        """Test that the adhesive system module can be imported."""
        try:
            from glue.core.adhesive import AdhesiveManager
            assert True
        except ImportError:
            pytest.fail("Failed to import AdhesiveManager")
    
    def test_adhesive_system_initialization(self):
        """Test that the AdhesiveManager initializes correctly."""
        from glue.core.adhesive import AdhesiveManager
        
        system = AdhesiveManager()
        
        # Verify the system has empty storage for each adhesive type
        assert hasattr(system, "glue_storage")
        assert isinstance(system.glue_storage, Dict)
        assert len(system.glue_storage) == 0
        
        assert hasattr(system, "velcro_storage")
        assert isinstance(system.velcro_storage, Dict)
        assert len(system.velcro_storage) == 0
        
        assert hasattr(system, "tape_storage")
        assert isinstance(system.tape_storage, Dict)
        assert len(system.tape_storage) == 0
    
    def test_bind_tool_result_with_glue(self):
        """Test binding a tool result with GLUE adhesive type."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind the tool result with GLUE adhesive
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.web_search_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Verify the result is stored in the glue storage
        team_storage = system.glue_storage.get(self.research_team.name, {})
        assert self.web_search_result.tool_name in team_storage
        assert team_storage[self.web_search_result.tool_name] == {
            "model": self.researcher_model.name,
            "result": self.web_search_result,
            "timestamp": team_storage[self.web_search_result.tool_name]["timestamp"]  # Just check it exists
        }
    
    def test_bind_tool_result_with_velcro(self):
        """Test binding a tool result with VELCRO adhesive type."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind the tool result with VELCRO adhesive
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.code_result,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Verify the result is stored in the velcro storage
        model_storage = system.velcro_storage.get(self.researcher_model.name, {})
        assert self.code_result.tool_name in model_storage
        assert model_storage[self.code_result.tool_name] == {
            "team": self.research_team.name,
            "result": self.code_result,
            "timestamp": model_storage[self.code_result.tool_name]["timestamp"]  # Just check it exists
        }
    
    def test_bind_tool_result_with_tape(self):
        """Test binding a tool result with TAPE adhesive type."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind the tool result with TAPE adhesive
        bind_tool_result(
            manager=system,
            team=self.docs_team,
            model=self.writer_model,
            tool_result=self.file_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Verify the result is stored in the tape storage
        assert self.file_result.tool_name in system.tape_storage
        assert system.tape_storage[self.file_result.tool_name] == {
            "team": self.docs_team.name,
            "model": self.writer_model.name,
            "result": self.file_result,
            "timestamp": system.tape_storage[self.file_result.tool_name]["timestamp"]  # Just check it exists
        }
    
    def test_get_tool_results_for_team(self):
        """Test getting tool results for a team with GLUE adhesive."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind multiple tool results with GLUE adhesive
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.web_search_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.assistant_model,
            tool_result=self.code_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        # Get tool results for the research team
        results = system.get_team_tool_results(self.research_team.name)
        
        # Verify we got both results
        assert len(results) == 2
        assert any(r["result"].tool_name == self.web_search_result.tool_name for r in results)
        assert any(r["result"].tool_name == self.code_result.tool_name for r in results)
    
    def test_get_tool_results_for_model(self):
        """Test getting tool results for a model with VELCRO adhesive."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind a tool result with VELCRO adhesive
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.code_result,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        # Get tool results for the researcher model
        results = system.get_model_tool_results(self.researcher_model.name)
        
        # Verify we got the result
        assert len(results) == 1
        assert results[0]["result"].tool_name == self.code_result.tool_name
    
    def test_use_tape_result_once(self):
        """Test that TAPE results can only be used once."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result, get_tool_result
        
        system = AdhesiveManager()
        
        # Bind a tool result with TAPE adhesive
        bind_tool_result(
            manager=system,
            team=self.docs_team,
            model=self.writer_model,
            tool_result=self.file_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Get the result once
        result1 = get_tool_result(
            manager=system,
            tool_name=self.file_result.tool_name
        )
        
        # Verify we got the result
        assert result1 is not None
        assert result1.tool_name == self.file_result.tool_name
        
        # Try to get the result again
        result2 = get_tool_result(
            manager=system,
            tool_name=self.file_result.tool_name
        )
        
        # Verify the result is no longer available
        assert result2 is None
    
    def test_adhesive_compatibility(self):
        """Test that models can only use adhesives they support."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result, check_adhesive_compatibility
        
        # Check that the researcher model can use GLUE and VELCRO
        assert check_adhesive_compatibility(self.researcher_model, AdhesiveType.GLUE) is True
        assert check_adhesive_compatibility(self.researcher_model, AdhesiveType.VELCRO) is True
        assert check_adhesive_compatibility(self.researcher_model, AdhesiveType.TAPE) is False
        
        # Check that the writer model can use TAPE
        assert check_adhesive_compatibility(self.writer_model, AdhesiveType.GLUE) is False
        assert check_adhesive_compatibility(self.writer_model, AdhesiveType.VELCRO) is False
        assert check_adhesive_compatibility(self.writer_model, AdhesiveType.TAPE) is True
    
    def test_clear_team_storage(self):
        """Test clearing storage for a specific team."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind tool results for different teams
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.web_search_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        bind_tool_result(
            manager=system,
            team=self.docs_team,
            model=self.writer_model,
            tool_result=self.file_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Clear storage for the research team
        system.clear_team_storage(self.research_team.name)
        
        # Verify research team storage is empty but docs team storage is not
        assert self.research_team.name not in system.glue_storage
        assert self.file_result.tool_name in system.tape_storage
    
    def test_adhesive_system_reset(self):
        """Test resetting the entire adhesive system."""
        from glue.core.adhesive import AdhesiveManager, bind_tool_result
        
        system = AdhesiveManager()
        
        # Bind tool results with different adhesive types
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.web_search_result,
            adhesive_type=AdhesiveType.GLUE
        )
        
        bind_tool_result(
            manager=system,
            team=self.research_team,
            model=self.researcher_model,
            tool_result=self.code_result,
            adhesive_type=AdhesiveType.VELCRO
        )
        
        bind_tool_result(
            manager=system,
            team=self.docs_team,
            model=self.writer_model,
            tool_result=self.file_result,
            adhesive_type=AdhesiveType.TAPE
        )
        
        # Reset the system
        system.reset()
        
        # Verify all storage is empty
        assert len(system.glue_storage) == 0
        assert len(system.velcro_storage) == 0
        assert len(system.tape_storage) == 0
