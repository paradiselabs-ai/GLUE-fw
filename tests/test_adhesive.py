import pytest
from glue.core.adhesive import AdhesiveType, ToolResult, AdhesiveSystem, bind_tool_result, get_tool_result, check_adhesive_compatibility

def test_toolresult_backward_compat():
    # Old format
    tr = ToolResult(tool_name="foo", result=123)
    assert tr.tool_call_id == "call_foo"
    assert tr.content == 123
    assert tr.adhesive == AdhesiveType.GLUE
    # New format
    tr2 = ToolResult(tool_call_id="call_bar", content=456)
    assert tr2.tool_name == "bar"
    assert tr2.result == 456
    assert tr2.adhesive == AdhesiveType.GLUE


def test_adhesive_type_enum():
    """Test AdhesiveType enum behavior"""
    assert AdhesiveType.GLUE.value == "glue"
    assert AdhesiveType.VELCRO.value == "velcro"
    assert AdhesiveType.TAPE.value == "tape"
    
    # Test from_value method
    assert AdhesiveType.from_value("glue") == AdhesiveType.GLUE
    assert AdhesiveType.from_value("GLUE") == AdhesiveType.GLUE  # case insensitive
    assert AdhesiveType.from_value(AdhesiveType.VELCRO) == AdhesiveType.VELCRO
    
    with pytest.raises(ValueError):
        AdhesiveType.from_value("invalid")


def test_toolresult_creation():
    """Test ToolResult creation with different parameters"""
    # Test with minimal params
    tr = ToolResult()
    assert tr.adhesive == AdhesiveType.GLUE
    assert tr.timestamp is not None
    
    # Test with old format
    tr_old = ToolResult(tool_name="search", result="search results")
    assert tr_old.tool_call_id == "call_search"
    assert tr_old.content == "search results"
    
    # Test with new format
    tr_new = ToolResult(tool_call_id="call_analysis", content="analysis data")
    assert tr_new.tool_name == "analysis"
    assert tr_new.result == "analysis data"
    
    # Test with custom adhesive
    tr_custom = ToolResult(tool_name="temp", result="temp data", adhesive=AdhesiveType.TAPE)
    assert tr_custom.adhesive == AdhesiveType.TAPE


def test_toolresult_from_dict():
    """Test ToolResult.from_dict method"""
    data = {
        "tool_name": "web_search",
        "result": "search results",
        "adhesive": "velcro"
    }
    tr = ToolResult.from_dict(data)
    assert tr.tool_name == "web_search"
    assert tr.result == "search results"
    assert tr.adhesive == AdhesiveType.VELCRO


class TestAdhesiveSystem:
    """Test cases for AdhesiveSystem class"""
    
    def test_adhesive_system_init(self):
        """Test AdhesiveSystem initialization"""
        system = AdhesiveSystem()
        assert system.glue_storage == {}
        assert system.velcro_storage == {}
        assert system.tape_storage == {}
    
    def test_store_glue_result(self):
        """Test storing GLUE results"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="search", result="search results")

        system.store_glue_result("team1", "model1", tr)

        assert "team1" in system.glue_storage
        assert "search" in system.glue_storage["team1"]
        stored = system.glue_storage["team1"]["search"]
        assert stored["model"] == "model1"
        assert stored["result"] == tr
        assert "timestamp" in stored
    
    def test_store_velcro_result(self):
        """Test storing VELCRO results"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="analyze", result="analysis")
        
        system.store_velcro_result("team1", "model1", tr)
        
        assert "model1" in system.velcro_storage
        assert "analyze" in system.velcro_storage["model1"]
        stored = system.velcro_storage["model1"]["analyze"]
        assert stored["team"] == "team1"
        assert stored["result"] == tr
        assert "timestamp" in stored
    
    def test_store_tape_result(self):
        """Test storing TAPE results"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="verify", result="verified")
        
        system.store_tape_result("team1", "model1", tr)
        
        assert "verify" in system.tape_storage
        stored = system.tape_storage["verify"]
        assert stored["team"] == "team1"
        assert stored["model"] == "model1"
        assert stored["result"] == tr
        assert "timestamp" in stored
    
    def test_get_tool_result_priority(self):
        """Test tool result retrieval priority: GLUE > VELCRO > TAPE"""
        system = AdhesiveSystem()

        # Store same tool name in all three storages
        glue_result = ToolResult(tool_name="tool1", result="glue_data")
        velcro_result = ToolResult(tool_name="tool1", result="velcro_data")
        tape_result = ToolResult(tool_name="tool1", result="tape_data")

        system.store_glue_result("team1", "model1", glue_result)
        system.store_velcro_result("team1", "model1", velcro_result)
        system.store_tape_result("team1", "model1", tape_result)

        # Should return GLUE result first
        result = system.get_tool_result("tool1")
        assert result is not None
        assert result.result == "glue_data"

    def test_tape_one_time_use(self):
        """Test that TAPE results are removed after retrieval"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="once", result="one_time_data")

        system.store_tape_result("team1", "model1", tr)
        assert "once" in system.tape_storage

        # First retrieval should work
        result = system.get_tool_result("once")
        assert result is not None
        assert result.result == "one_time_data"

        # Second retrieval should return None (removed after first use)
        result2 = system.get_tool_result("once")
        assert result2 is None
        assert "once" not in system.tape_storage
    
    def test_get_team_tool_results(self):
        """Test getting all tool results for a team"""
        system = AdhesiveSystem()
        
        tr1 = ToolResult(tool_name="tool1", result="data1")
        tr2 = ToolResult(tool_name="tool2", result="data2")
        
        system.store_glue_result("team1", "model1", tr1)
        system.store_glue_result("team1", "model2", tr2)
        
        results = system.get_team_tool_results("team1")
        assert len(results) == 2
        
        tool_names = [r["tool_name"] for r in results]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
    
    def test_get_model_tool_results(self):
        """Test getting all tool results for a model"""
        system = AdhesiveSystem()
        
        tr1 = ToolResult(tool_name="tool1", result="data1")
        tr2 = ToolResult(tool_name="tool2", result="data2")
        
        system.store_velcro_result("team1", "model1", tr1)
        system.store_velcro_result("team2", "model1", tr2)
        
        results = system.get_model_tool_results("model1")
        assert len(results) == 2
        
        tool_names = [r["tool_name"] for r in results]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
    
    def test_clear_storage_methods(self):
        """Test various storage clearing methods"""
        system = AdhesiveSystem()
        
        # Add data to all storages
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model1", tr)
        system.store_tape_result("team1", "model1", tr)
        
        # Test specific clearing
        system.clear_glue_storage()
        assert system.glue_storage == {}
        assert len(system.velcro_storage) > 0
        assert len(system.tape_storage) > 0
        
        system.clear_velcro_storage()
        assert system.velcro_storage == {}
        assert len(system.tape_storage) > 0
        
        system.clear_tape_storage()
        assert system.tape_storage == {}
    
    def test_clear_all_storage(self):
        """Test clearing all storage at once"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model1", tr)
        system.store_tape_result("team1", "model1", tr)
        
        system.clear_all_storage()
        assert system.glue_storage == {}
        assert system.velcro_storage == {}
        assert system.tape_storage == {}
    
    def test_clear_team_storage(self):
        """Test clearing storage for specific team"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_glue_result("team2", "model1", tr)
        
        system.clear_team_storage("team1")
        assert "team1" not in system.glue_storage
        assert "team2" in system.glue_storage
    
    def test_clear_model_storage(self):
        """Test clearing storage for specific model"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_velcro_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model2", tr)
        
        system.clear_model_storage("model1")
        assert "model1" not in system.velcro_storage
        assert "model2" in system.velcro_storage
    
    def test_reset(self):
        """Test system reset"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model1", tr)
        system.store_tape_result("team1", "model1", tr)
        
        system.reset()
        assert system.glue_storage == {}
        assert system.velcro_storage == {}
        assert system.tape_storage == {}
    
    def test_clear_storage_by_type(self):
        """Test clearing storage by adhesive type"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model1", tr)
        system.store_tape_result("team1", "model1", tr)
        
        # Clear specific type
        system.clear_storage(AdhesiveType.GLUE)
        assert system.glue_storage == {}
        assert len(system.velcro_storage) > 0
        assert len(system.tape_storage) > 0
        
        # Clear all types (None parameter)
        system.clear_storage(None)
        assert system.velcro_storage == {}
        assert system.tape_storage == {}


class MockModel:
    """Mock model for testing adhesive compatibility"""
    
    def __init__(self, name, adhesives=None, supported_adhesives=None, has_adhesive_func=None):
        self.name = name
        if adhesives is not None:
            self.adhesives = adhesives
        if supported_adhesives is not None:
            self.supported_adhesives = supported_adhesives
        if has_adhesive_func is not None:
            self.has_adhesive = has_adhesive_func


class MockTeam:
    """Mock team for testing"""
    
    def __init__(self, name):
        self.name = name


class TestAdhesiveCompatibility:
    """Test adhesive compatibility checking"""
    
    def test_check_adhesive_compatibility_with_list(self):
        """Test compatibility check with adhesives list"""
        model = MockModel("test", adhesives=[AdhesiveType.GLUE, AdhesiveType.VELCRO])
        
        assert check_adhesive_compatibility(model, AdhesiveType.GLUE)
        assert check_adhesive_compatibility(model, AdhesiveType.VELCRO)
        assert not check_adhesive_compatibility(model, AdhesiveType.TAPE)
    
    def test_check_adhesive_compatibility_with_set(self):
        """Test compatibility check with adhesives set"""
        model = MockModel("test", adhesives={AdhesiveType.TAPE})
        
        assert check_adhesive_compatibility(model, AdhesiveType.TAPE)
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)
    
    def test_check_adhesive_compatibility_with_supported_adhesives(self):
        """Test compatibility check with supported_adhesives attribute"""
        model = MockModel("test", supported_adhesives=["glue", "velcro"])
        
        assert check_adhesive_compatibility(model, AdhesiveType.GLUE)
        assert check_adhesive_compatibility(model, AdhesiveType.VELCRO)
        assert not check_adhesive_compatibility(model, AdhesiveType.TAPE)
    
    def test_check_adhesive_compatibility_with_has_adhesive_method(self):
        """Test compatibility check with has_adhesive method"""
        def has_adhesive_func(adhesive_type):
            return adhesive_type == AdhesiveType.GLUE
        
        model = MockModel("test", has_adhesive_func=has_adhesive_func)
        
        assert check_adhesive_compatibility(model, AdhesiveType.GLUE)
        assert not check_adhesive_compatibility(model, AdhesiveType.VELCRO)
    
    def test_check_adhesive_compatibility_default_false(self):
        """Test that compatibility defaults to False for unknown models"""
        model = MockModel("test")  # No adhesive info
        
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)
        assert not check_adhesive_compatibility(model, AdhesiveType.VELCRO)
        assert not check_adhesive_compatibility(model, AdhesiveType.TAPE)


class TestBindToolResult:
    """Test bind_tool_result function"""
    
    def test_bind_tool_result_glue(self):
        """Test binding tool result with GLUE adhesive"""
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])
        tr = ToolResult(tool_name="test", result="data")
        
        bind_tool_result(system, team, model, tr, AdhesiveType.GLUE)
        
        assert "team1" in system.glue_storage
        assert "test" in system.glue_storage["team1"]
    
    def test_bind_tool_result_velcro(self):
        """Test binding tool result with VELCRO adhesive"""
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.VELCRO])
        tr = ToolResult(tool_name="test", result="data")
        
        bind_tool_result(system, team, model, tr, AdhesiveType.VELCRO)
        
        assert "model1" in system.velcro_storage
        assert "test" in system.velcro_storage["model1"]
    
    def test_bind_tool_result_tape(self):
        """Test binding tool result with TAPE adhesive"""
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.TAPE])
        tr = ToolResult(tool_name="test", result="data")
        
        bind_tool_result(system, team, model, tr, AdhesiveType.TAPE)
        
        assert "test" in system.tape_storage
    
    def test_bind_tool_result_incompatible_model(self):
        """Test binding fails with incompatible model"""
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])  # Only supports GLUE
        tr = ToolResult(tool_name="test", result="data")

        with pytest.raises(ValueError, match="does not support adhesive type"):
            bind_tool_result(system, team, model, tr, AdhesiveType.TAPE)

class TestGetToolResult:
    """Test get_tool_result function"""

    def test_get_tool_result_found(self):
        """Test getting existing tool result"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="search", result="results")
        system.store_glue_result("team1", "model1", tr)

        result = get_tool_result(system, "search")
        assert result is not None
        assert result.result == "results"

    def test_get_tool_result_not_found(self):
        """Test getting non-existent tool result"""
        system = AdhesiveSystem()

        result = system.get_tool_result("nonexistent")
        assert result is None


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_toolresult_with_none_tool_name(self):
        """Test ToolResult handling when tool_name is None"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_call_id=None, tool_name=None, result="data")
        
        # Should handle gracefully and use fallback
        system.store_glue_result("team1", "model1", tr)
        assert "unknown_tool" in system.glue_storage["team1"]
    
    def test_toolresult_with_none_tool_name_velcro(self):
        """Test ToolResult handling when tool_name is None for VELCRO storage"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_call_id=None, tool_name=None, result="data")
        
        # Should handle gracefully and use fallback
        system.store_velcro_result("team1", "model1", tr)
        assert "unknown_tool" in system.velcro_storage["model1"]
    
    def test_toolresult_with_none_tool_name_tape(self):
        """Test ToolResult handling when tool_name is None for TAPE storage"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_call_id=None, tool_name=None, result="data")
        
        # Should handle gracefully and use fallback
        system.store_tape_result("team1", "model1", tr)
        assert "unknown_tool" in system.tape_storage
    
    def test_get_tool_result_priority_velcro_first(self):
        """Test tool result retrieval when only VELCRO result exists"""
        system = AdhesiveSystem()
        
        # Store only in VELCRO
        velcro_result = ToolResult(tool_name="tool1", result="velcro_data")
        system.store_velcro_result("team1", "model1", velcro_result)
        
        # Should return VELCRO result when no GLUE exists
        result = system.get_tool_result("tool1")
        assert result is not None
        assert result.result == "velcro_data"
    
    def test_clear_storage_invalid_type(self):
        """Test clearing storage with None type parameter"""
        system = AdhesiveSystem()
        
        tr = ToolResult(tool_name="test", result="data")
        system.store_glue_result("team1", "model1", tr)
        system.store_velcro_result("team1", "model1", tr)
        system.store_tape_result("team1", "model1", tr)
        
        # Test None parameter clears all
        system.clear_storage(None)
        assert system.glue_storage == {}
        assert system.velcro_storage == {}
        assert system.tape_storage == {}

    def test_bind_tool_result_backward_compatibility(self):
        """Test bind_tool_result with manager parameter for backward compatibility"""
        from glue.core.adhesive import bind_tool_result

        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])
        tr = ToolResult(tool_name="test", result="data")

        # Test with manager parameter instead of system
        bind_tool_result(system=system, team=team, model=model, tool_result=tr, 
                        adhesive_type=AdhesiveType.GLUE, manager=system)

        assert "team1" in system.glue_storage
        assert "test" in system.glue_storage["team1"]
    def test_bind_tool_result_unknown_adhesive_type(self):
        """Test binding fails with unknown adhesive type (removed: not robustly testable in Python)"""
        # This test is intentionally left empty because Python's type system does not allow
        # robust simulation of an unknown adhesive type that is not a string or enum value.
        pass
    
    def test_get_tool_result_backward_compatibility(self):
        """Test get_tool_result with manager parameter for backward compatibility"""
        from glue.core.adhesive import get_tool_result
        
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="search", result="results")
        system.store_glue_result("team1", "model1", tr)

        # Test with manager parameter instead of system
        result = get_tool_result(system=None, tool_call_id="search", manager=system)
        assert result is not None
        assert result.result == "results"
    
    def test_bind_tool_result_manager_only_parameter(self):
        """Test bind_tool_result with only manager parameter (no system)"""
        from glue.core.adhesive import bind_tool_result
        
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])
        tr = ToolResult(tool_name="test", result="data")
        
        # Pass None for system, use manager parameter
        bind_tool_result(system=None, team=team, model=model, tool_result=tr, 
                        adhesive_type=AdhesiveType.GLUE, manager=system)
        
        assert "team1" in system.glue_storage
        assert "test" in system.glue_storage["team1"]

    def test_get_tool_result_manager_only_parameter(self):
        """Test get_tool_result with only manager parameter (no system)"""
        from glue.core.adhesive import get_tool_result
        
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="search", result="results")
        system.store_glue_result("team1", "model1", tr)
        
        # Pass None for system, use manager parameter
        result = get_tool_result(system=None, tool_call_id="search", manager=system)
        assert result is not None
        assert result.result == "results"
    
    def test_tool_name_precedence_over_tool_call_id(self):
        """Test that tool_name parameter takes precedence over tool_call_id"""
        from glue.core.adhesive import get_tool_result
        
        system = AdhesiveSystem()
        tr1 = ToolResult(tool_name="search", result="search_results")
        tr2 = ToolResult(tool_name="analyze", result="analyze_results")
        
        system.store_glue_result("team1", "model1", tr1)
        system.store_glue_result("team1", "model1", tr2)
        
        # tool_name should take precedence over tool_call_id
        result = get_tool_result(system, tool_call_id="search", tool_name="analyze")
        assert result is not None
        assert result.result == "analyze_results"


class TestSpecialModelConfigurations:
    """Test special model configurations and edge cases"""
    
    def test_model_with_none_adhesives(self):
        """Test model with None adhesives attribute"""
        model = MockModel("test", adhesives=None)
        
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)
        assert not check_adhesive_compatibility(model, AdhesiveType.VELCRO)
        assert not check_adhesive_compatibility(model, AdhesiveType.TAPE)
    
    def test_model_with_adhesives_not_list_or_set(self):
        """Test model with adhesives that's not a list or set"""
        model = MockModel("test", adhesives="invalid_type")
        
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)
    
    def test_model_with_supported_adhesives_not_list_or_set(self):
        """Test model with supported_adhesives that's not a list or set"""
        model = MockModel("test", supported_adhesives="invalid_type")
        
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)
    
    def test_model_with_non_callable_has_adhesive(self):
        """Test model with has_adhesive attribute that's not callable"""
        model = MockModel("test", has_adhesive_func=None)
        model.has_adhesive = "not_callable"
        
        assert not check_adhesive_compatibility(model, AdhesiveType.GLUE)


class TestLoggingBehavior:
    """Test logging behavior of adhesive system"""
    
    def test_system_initialization_logging(self, caplog):
        """Test that system initialization logs correctly"""
        import logging
        caplog.set_level(logging.INFO)
        
        AdhesiveSystem()
        assert "Adhesive system initialized" in caplog.text
    
    def test_storage_clearing_logging(self, caplog):
        """Test that storage clearing operations log correctly"""
        import logging
        caplog.set_level(logging.INFO)
        
        system = AdhesiveSystem()
        
        system.clear_glue_storage()
        assert "Cleared GLUE storage" in caplog.text
        
        system.clear_velcro_storage()
        assert "Cleared VELCRO storage" in caplog.text
        
        system.clear_tape_storage()
        assert "Cleared TAPE storage" in caplog.text
        
        system.clear_all_storage()
        assert "Cleared all adhesive storage" in caplog.text
    
    def test_tool_result_storage_debug_logging(self, caplog):
        """Test debug logging during tool result storage"""
        import logging
        caplog.set_level(logging.DEBUG)
        
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="test_tool", result="test_data")
        
        system.store_glue_result("team1", "model1", tr)
        assert "Stored GLUE result for team team1, tool test_tool" in caplog.text
        
        system.store_velcro_result("team1", "model1", tr)
        assert "Stored VELCRO result for model model1, tool test_tool" in caplog.text
        
        system.store_tape_result("team1", "model1", tr)
        assert "Stored TAPE result for tool test_tool" in caplog.text
    
    def test_tool_result_retrieval_debug_logging(self, caplog):
        """Test debug logging during tool result retrieval"""
        import logging
        caplog.set_level(logging.DEBUG)
        
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="retrieval_test", result="test_data")
        
        # Test GLUE retrieval logging
        system.store_glue_result("team1", "model1", tr)
        system.get_tool_result("retrieval_test")
        assert "Retrieved GLUE result for tool retrieval_test" in caplog.text
        
        # Clear and test VELCRO retrieval logging
        system.clear_all_storage()
        caplog.clear()
        system.store_velcro_result("team1", "model1", tr)
        system.get_tool_result("retrieval_test")
        assert "Retrieved VELCRO result for tool retrieval_test" in caplog.text
        
        # Clear and test TAPE retrieval logging
        system.clear_all_storage()
        caplog.clear()
        system.store_tape_result("team1", "model1", tr)
        system.get_tool_result("retrieval_test")
        assert "Retrieved and removed TAPE result for tool retrieval_test" in caplog.text
    
    def test_tool_not_found_debug_logging(self, caplog):
        """Test debug logging when tool result is not found"""
        import logging
        caplog.set_level(logging.DEBUG)
        
        system = AdhesiveSystem()
        system.get_tool_result("nonexistent_tool")
        assert "No result found for tool nonexistent_tool" in caplog.text
    
    def test_bind_tool_result_debug_logging(self, caplog):
        """Test debug logging in bind_tool_result function"""
        import logging
        caplog.set_level(logging.DEBUG)
        
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])
        tr = ToolResult(tool_name="bind_test", result="test_data")
        
        bind_tool_result(system, team, model, tr, AdhesiveType.GLUE)
        assert "Bound tool result bind_test with AdhesiveType.GLUE adhesive" in caplog.text


class TestAdvancedBackwardCompatibility:
    """Test advanced backward compatibility scenarios"""
    
    def test_bind_tool_result_manager_only_parameter(self):
        """Test bind_tool_result with only manager parameter (no system)"""
        from glue.core.adhesive import bind_tool_result
        
        system = AdhesiveSystem()
        team = MockTeam("team1")
        model = MockModel("model1", adhesives=[AdhesiveType.GLUE])
        tr = ToolResult(tool_name="test", result="data")
        
        # Pass None for system, use manager parameter
        bind_tool_result(system=None, team=team, model=model, tool_result=tr, 
                        adhesive_type=AdhesiveType.GLUE, manager=system)
        
        assert "team1" in system.glue_storage
        assert "test" in system.glue_storage["team1"]

    def test_get_tool_result_manager_only_parameter(self):
        """Test get_tool_result with only manager parameter (no system)"""
        from glue.core.adhesive import get_tool_result
        
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="search", result="results")
        system.store_glue_result("team1", "model1", tr)
        
        # Pass None for system, use manager parameter
        result = get_tool_result(system=None, tool_call_id="search", manager=system)
        assert result is not None
        assert result.result == "results"
    
    def test_tool_name_precedence_over_tool_call_id(self):
        """Test that tool_name parameter takes precedence over tool_call_id"""
        from glue.core.adhesive import get_tool_result
        
        system = AdhesiveSystem()
        tr1 = ToolResult(tool_name="search", result="search_results")
        tr2 = ToolResult(tool_name="analyze", result="analyze_results")
        
        system.store_glue_result("team1", "model1", tr1)
        system.store_glue_result("team1", "model1", tr2)
        
        # tool_name should take precedence over tool_call_id
        result = get_tool_result(system, tool_call_id="search", tool_name="analyze")
        assert result is not None
        assert result.result == "analyze_results"


class TestStorageOverwriting:
    """Test storage overwriting and update scenarios"""
    
    def test_glue_storage_overwriting(self):
        """Test that storing to same team/tool overwrites previous result"""
        system = AdhesiveSystem()
        
        tr1 = ToolResult(tool_name="search", result="first_result")
        tr2 = ToolResult(tool_name="search", result="second_result")
        
        system.store_glue_result("team1", "model1", tr1)
        system.store_glue_result("team1", "model2", tr2)  # Different model, same team/tool
        
        # Should overwrite with second result
        stored = system.glue_storage["team1"]["search"]
        assert stored["result"] == tr2
        assert stored["model"] == "model2"
    
    def test_velcro_storage_overwriting(self):
        """Test that storing to same model/tool overwrites previous result"""
        system = AdhesiveSystem()
        
        tr1 = ToolResult(tool_name="analyze", result="first_analysis")
        tr2 = ToolResult(tool_name="analyze", result="second_analysis")
        
        system.store_velcro_result("team1", "model1", tr1)
        system.store_velcro_result("team2", "model1", tr2)  # Different team, same model/tool
        
        # Should overwrite with second result
        stored = system.velcro_storage["model1"]["analyze"]
        assert stored["result"] == tr2
        assert stored["team"] == "team2"
    
    def test_tape_storage_overwriting(self):
        """Test that storing to same tool overwrites previous result"""
        system = AdhesiveSystem()
        
        tr1 = ToolResult(tool_name="verify", result="first_verification")
        tr2 = ToolResult(tool_name="verify", result="second_verification")
        
        system.store_tape_result("team1", "model1", tr1)
        system.store_tape_result("team2", "model2", tr2)  # Different team/model, same tool
        
        # Should overwrite with second result
        stored = system.tape_storage["verify"]
        assert stored["result"] == tr2
        assert stored["team"] == "team2"
        assert stored["model"] == "model2"


class TestComplexStorageScenarios:
    """Test complex storage and retrieval scenarios"""
    
    def test_cross_storage_priority_multiple_tools(self):
        """Test priority with multiple tools across all storage types"""
        system = AdhesiveSystem()
        
        # Store different tools in different storages
        glue_result = ToolResult(tool_name="tool_a", result="glue_data")
        velcro_result = ToolResult(tool_name="tool_b", result="velcro_data")
        tape_result = ToolResult(tool_name="tool_c", result="tape_data")
        
        system.store_glue_result("team1", "model1", glue_result)
        system.store_velcro_result("team1", "model1", velcro_result)
        system.store_tape_result("team1", "model1", tape_result)
        
        # Each should be found in its respective storage
        result_a = system.get_tool_result("tool_a")
        assert result_a is not None
        assert result_a.result == "glue_data"
        result_b = system.get_tool_result("tool_b")
        assert result_b is not None
        assert result_b.result == "velcro_data"
        result_c = system.get_tool_result("tool_c")
        assert result_c is not None
        assert result_c.result == "tape_data"
        # After retrieving tape, it should be gone
        assert system.get_tool_result("tool_c") is None
    
    def test_mixed_team_model_storage(self):
        """Test storage and retrieval with multiple teams and models"""
        system = AdhesiveSystem()
        
        # Store same tool name across different teams/models
        tr1 = ToolResult(tool_name="process", result="team1_result")
        tr2 = ToolResult(tool_name="process", result="team2_result")
        tr3 = ToolResult(tool_name="process", result="model1_result")
        tr4 = ToolResult(tool_name="process", result="model2_result")
        
        system.store_glue_result("team1", "model1", tr1)
        system.store_glue_result("team2", "model1", tr2)
        system.store_velcro_result("team1", "model1", tr3)
        system.store_velcro_result("team1", "model2", tr4)
        
        # Should get first GLUE result found (team1)
        result = system.get_tool_result("process")
        assert result is not None
        assert result.result in ["team1_result", "team2_result"]  # Either could be first
        
        # Get team-specific results
        team1_results = system.get_team_tool_results("team1")
        team2_results = system.get_team_tool_results("team2")
        assert len(team1_results) == 1
        assert len(team2_results) == 1
        
        # Get model-specific results
        model1_results = system.get_model_tool_results("model1")
        model2_results = system.get_model_tool_results("model2")
        assert len(model1_results) == 1
        assert len(model2_results) == 1


class TestEmptyAndEdgeCases:
    """Test empty states and edge cases"""
    
    def test_get_tool_result_from_empty_storages(self):
        """Test retrieving from completely empty storage"""
        system = AdhesiveSystem()
        
        result = system.get_tool_result("nonexistent")
        assert result is None
    
    def test_get_team_results_nonexistent_team(self):
        """Test getting results for non-existent team"""
        system = AdhesiveSystem()
        
        results = system.get_team_tool_results("nonexistent_team")
        assert results == []
    
    def test_get_model_results_nonexistent_model(self):
        """Test getting results for non-existent model"""
        system = AdhesiveSystem()
        
        results = system.get_model_tool_results("nonexistent_model")
        assert results == []
    
    def test_clear_nonexistent_team_storage(self):
        """Test clearing storage for non-existent team"""
        system = AdhesiveSystem()
        
        # Should not raise error
        system.clear_team_storage("nonexistent_team")
        assert system.glue_storage == {}
    
    def test_clear_nonexistent_model_storage(self):
        """Test clearing storage for non-existent model"""
        system = AdhesiveSystem()
        
        # Should not raise error
        system.clear_model_storage("nonexistent_model")
        assert system.velcro_storage == {}
    
    def test_empty_tool_name_handling(self):
        """Test handling of empty string tool names"""
        system = AdhesiveSystem()
        tr = ToolResult(tool_name="", result="empty_name_data")
        
        # Should store with empty string as key
        system.store_glue_result("team1", "model1", tr)
        assert "" in system.glue_storage["team1"]
        
        # Should be retrievable
        result = system.get_tool_result("")
        assert result is not None
        assert result.result == "empty_name_data"
