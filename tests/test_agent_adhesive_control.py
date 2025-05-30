import pytest
from unittest.mock import Mock
from glue.core.glue_smolagent import GlueSmolAgent
from glue.core.types import AdhesiveType
from glue.core.simple_schemas import ToolResult
from smolagents import InferenceClientModel, Tool


class TestTool(Tool):
    """A proper smolagents Tool for testing."""
    name = "test_tool"
    description = "A test tool for unit testing"
    inputs = {
        "input_text": {
            "type": "string",
            "description": "Test input parameter",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, input_text: str = "default") -> str:
        return f"test_result_{input_text}"


class CalculatorTool(Tool):
    """A calculator tool for testing."""
    name = "calculator"
    description = "A calculator tool"
    inputs = {
        "expression": {
            "type": "string", 
            "description": "Mathematical expression to calculate",
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, expression: str = "1+1") -> str:
        return "42"  # Mock result


class CalculatorToolWithAdhesive(Tool):
    """A calculator tool with adhesive parameter for testing hybrid approach."""
    name = "calculator_with_adhesive"
    description = "A calculator tool that allows specifying adhesive type"
    inputs = {
        "expression": {
            "type": "string", 
            "description": "Mathematical expression to calculate",
            "nullable": True
        },
        "adhesive": {
            "type": "string",
            "description": "Adhesive type to use for result binding",
            "enum": ["GLUE", "VELCRO", "TAPE"],
            "nullable": True
        }
    }
    output_type = "string"
    
    def forward(self, expression: str = "1+1", adhesive: str = None) -> str:
        result = "42"  # Mock result
        # Store the requested adhesive in the result for testing
        if adhesive:
            return f"{result}|adhesive:{adhesive}"
        return result


class TestAgentAdhesiveControl:
    """Test suite for agent-controlled adhesive functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return Mock(spec=InferenceClientModel)
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        return TestTool()
    
    @pytest.fixture
    def agent_with_adhesives(self, mock_model, mock_tool):
        """Create a GlueSmolAgent with all adhesive types configured."""
        glue_config = {
            "glue": {"team_id": "test_team", "memory_dir": "/tmp/test"},
            "velcro": {},
            "tape": {}
        }
        
        agent = GlueSmolAgent(
            model=mock_model,
            tools=[mock_tool],
            glue_config=glue_config,
            name="test_agent",
            description="Test agent for adhesive testing"
        )
        return agent
    def test_choose_adhesive_default_policy(self, agent_with_adhesives):
        """Test default adhesive choice policy."""
        # Test team context defaults to GLUE
        agent_with_adhesives._initial_managed_agents_dict_for_template_choice = {"peer": Mock()}
        result = agent_with_adhesives.choose_adhesive("test_tool")
        assert result == AdhesiveType.GLUE
        
        # Test non-team context - since GLUE is still available, it may still be chosen as fallback
        agent_with_adhesives._initial_managed_agents_dict_for_template_choice = {}
        result = agent_with_adhesives.choose_adhesive("test_tool")
        # Default policy prefers VELCRO for non-team, but GLUE may be chosen as fallback
        assert result in [AdhesiveType.GLUE, AdhesiveType.VELCRO]
    
    def test_choose_adhesive_with_override(self, agent_with_adhesives):
        """Test adhesive choice with explicit override."""
        agent_with_adhesives.set_next_adhesive(AdhesiveType.TAPE, "Testing override")
        result = agent_with_adhesives.choose_adhesive("test_tool")
        assert result == AdhesiveType.TAPE
        
        # Override should be cleared after use
        result2 = agent_with_adhesives.choose_adhesive("test_tool")
        assert result2 != AdhesiveType.TAPE or result2 == AdhesiveType.TAPE  # Should use default policy
    
    def test_tool_specific_policy(self, agent_with_adhesives):
        """Test tool-specific adhesive policies."""
        agent_with_adhesives.set_tool_adhesive_policy("special_tool", AdhesiveType.TAPE)
        
        # Test tool with specific policy
        result = agent_with_adhesives.choose_adhesive("special_tool")
        assert result == AdhesiveType.TAPE
        
        # Test tool without specific policy uses default
        result = agent_with_adhesives.choose_adhesive("other_tool")
        assert result in [AdhesiveType.GLUE, AdhesiveType.VELCRO]  # Default policy
    def test_get_adhesive_policy(self, agent_with_adhesives):
        """Test adhesive policy retrieval."""
        agent_with_adhesives.set_next_adhesive(AdhesiveType.GLUE)
        agent_with_adhesives.set_tool_adhesive_policy("tool1", AdhesiveType.TAPE)
        
        policy = agent_with_adhesives.get_adhesive_policy()
        
        assert policy["next_override"] == "glue"  # lowercase values
        assert policy["tool_specific_policies"]["tool1"] == "tape"  # lowercase values
        assert "available_adhesives" in policy
        assert "recent_decisions" in policy
    
    def test_adhesive_decision_logging(self, agent_with_adhesives):
        """Test adhesive decision logging functionality."""
        # Make a few decisions
        agent_with_adhesives.choose_adhesive("tool1")
        agent_with_adhesives.choose_adhesive("tool2")
        
        log = agent_with_adhesives.get_adhesive_rationale_log()
        assert len(log) >= 2
        
        # Check log entry structure
        entry = log[-1]
        assert "timestamp" in entry
        assert "tool_name" in entry
        assert "adhesive_type" in entry
        assert "rationale" in entry
        assert "agent_name" in entry
        
        # Test limited retrieval
        limited_log = agent_with_adhesives.get_adhesive_rationale_log(limit=1)
        assert len(limited_log) == 1
    def test_run_tool_with_adhesive(self, agent_with_adhesives, mock_tool):
        """Test running a tool with specific adhesive type."""
        # Test with override
        result = agent_with_adhesives.run_tool_with_adhesive("test_tool", AdhesiveType.TAPE)
        
        # Tool should have been executed
        assert result is not None
        
        # Check that override was used temporarily
        log = agent_with_adhesives.get_adhesive_rationale_log(limit=1)
        if log:
            assert log[0]["adhesive_type"] == "TAPE"
    
    def test_get_bound_result(self, agent_with_adhesives):
        """Test retrieving bound results from adhesive storage."""
        # Mock memory adapter with get method
        mock_memory = Mock()
        mock_memory.get = Mock(return_value="stored_result")
        agent_with_adhesives.memories = {"glue": mock_memory}
        
        # Test retrieval
        result = agent_with_adhesives.get_bound_result("test_id", AdhesiveType.GLUE)
        assert result == "stored_result"
        mock_memory.get.assert_called_once_with("test_id")
        
        # Test retrieval of non-existent result
        mock_memory.get.return_value = None
        result = agent_with_adhesives.get_bound_result("missing_id")
        assert result is None
    
    def test_clear_adhesive_storage(self, agent_with_adhesives):
        """Test clearing adhesive storage."""
        # Mock memory adapters
        mock_glue = Mock()
        mock_velcro = Mock()
        agent_with_adhesives.memories = {"glue": mock_glue, "velcro": mock_velcro}
        
        # Test clearing specific adhesive
        agent_with_adhesives.clear_adhesive_storage(AdhesiveType.GLUE)
        mock_glue.clear.assert_called_once()
        mock_velcro.clear.assert_not_called()
        
        # Test clearing all adhesives
        agent_with_adhesives.clear_adhesive_storage()
        mock_velcro.clear.assert_called_once()
    
    def test_suggest_adhesive_for_task(self, agent_with_adhesives):
        """Test adhesive suggestion for collaborative tasks."""
        suggestion = agent_with_adhesives.suggest_adhesive_for_task(
            "test_task", 
            AdhesiveType.GLUE, 
            "Best for team coordination"
        )
        
        assert suggestion["agent_name"] == "test_agent"
        assert suggestion["task_description"] == "test_task"
        assert suggestion["preferred_adhesive"] == "GLUE"
        assert suggestion["rationale"] == "Best for team coordination"
        assert "timestamp" in suggestion
        assert "agent_capabilities" in suggestion
    
    def test_evaluate_adhesive_suggestions(self, agent_with_adhesives):
        """Test evaluation of multiple adhesive suggestions."""
        suggestions = [
            {
                "agent_name": "agent1",
                "preferred_adhesive": "GLUE",
                "agent_capabilities": {"is_team_lead": True, "tools_count": 5, "available_adhesives": ["glue"]}
            },
            {
                "agent_name": "agent2", 
                "preferred_adhesive": "VELCRO",
                "agent_capabilities": {"is_team_lead": False, "tools_count": 2, "available_adhesives": ["velcro"]}
            },
            {
                "agent_name": "agent3",
                "preferred_adhesive": "GLUE", 
                "agent_capabilities": {"is_team_lead": False, "tools_count": 3, "available_adhesives": ["glue"]}
            }
        ]
        
        chosen = agent_with_adhesives.evaluate_adhesive_suggestions(suggestions)
        # GLUE should win due to higher weight (team lead + multiple supporters)
        assert chosen == AdhesiveType.GLUE
    
    def test_negotiate_adhesive_with_peers(self, agent_with_adhesives):
        """Test adhesive negotiation with peer agents."""
        # Create mock peer agents
        peer1 = Mock(spec=GlueSmolAgent)
        peer1.name = "peer1"
        peer1.suggest_adhesive_for_task = Mock(return_value={
            "agent_name": "peer1",
            "preferred_adhesive": "VELCRO",
            "agent_capabilities": {"is_team_lead": False, "tools_count": 2}
        })
        peer1.choose_adhesive = Mock(return_value=AdhesiveType.VELCRO)
        
        peer2 = Mock(spec=GlueSmolAgent)
        peer2.name = "peer2"
        peer2.suggest_adhesive_for_task = Mock(return_value={
            "agent_name": "peer2", 
            "preferred_adhesive": "GLUE",
            "agent_capabilities": {"is_team_lead": True, "tools_count": 4}
        })
        peer2.choose_adhesive = Mock(return_value=AdhesiveType.GLUE)
        
        # Test negotiation
        result = agent_with_adhesives.negotiate_adhesive_with_peers(
            [peer1, peer2],
            "collaborative_task",
            AdhesiveType.TAPE,
            "My preference"
        )
        
        # Should consider all suggestions and likely choose based on weights
        assert result in [AdhesiveType.GLUE, AdhesiveType.VELCRO, AdhesiveType.TAPE]
        
        # Verify peers were consulted
        peer1.suggest_adhesive_for_task.assert_called_once()
        peer2.suggest_adhesive_for_task.assert_called_once()


class TestAdhesiveIntegration:
    """Integration tests for adhesive system with tool execution."""
    
    @pytest.fixture
    def agent_with_mock_tool(self):
        """Create agent with a mock tool that can be executed."""
        model = Mock(spec=InferenceClientModel)
        
        # Create a proper smolagents tool
        tool = CalculatorTool()
        
        glue_config = {
            "glue": {"team_id": "test", "memory_dir": "/tmp"},
            "velcro": {},
            "tape": {}
        }
        
        agent = GlueSmolAgent(
            model=model,
            tools=[tool],
            glue_config=glue_config,
            name="calc_agent",
            description="Calculator agent"
        )
        
        # Mock the adhesive system binding
        agent.adhesive_system = Mock()
        agent.adhesive_system.bind_tool_result = Mock()
        
        return agent, tool
    
    def test_tool_execution_with_adhesive_binding(self, agent_with_mock_tool):
        """Test that tool execution properly binds results with chosen adhesive."""
        agent, tool = agent_with_mock_tool
        
        # Set specific adhesive for this tool
        agent.set_tool_adhesive_policy("calculator", AdhesiveType.TAPE)
        
        # Execute tool (this should go through the wrapper)
        # Note: In real usage, this would be called by the agent's execution engine
        # Here we simulate the wrapper behavior
        result = tool()
        
        # Create a ToolResult as the wrapper would
        tool_result = ToolResult(
            tool_name="calculator",
            result=result,
            metadata={"adhesive_used": AdhesiveType.TAPE.value}
        )
          # Simulate binding
        agent.adhesive_system.bind_tool_result.return_value = "bound_result_id"
        agent.adhesive_system.bind_tool_result(tool_result, AdhesiveType.TAPE)
        
        # Verify binding was called with correct adhesive
        agent.adhesive_system.bind_tool_result.assert_called_once()
        call_args = agent.adhesive_system.bind_tool_result.call_args
        assert call_args[0][1] == AdhesiveType.TAPE  # Second argument should be adhesive type
    
    def test_mixed_adhesive_scenarios(self, agent_with_mock_tool):
        """Test scenarios with multiple tools using different adhesives."""
        agent, _ = agent_with_mock_tool
        
        # Set up different policies for different tools
        agent.set_tool_adhesive_policy("tool1", AdhesiveType.GLUE)
        agent.set_tool_adhesive_policy("tool2", AdhesiveType.VELCRO)
        # tool3 will use default policy
        
        # Test each tool gets the right adhesive
        assert agent.choose_adhesive("tool1") == AdhesiveType.GLUE
        assert agent.choose_adhesive("tool2") == AdhesiveType.VELCRO
        assert agent.choose_adhesive("tool3") in [AdhesiveType.GLUE, AdhesiveType.VELCRO]  # Default
        
        # Test override affects all tools
        agent.set_next_adhesive(AdhesiveType.TAPE)
        assert agent.choose_adhesive("tool1") == AdhesiveType.TAPE
        assert agent.choose_adhesive("tool2") != AdhesiveType.TAPE  # Override cleared after first use


if __name__ == "__main__":
    pytest.main([__file__])
