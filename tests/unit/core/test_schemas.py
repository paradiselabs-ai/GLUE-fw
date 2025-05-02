"""
Tests for the Pydantic models in the GLUE framework.

This module contains tests for the Pydantic models defined in the schemas.py module,
ensuring that validation works correctly for all model types.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from glue.core.schemas import (
    AdhesiveType,
    FlowType,
    ToolCall,
    ToolResult,
    Message,
    ModelConfig,
    ToolConfig,
    TeamConfig,
    MagnetConfig,
    AppConfig,
)


class TestAdhesiveType:
    """Tests for the AdhesiveType enum"""

    def test_valid_adhesive_types(self):
        """Test that valid adhesive types are accepted"""
        assert AdhesiveType.GLUE == "glue"
        assert AdhesiveType.VELCRO == "velcro"
        assert AdhesiveType.TAPE == "tape"

    def test_invalid_adhesive_type(self):
        """Test that invalid adhesive types are rejected"""
        with pytest.raises(ValueError):
            AdhesiveType("invalid")


class TestFlowType:
    """Tests for the FlowType enum"""

    def test_valid_flow_types(self):
        """Test that valid flow types are accepted"""
        assert FlowType.PUSH == "push"
        assert FlowType.PULL == "pull"
        assert FlowType.BIDIRECTIONAL == "bidirectional"
        assert FlowType.REPEL == "repel"

    def test_invalid_flow_type(self):
        """Test that invalid flow types are rejected"""
        with pytest.raises(ValueError):
            FlowType("invalid")


class TestToolCall:
    """Tests for the ToolCall model"""

    def test_valid_tool_call(self):
        """Test that a valid tool call is accepted"""
        tool_call = ToolCall(
            tool_id="web_search_1",
            name="web_search",
            arguments={"query": "GLUE framework for AI", "max_results": 5},
        )
        assert tool_call.tool_id == "web_search_1"
        assert tool_call.name == "web_search"
        assert tool_call.arguments["query"] == "GLUE framework for AI"
        assert tool_call.arguments["max_results"] == 5

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            ToolCall(name="web_search")  # Missing tool_id

        with pytest.raises(ValidationError):
            ToolCall(tool_id="web_search_1")  # Missing name


class TestToolResult:
    """Tests for the ToolResult model"""

    def test_valid_tool_result(self):
        """Test that a valid tool result is accepted"""
        tool_result = ToolResult(
            tool_name="web_search",
            result=["Result 1", "Result 2"],
            adhesive=AdhesiveType.GLUE,
        )
        assert tool_result.tool_name == "web_search"
        assert tool_result.result == ["Result 1", "Result 2"]
        assert tool_result.adhesive == AdhesiveType.GLUE
        assert isinstance(tool_result.timestamp, datetime)
        assert isinstance(tool_result.metadata, dict)

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            ToolResult(result=["Result 1"], adhesive=AdhesiveType.GLUE)  # Missing tool_name

        with pytest.raises(ValidationError):
            ToolResult(tool_name="web_search", adhesive=AdhesiveType.GLUE)  # Missing result

        with pytest.raises(ValidationError):
            ToolResult(tool_name="web_search", result=["Result 1"])  # Missing adhesive


class TestMessage:
    """Tests for the Message model"""

    def test_valid_message(self):
        """Test that a valid message is accepted"""
        message = Message(
            role="assistant",
            content="I found the following information about the GLUE framework.",
        )
        assert message.role == "assistant"
        assert message.content == "I found the following information about the GLUE framework."
        assert isinstance(message.tool_calls, list)
        assert isinstance(message.metadata, dict)

    def test_message_with_tool_calls(self):
        """Test a message with tool calls"""
        message = Message(
            role="assistant",
            content="Let me search for information about the GLUE framework.",
            tool_calls=[
                ToolCall(
                    tool_id="web_search_1",
                    name="web_search",
                    arguments={"query": "GLUE framework for AI"},
                )
            ],
        )
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "web_search"

    def test_invalid_role(self):
        """Test that an invalid role raises a validation error"""
        with pytest.raises(ValidationError):
            Message(
                role="invalid_role",
                content="This message has an invalid role.",
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            Message(content="Missing role")  # Missing role

        with pytest.raises(ValidationError):
            Message(role="assistant")  # Missing content


class TestModelConfig:
    """Tests for the ModelConfig model"""

    def test_valid_model_config(self):
        """Test that a valid model config is accepted"""
        model_config = ModelConfig(
            name="gpt4",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048,
        )
        assert model_config.name == "gpt4"
        assert model_config.provider == "openai"
        assert model_config.model == "gpt-4"
        assert model_config.temperature == 0.7
        assert model_config.max_tokens == 2048
        assert model_config.api_key is None
        assert isinstance(model_config.api_params, dict)

    def test_invalid_provider(self):
        """Test that an invalid provider raises a validation error"""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="invalid_provider",
                model="model-1",
            )

    def test_invalid_temperature(self):
        """Test that an invalid temperature raises a validation error"""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=1.5,  # Temperature must be between 0 and 1
            )

        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model="gpt-4",
                temperature=-0.5,  # Temperature must be between 0 and 1
            )

    def test_invalid_max_tokens(self):
        """Test that an invalid max_tokens raises a validation error"""
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model="gpt-4",
                max_tokens=0,  # max_tokens must be greater than 0
            )

        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model="gpt-4",
                max_tokens=-100,  # max_tokens must be greater than 0
            )


class TestToolConfig:
    """Tests for the ToolConfig model"""

    def test_valid_tool_config(self):
        """Test that a valid tool config is accepted"""
        tool_config = ToolConfig(
            name="web_search",
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results"},
            },
            required_permissions=["internet_access"],
        )
        assert tool_config.name == "web_search"
        assert tool_config.description == "Search the web for information"
        assert "query" in tool_config.parameters
        assert "max_results" in tool_config.parameters
        assert tool_config.required_permissions == ["internet_access"]

    def test_minimal_tool_config(self):
        """Test that a minimal tool config is accepted"""
        tool_config = ToolConfig(name="simple_tool")
        assert tool_config.name == "simple_tool"
        assert tool_config.description == ""
        assert isinstance(tool_config.parameters, dict)
        assert isinstance(tool_config.required_permissions, list)

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            ToolConfig()  # Missing name


class TestTeamConfig:
    """Tests for the TeamConfig model"""

    def test_valid_team_config(self):
        """Test that a valid team config is accepted"""
        team_config = TeamConfig(
            name="research_team",
            lead="gpt4_researcher",
            members=["claude_analyst", "gemini_assistant"],
            tools=["web_search", "document_reader"],
        )
        assert team_config.name == "research_team"
        assert team_config.lead == "gpt4_researcher"
        assert team_config.members == ["claude_analyst", "gemini_assistant"]
        assert team_config.tools == ["web_search", "document_reader"]

    def test_minimal_team_config(self):
        """Test that a minimal team config is accepted"""
        team_config = TeamConfig(name="simple_team", lead="leader_model")
        assert team_config.name == "simple_team"
        assert team_config.lead == "leader_model"
        assert isinstance(team_config.members, list)
        assert isinstance(team_config.tools, list)

    def test_lead_in_members(self):
        """Test that having the lead also in members raises a validation error"""
        with pytest.raises(ValidationError):
            TeamConfig(
                name="invalid_team",
                lead="model1",
                members=["model1", "model2"],  # Lead should not be in members
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            TeamConfig(lead="leader_model")  # Missing name

        with pytest.raises(ValidationError):
            TeamConfig(name="team_without_lead")  # Missing lead


class TestMagnetConfig:
    """Tests for the MagnetConfig model"""

    def test_valid_magnet_config(self):
        """Test that a valid magnet config is accepted"""
        magnet_config = MagnetConfig(
            source="research_team",
            target="writing_team",
            flow_type=FlowType.PUSH,
            filters={"topics": ["research_findings", "data_analysis"]},
        )
        assert magnet_config.source == "research_team"
        assert magnet_config.target == "writing_team"
        assert magnet_config.flow_type == FlowType.PUSH
        assert "topics" in magnet_config.filters

    def test_default_flow_type(self):
        """Test that the default flow type is BIDIRECTIONAL"""
        magnet_config = MagnetConfig(source="team1", target="team2")
        assert magnet_config.flow_type == FlowType.BIDIRECTIONAL

    def test_same_source_and_target(self):
        """Test that having the same source and target raises a validation error"""
        with pytest.raises(ValidationError):
            MagnetConfig(
                source="same_team",
                target="same_team",  # Source and target must be different
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            MagnetConfig(target="target_team")  # Missing source

        with pytest.raises(ValidationError):
            MagnetConfig(source="source_team")  # Missing target


class TestAppConfig:
    """Tests for the AppConfig model"""

    def test_valid_app_config(self):
        """Test that a valid app config is accepted"""
        app_config = AppConfig(
            name="Research Assistant",
            description="An AI research assistant application",
            version="0.1.0",
            development=True,
            log_level="info",
            models=[
                ModelConfig(name="gpt4_model", provider="openai", model="gpt-4"),
                ModelConfig(name="claude_model", provider="anthropic", model="claude-3-opus"),
            ],
            tools=[
                ToolConfig(name="web_search", description="Search the web"),
                ToolConfig(name="document_reader", description="Read documents"),
            ],
            teams=[
                TeamConfig(name="research_team", lead="gpt4_model"),
                TeamConfig(name="writing_team", lead="claude_model"),
            ],
            magnets=[
                MagnetConfig(source="research_team", target="writing_team", flow_type=FlowType.PUSH),
            ],
        )
        assert app_config.name == "Research Assistant"
        assert len(app_config.models) == 2
        assert len(app_config.tools) == 2
        assert len(app_config.teams) == 2
        assert len(app_config.magnets) == 1

    def test_minimal_app_config(self):
        """Test that a minimal app config is accepted"""
        app_config = AppConfig(name="Simple App")
        assert app_config.name == "Simple App"
        assert app_config.description == ""
        assert app_config.version == "0.1.0"
        assert app_config.development is True
        assert app_config.log_level == "info"
        assert isinstance(app_config.models, list)
        assert isinstance(app_config.tools, list)
        assert isinstance(app_config.teams, list)
        assert isinstance(app_config.magnets, list)

    def test_invalid_log_level(self):
        """Test that an invalid log level raises a validation error"""
        with pytest.raises(ValidationError):
            AppConfig(
                name="Invalid App",
                log_level="invalid_level",  # Must be one of debug, info, warning, error, critical
            )

    def test_invalid_magnet_references(self):
        """Test that magnets referencing non-existent teams raise validation errors"""
        with pytest.raises(ValidationError):
            AppConfig(
                name="Invalid App",
                teams=[
                    TeamConfig(name="team1", lead="model1"),
                ],
                magnets=[
                    MagnetConfig(source="team1", target="non_existent_team"),  # Target team doesn't exist
                ],
            )

        with pytest.raises(ValidationError):
            AppConfig(
                name="Invalid App",
                teams=[
                    TeamConfig(name="team1", lead="model1"),
                ],
                magnets=[
                    MagnetConfig(source="non_existent_team", target="team1"),  # Source team doesn't exist
                ],
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors"""
        with pytest.raises(ValidationError):
            AppConfig()  # Missing name
