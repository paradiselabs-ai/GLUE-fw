"""
Pydantic models for GLUE framework type validation.

This module contains all the Pydantic models used for validating data structures
throughout the GLUE framework. These models provide runtime type checking,
data validation, and schema generation capabilities.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationInfo
import logging


# ==================== Enumerations ====================
class AdhesiveType(str, Enum):
    """Types of adhesive bindings that control tool result persistence"""

    GLUE = "glue"  # Team-wide persistent results
    VELCRO = "velcro"  # Session-based persistence
    TAPE = "tape"  # One-time use, no persistence


class FlowType(str, Enum):
    """Types of information flow between teams"""

    PUSH = "push"  # One-way information flow (source to target)
    PULL = "pull"  # One-way information flow (target from source)
    BIDIRECTIONAL = "bidirectional"  # Two-way information flow
    REPEL = "repel"  # No information flow (isolation)


# ==================== Tool Models ====================
class ToolCall(BaseModel):
    """Model for a tool call request"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_id": "call_1",
                "name": "web_search",
                "arguments": {"query": "GLUE framework for AI", "max_results": 5},
            }
        }
    )

    tool_id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool call"
    )


class ToolResult(BaseModel):
    """Result from a tool execution"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_call_id": "call_1",
                "content": "Search results for GLUE framework for AI",
            }
        }
    )

    # For backward compatibility with tests
    tool_name: Optional[str] = Field(
        None, description="Name of the tool (backward compatibility)"
    )
    result: Optional[Any] = Field(
        None, description="Result data (backward compatibility)"
    )
    adhesive: Optional[AdhesiveType] = Field(
        None, description="Adhesive type (backward compatibility)"
    )

    # New fields
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call that was executed"
    )
    content: Optional[Any] = Field(
        None, description="Result content from the tool execution"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the result"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when the result was created"
    )

    @model_validator(mode="before")
    def handle_backward_compatibility(cls, data):
        """Handle backward compatibility with old field names"""
        if isinstance(data, dict):
            # If using old format (tool_name, result) but missing new fields
            if "tool_name" in data and "result" in data:
                if "tool_call_id" not in data:
                    data["tool_call_id"] = f"call_{data['tool_name']}"
                if "content" not in data:
                    data["content"] = data["result"]
                if "metadata" not in data:
                    data["metadata"] = {}
            # If using new format but missing old fields
            elif "tool_call_id" in data and "content" in data:
                if "tool_name" not in data:
                    data["tool_name"] = data["tool_call_id"].replace("call_", "", 1)
                if "result" not in data:
                    data["result"] = data["content"]
                if "adhesive" not in data:
                    data["adhesive"] = AdhesiveType.GLUE
        return data

    @model_validator(mode="after")
    def validate_required_fields(cls, model):
        """Ensure that either old or new format fields are present and adhesive is provided"""
        if (model.tool_name is None or model.result is None) and (
            model.tool_call_id is None or model.content is None
        ):
            raise ValueError(
                "Either (tool_name, result) or (tool_call_id, content) must be provided"
            )

        # Ensure adhesive is provided when using the old format
        if (
            model.tool_name is not None
            and model.result is not None
            and model.adhesive is None
        ):
            raise ValueError(
                "Adhesive type must be provided when using tool_name and result"
            )

        return model


# ==================== Message Models ====================
class Message(BaseModel):
    """Message for communication between agents"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "assistant",
                "content": "I found the following information about the GLUE framework.",
                "tool_calls": [],
                "metadata": {"timestamp": "2025-03-14T13:45:30.123456"},
            }
        }
    )

    role: str = Field(
        ..., description="Role of the message sender (system, user, assistant)"
    )
    content: str = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tool calls included in the message"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the message"
    )

    @field_validator("role")
    def validate_role(cls, v):
        allowed_roles = {"system", "user", "assistant", "tool", "model", "function"}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


# ==================== Agent Output Models ====================
class ParseAnalyzeOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thought_process": "Parsed objectives and constraints from task description.",
                "analysis": {
                    "objectives": ["obj1", "obj2"],
                    "constraints": ["constraint1"],
                },
            }
        }
    )
    thought_process: str = Field(
        ..., description="The agent's reasoning about the task requirements."
    )
    analysis: Dict[str, Any] = Field(
        ...,
        description="Structured analysis of objectives, constraints, and key information.",
    )


class PlanPhaseOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "substeps": ["Step 1", "Step 2", "Step 3"],
                "tool_requirements": ["tool1", "tool2"],
                "estimated_confidence": "high",
            }
        }
    )
    substeps: List[str] = Field(
        ..., description="Sequential substeps to execute the task."
    )
    tool_requirements: List[str] = Field(
        ..., description="List of tools required for the substeps."
    )
    estimated_confidence: str = Field(
        ..., description="Estimated confidence level of the plan (high/medium/low)."
    )


class ToolSelectionOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected_tool_name": "performTaskStub",
                "tool_parameters": {"parameter": "value"},
            }
        }
    )
    selected_tool_name: str = Field(
        ..., description="Name of the tool the agent decided to use."
    )
    tool_parameters: Dict[str, Any] = Field(
        ..., description="Parameters for invoking the selected tool."
    )


class MemoryDecisionOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "save_to_memory": True,
                "analysis": "Identified key details to store.",
            }
        }
    )
    save_to_memory: bool = Field(
        ..., description="Whether to save the result into curated memory."
    )
    analysis: Optional[str] = Field(
        None, description="Summary content to save to memory if save_to_memory is true."
    )


class SelfEvalOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "evaluation_summary": "Result aligns with constraints but misses edge case.",
                "consistency_check": "Passed",
                "alignment_check": "Partial Fail",
                "confidence_level": "MediumConfidence",
                "error_detected": False,
            }
        }
    )
    evaluation_summary: str = Field(..., description="Summary of the self-evaluation.")
    consistency_check: str = Field(..., description="Result of consistency check.")
    alignment_check: str = Field(..., description="Result of alignment check.")
    confidence_level: str = Field(
        ...,
        description="Confidence level (HighConfidence/MediumConfidence/LowConfidence/CriticalError).",
    )
    error_detected: bool = Field(
        ..., description="Whether an error was detected during evaluation."
    )


class FormatResultOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "final_answer": "Here is the final result with context.",
                "supporting_context": ["Entry 1", "Entry 2"],
            }
        }
    )
    final_answer: Any = Field(..., description="The agent's final formatted answer.")
    supporting_context: List[str] = Field(
        ..., description="List of context entries used to produce the final answer."
    )


# ==================== Configuration Models ====================
class ModelConfig(BaseModel):
    """Configuration for an LLM model"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "gpt4",
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048,
                "description": "GPT-4 model for complex reasoning tasks",
                "api_key": None,
                "api_params": {"top_p": 1.0},
                "provider_class": None,
            }
        }
    )

    name: str = Field(..., description="Name of the model configuration")
    provider: str = Field(
        ..., description="Provider of the model (e.g., openai, anthropic)"
    )
    model: str = Field(..., description="Model identifier (e.g., gpt-4, claude-3-opus)")
    temperature: float = Field(
        0.7, ge=0.0, le=1.0, description="Temperature for response generation"
    )
    max_tokens: int = Field(2048, gt=0, description="Maximum tokens in the response")
    description: str = Field("", description="Description of the model configuration")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional API parameters"
    )
    provider_class: Optional[str] = Field(
        None, description="Custom provider class path for custom providers"
    )

    @field_validator("provider")
    def validate_provider(cls, v):
        supported_providers = {"openai", "anthropic", "openrouter", "gemini", "custom"}
        if v not in supported_providers:
            raise ValueError(f"Provider must be one of {supported_providers}")
        return v


class ToolConfig(BaseModel):
    """Configuration for a tool"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required_permissions": ["internet_access"],
            }
        }
    )

    name: str = Field(..., description="Name of the tool")
    description: str = Field("", description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool"
    )
    required_permissions: List[str] = Field(
        default_factory=list, description="Permissions required to use this tool"
    )


class TeamConfig(BaseModel):
    """Team configuration"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "research_team",
                "lead": "gpt4_researcher",
                "members": ["claude_analyst", "gemini_assistant"],
                "tools": ["web_search", "document_reader"],
            }
        }
    )

    name: str = Field(..., description="Name of the team")
    lead: str = Field(..., description="Name of the lead model for this team")
    members: List[str] = Field(
        default_factory=list, description="Names of member models in this team"
    )
    tools: List[str] = Field(
        default_factory=list, description="Names of tools available to this team"
    )

    @field_validator("members")
    def validate_members(cls, v, values):
        if "lead" in values.data and values.data["lead"] in v:
            raise ValueError("Lead model should not also be listed in members")
        return v


class MagnetConfig(BaseModel):
    """Configuration for magnetic connections between teams"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "research_team",
                "target": "writing_team",
                "flow_type": "push",
                "filters": {"topics": ["research_findings", "data_analysis"]},
            }
        }
    )

    source: str = Field(..., description="Source team name")
    target: str = Field(..., description="Target team name")
    flow_type: FlowType = Field(
        FlowType.BIDIRECTIONAL, description="Type of information flow"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Filters for information flow"
    )

    @model_validator(mode="after")
    def validate_teams(self):
        if self.source == self.target:
            raise ValueError("Source and target teams must be different")
        return self


class FlowDefinitionConfig(BaseModel):
    """Configuration for a specific flow definition between two teams."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "research_to_analysis",
                "source": "research_team",
                "target": "analysis_team",
                "type": "PUSH",
                "config": {"priority": 1, "conditions": {"status": "approved"}}
            },
            "example_pull_any": { # Example for team <- pull
                "name": "docs_pull_fallback",
                "source": None, # Source is None for 'team <- pull'
                "target": "docs_team", # docs_team is the one that can pull
                "type": "pull",
                "config": {}
            }
        }
    )

    name: str = Field(..., description="Unique name of the flow definition")
    source: Optional[str] = Field(None, description="Name of the source team for this flow. None if type is PULL and it's a 'pull from any' flow.")
    target: str = Field(..., description="Name of the target team for this flow")
    type: FlowType = Field(..., description="Type of flow (PUSH, PULL, or BIDIRECTIONAL)")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional flow-specific configurations (e.g., conditions, transformations)"
    )

    @field_validator('type', mode='before')
    @classmethod
    def normalize_flow_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def validate_teams_different(self):
        if self.source == self.target:
            raise ValueError(f"Flow definition '{self.name}': Source and target teams must be different.")
        return self


class AppConfig(BaseModel):
    """Configuration for a GLUE application"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Research Assistant",
                "description": "An AI research assistant application",
                "version": "0.1.0",
                "development": True,
                "log_level": "info",
                "models": [],
                "tools": [],
                "teams": [],
                "magnets": [],
                "flows": [], 
            }
        }
    )

    name: str = Field(..., description="Name of the application")
    description: str = Field("", description="Description of the application")
    version: str = Field("0.1.0", description="Version of the application")
    development: bool = Field(
        True, description="Whether the app is in development mode"
    )
    log_level: str = Field("info", description="Logging level")
    models: List[ModelConfig] = Field(
        default_factory=list, description="Model configurations"
    )
    tools: List[ToolConfig] = Field(
        default_factory=list, description="Tool configurations"
    )
    teams: List[TeamConfig] = Field(
        default_factory=list, description="Team configurations"
    )
    magnets: List[MagnetConfig] = Field(
        default_factory=list, description="Magnetic connections for dynamic field interactions"
    )
    flows: List[FlowDefinitionConfig] = Field(
        default_factory=list, description="Specific, persistent flow definitions between teams"
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_app_config_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Normalize 'name' from 'app': {'name': ...} to top-level 'name'
        if "app" in data and isinstance(data["app"], dict) and "name" in data["app"]:
            if "name" not in data:
                data["name"] = data["app"].pop("name")
            if not data["app"]:
                data.pop("app")
        
        if "name" not in data:
            data["name"] = "Unnamed GLUE App (defaulted in validator)"

        # 2. Transform 'models' (as before)
        if "models" in data and isinstance(data["models"], dict):
            models_list = []
            for model_name, model_body_dict in data["models"].items():
                if isinstance(model_body_dict, dict):
                    final_model_config = model_body_dict.copy()
                    nested_config_params = final_model_config.pop("config", None)
                    if isinstance(nested_config_params, dict):
                        for key, value in nested_config_params.items():
                            final_model_config[key] = value
                    final_model_config["name"] = model_name
                    models_list.append(final_model_config)
                else:
                    logging.warning(f"Model configuration for '{model_name}' is not a dictionary. Skipping.")
            data["models"] = models_list

        # 3. Transform 'tools' (as before)
        if "tools" in data and isinstance(data["tools"], dict):
            tools_list = []
            for tool_name, tool_config_dict in data["tools"].items():
                if isinstance(tool_config_dict, dict):
                    if "name" not in tool_config_dict:
                        tool_config_dict["name"] = tool_name
                    tools_list.append(tool_config_dict)
                else:
                    logging.warning(f"Tool configuration for '{tool_name}' is not a dictionary. Skipping.")
            data["tools"] = tools_list

        # 4. Transform 'magnetize' block into 'teams' List[TeamConfig-like dicts]
        # The DSL parser might provide team definitions under a 'magnetize' key.
        # If 'teams' is present and already a list (e.g. from direct JSON), it might be used as is,
        # but DSL's 'magnetize' should populate 'data["teams"]' for AppConfig validation.
        if "magnetize" in data and isinstance(data["magnetize"], dict):
            if not data.get("teams"): # If 'teams' is empty or not present, populate from 'magnetize'
                teams_list = []
                for team_name, team_config_dict in data["magnetize"].items():
                    if isinstance(team_config_dict, dict):
                        if "name" not in team_config_dict:
                            team_config_dict["name"] = team_name
                        teams_list.append(team_config_dict)
                    else:
                        logging.warning(f"Team configuration (from magnetize) for '{team_name}' is not a dictionary. Skipping.")
                data["teams"] = teams_list
            # If 'teams' was already populated (e.g. from a direct list input), decide on merge strategy or overwrite.
            # For now, if 'teams' already exists and is a list, we assume it's authoritative.
            # If 'teams' was a dict, the old logic would handle it; this new logic prioritizes 'magnetize' if 'teams' is empty.
        
        # Fallback: if 'teams' itself was a dict (old logic, less likely with 'magnetize' present)
        elif "teams" in data and isinstance(data["teams"], dict):
            teams_list = []
            for team_name, team_config_dict in data["teams"].items():
                if isinstance(team_config_dict, dict):
                    if "name" not in team_config_dict:
                        team_config_dict["name"] = team_name
                    teams_list.append(team_config_dict)
                else:
                    logging.warning(f"Team configuration for '{team_name}' is not a dictionary. Skipping.")
            data["teams"] = teams_list


        # 5. Transform 'flows' if it's a dict {name: config} to List[FlowDefinitionConfig-like dicts]
        if "flows" in data and isinstance(data["flows"], dict):
            flows_list = []
            for flow_name, flow_details_dict in data["flows"].items():
                if isinstance(flow_details_dict, dict):
                    transformed_flow_dict = {"name": flow_name} # Correctly add name here
                    # Pop known fields to ensure they are not duplicated in 'config'
                    if "source" in flow_details_dict:
                        transformed_flow_dict["source"] = flow_details_dict.pop("source")
                    if "target" in flow_details_dict:
                        transformed_flow_dict["target"] = flow_details_dict.pop("target")
                    if "type" in flow_details_dict:
                        transformed_flow_dict["type"] = flow_details_dict.pop("type")
                    # Any remaining items in flow_details_dict go into the 'config' field
                    transformed_flow_dict["config"] = flow_details_dict 
                    flows_list.append(transformed_flow_dict)
                else:
                    logging.warning(f"Flow definition for '{flow_name}' is not a dictionary. Skipping.")
            data["flows"] = flows_list
        
        # Note: 'magnets' might also need similar dict-to-list transformation if DSL allows dict form.
        # For now, assuming magnets are parsed as a list of dicts or handled by Pydantic directly if it's List[MagnetConfig].

        return data

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.lower()

    @field_validator("magnets")
    @classmethod
    def validate_magnets_teams_exist(cls, v: List[MagnetConfig], info: ValidationInfo) -> List[MagnetConfig]:
        if "teams" not in info.data or not info.data["teams"]:
            # If teams list is not yet available or empty in the (partially) validated data.
            # This might happen if 'teams' itself fails validation or is processed later.
            # Depending on strictness, could raise error or just return v.
            # For now, assume if 'teams' is empty/missing, it's an issue caught elsewhere or config is invalid.
            return v 

        team_names = {team.name for team in info.data["teams"]}
        for magnet in v:
            if magnet.source not in team_names:
                raise ValueError(f"Magnet source team '{magnet.source}' does not exist in defined teams.")
            if magnet.target not in team_names:
                raise ValueError(f"Magnet target team '{magnet.target}' does not exist in defined teams.")
        return v

    @field_validator("flows")
    @classmethod
    def validate_flow_teams_exist(cls, v: List[FlowDefinitionConfig], info: ValidationInfo) -> List[FlowDefinitionConfig]:
        if "teams" not in info.data or not info.data["teams"]:
            return v # Similar to magnets, if teams aren't there, can't validate against them.

        team_names = {team.name for team in info.data["teams"]}
        for flow_def in v:
            # Validate Target: Target must always be a defined team.
            if flow_def.target not in team_names:
                raise ValueError(
                    f"Flow '{flow_def.name}': Target team '{flow_def.target}' is not defined in the application's teams."
                )

            # Validate Source: Source must be a defined team UNLESS it's None (for 'team <- pull' case).
            if flow_def.source is not None and flow_def.source not in team_names:
                raise ValueError(
                    f"Flow '{flow_def.name}': Source team '{flow_def.source}' is not defined in the application's teams."
                )
            
            # Additional check: If type is PULL and source is None, target must be specified.
            if flow_def.type == FlowType.PULL and flow_def.source is None and not flow_def.target:
                 raise ValueError(
                    f"Flow '{flow_def.name}' is a generic PULL flow but has no target team specified to initiate the pull."
                ) # This case should ideally be caught by target being required anyway.

        return v
