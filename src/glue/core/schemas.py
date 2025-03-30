"""
Pydantic models for GLUE framework type validation.

This module contains all the Pydantic models used for validating data structures
throughout the GLUE framework. These models provide runtime type checking,
data validation, and schema generation capabilities.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ==================== Enumerations ====================
class AdhesiveType(str, Enum):
    """Types of adhesive bindings that control tool result persistence"""
    GLUE = "glue"      # Team-wide persistent results
    VELCRO = "velcro"  # Session-based persistence
    TAPE = "tape"      # One-time use, no persistence


class FlowType(str, Enum):
    """Types of information flow between teams"""
    PUSH = "push"           # One-way information flow (source to target)
    PULL = "pull"           # One-way information flow (target from source)
    BIDIRECTIONAL = "bidirectional"  # Two-way information flow
    REPEL = "repel"         # No information flow (isolation)


# ==================== Tool Models ====================
class ToolCall(BaseModel):
    """Model for a tool call request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_id": "call_1",
                "name": "web_search",
                "arguments": {"query": "GLUE framework for AI", "max_results": 5}
            }
        }
    )
    
    tool_id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")


class ToolResult(BaseModel):
    """Result from a tool execution"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_call_id": "call_1",
                "content": "Search results for GLUE framework for AI"
            }
        }
    )
    
    tool_call_id: str = Field(..., description="ID of the tool call that was executed")
    content: Any = Field(..., description="Result content from the tool execution")
    # For backward compatibility with tests
    tool_name: Optional[str] = Field(None, description="Name of the tool (backward compatibility)")
    result: Optional[Any] = Field(None, description="Result data (backward compatibility)")
    adhesive: Optional[AdhesiveType] = Field(None, description="Adhesive type (backward compatibility)")
    
    @model_validator(mode='before')
    def handle_backward_compatibility(cls, data):
        """Handle backward compatibility with old field names"""
        if isinstance(data, dict):
            # If using old format (tool_name, result) but missing new fields
            if 'tool_name' in data and 'result' in data:
                if 'tool_call_id' not in data:
                    data['tool_call_id'] = f"call_{data['tool_name']}"
                if 'content' not in data:
                    data['content'] = data['result']
        return data


# ==================== Message Models ====================
class Message(BaseModel):
    """Message for communication between agents"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "assistant",
                "content": "I found the following information about the GLUE framework.",
                "tool_calls": [],
                "metadata": {"timestamp": "2025-03-14T13:45:30.123456"}
            }
        }
    )
    
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls included in the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the message")
    
    @field_validator('role')
    def validate_role(cls, v):
        allowed_roles = {'system', 'user', 'assistant', 'tool'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


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
                "provider_class": None
            }
        }
    )
    
    name: str = Field(..., description="Name of the model configuration")
    provider: str = Field(..., description="Provider of the model (e.g., openai, anthropic)")
    model: str = Field(..., description="Model identifier (e.g., gpt-4, claude-3-opus)")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperature for response generation")
    max_tokens: int = Field(2048, gt=0, description="Maximum tokens in the response")
    description: str = Field("", description="Description of the model configuration")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_params: Dict[str, Any] = Field(default_factory=dict, description="Additional API parameters")
    provider_class: Optional[str] = Field(None, description="Custom provider class path for custom providers")
    
    @field_validator('provider')
    def validate_provider(cls, v):
        supported_providers = {'openai', 'anthropic', 'openrouter', 'custom'}
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
                    "max_results": {"type": "integer", "description": "Maximum number of results"}
                },
                "required_permissions": ["internet_access"]
            }
        }
    )
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field("", description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    required_permissions: List[str] = Field(default_factory=list, description="Permissions required to use this tool")


class TeamConfig(BaseModel):
    """Team configuration"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "research_team",
                "lead": "gpt4_researcher",
                "members": ["claude_analyst", "gemini_assistant"],
                "tools": ["web_search", "document_reader"]
            }
        }
    )
    
    name: str = Field(..., description="Name of the team")
    lead: str = Field(..., description="Name of the lead model for this team")
    members: List[str] = Field(default_factory=list, description="Names of member models in this team")
    tools: List[str] = Field(default_factory=list, description="Names of tools available to this team")
    
    @field_validator('members')
    def validate_members(cls, v, values):
        if 'lead' in values.data and values.data['lead'] in v:
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
                "filters": {"topics": ["research_findings", "data_analysis"]}
            }
        }
    )
    
    source: str = Field(..., description="Source team name")
    target: str = Field(..., description="Target team name")
    flow_type: FlowType = Field(FlowType.BIDIRECTIONAL, description="Type of information flow")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters for information flow")
    
    @model_validator(mode='after')
    def validate_teams(self):
        if self.source == self.target:
            raise ValueError("Source and target teams must be different")
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
                "magnets": []
            }
        }
    )
    
    name: str = Field(..., description="Name of the application")
    description: str = Field("", description="Description of the application")
    version: str = Field("0.1.0", description="Version of the application")
    development: bool = Field(True, description="Whether the app is in development mode")
    log_level: str = Field("info", description="Logging level")
    models: List[ModelConfig] = Field(default_factory=list, description="Model configurations")
    tools: List[ToolConfig] = Field(default_factory=list, description="Tool configurations")
    teams: List[TeamConfig] = Field(default_factory=list, description="Team configurations")
    magnets: List[MagnetConfig] = Field(default_factory=list, description="Magnetic connections")
    
    @field_validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.lower()
    
    @field_validator('magnets')
    def validate_magnets(cls, v, values):
        if 'teams' not in values.data:
            return v
            
        team_names = {team.name for team in values.data['teams']}
        for magnet in v:
            if magnet.source not in team_names:
                raise ValueError(f"Magnet source team '{magnet.source}' does not exist")
            if magnet.target not in team_names:
                raise ValueError(f"Magnet target team '{magnet.target}' does not exist")
        return v
