"""
Pydantic models for GLUE framework type validation.

This module contains all the Pydantic models used for validating data structures
throughout the GLUE framework. These models provide runtime type checking,
data validation, and schema generation capabilities.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator

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
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "call_1",
                "name": "web_search",
                "arguments": {"query": "GLUE framework for AI", "max_results": 5}
            }
        }


class ToolResult(BaseModel):
    """Result from a tool execution"""
    tool_call_id: str = Field(..., description="ID of the tool call that was executed")
    content: Any = Field(..., description="Result content from the tool execution")
    
    class Config:
        schema_extra = {
            "example": {
                "tool_call_id": "call_1",
                "content": "Search results for GLUE framework for AI"
            }
        }


# ==================== Message Models ====================
class Message(BaseModel):
    """Message for communication between agents"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls included in the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the message")
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = {'system', 'user', 'assistant', 'tool'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "role": "assistant",
                "content": "I found the following information about the GLUE framework.",
                "tool_calls": [],
                "metadata": {"timestamp": "2025-03-14T13:45:30.123456"}
            }
        }


# ==================== Configuration Models ====================
class ModelConfig(BaseModel):
    """Configuration for an LLM model"""
    name: str = Field(..., description="Name of the model configuration")
    provider: str = Field(..., description="Provider of the model (e.g., openai, anthropic)")
    model: str = Field(..., description="Model identifier (e.g., gpt-4, claude-3-opus)")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperature for response generation")
    max_tokens: int = Field(2048, gt=0, description="Maximum tokens in the response")
    description: str = Field("", description="Description of the model configuration")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_params: Dict[str, Any] = Field(default_factory=dict, description="Additional API parameters")
    provider_class: Optional[str] = Field(None, description="Custom provider class path for custom providers")
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate that the provider is supported"""
        # Import here to avoid circular imports
        from glue.core.model import ModelProvider
        
        # Check if the value is a ModelProvider enum value
        if isinstance(v, ModelProvider):
            return v.value
        
        # Check if the value is a string that matches a ModelProvider enum value
        valid_providers = {p.value for p in ModelProvider}
        if v in valid_providers:
            return v
            
        # If it's a string but not a valid provider, raise an error
        raise ValueError(f"Provider should be one of {valid_providers}")
    
    class Config:
        schema_extra = {
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


class ToolConfig(BaseModel):
    """Configuration for a tool"""
    name: str = Field(..., description="Name of the tool")
    description: str = Field("", description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    required_permissions: List[str] = Field(default_factory=list, description="Permissions required to use this tool")
    
    class Config:
        schema_extra = {
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


class TeamConfig(BaseModel):
    """Team configuration"""
    name: str = Field(..., description="Name of the team")
    lead: str = Field(..., description="Name of the lead model for this team")
    members: List[str] = Field(default_factory=list, description="Names of member models in this team")
    tools: List[str] = Field(default_factory=list, description="Names of tools available to this team")
    
    @validator('members')
    def validate_members(cls, v, values):
        # Ensure lead is not also listed as a member
        if 'lead' in values and values['lead'] in v:
            raise ValueError(f"Lead model '{values['lead']}' should not also be listed as a member")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "research_team",
                "lead": "gpt4_researcher",
                "members": ["claude_analyst", "gemini_assistant"],
                "tools": ["web_search", "document_reader"]
            }
        }


class MagnetConfig(BaseModel):
    """Configuration for magnetic connections between teams"""
    source: str = Field(..., description="Source team name")
    target: str = Field(..., description="Target team name")
    flow_type: FlowType = Field(FlowType.BIDIRECTIONAL, description="Type of information flow")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters for information flow")
    
    @root_validator
    def validate_teams(cls, values):
        if values.get('source') == values.get('target'):
            raise ValueError("Source and target teams must be different")
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "source": "research_team",
                "target": "writing_team",
                "flow_type": "push",
                "filters": {"topics": ["research_findings", "data_analysis"]}
            }
        }


class AppConfig(BaseModel):
    """Configuration for a GLUE application"""
    name: str = Field(..., description="Name of the application")
    description: str = Field("", description="Description of the application")
    version: str = Field("0.1.0", description="Version of the application")
    development: bool = Field(True, description="Whether the app is in development mode")
    log_level: str = Field("info", description="Logging level")
    models: List[ModelConfig] = Field(default_factory=list, description="Model configurations")
    tools: List[ToolConfig] = Field(default_factory=list, description="Tool configurations")
    teams: List[TeamConfig] = Field(default_factory=list, description="Team configurations")
    magnets: List[MagnetConfig] = Field(default_factory=list, description="Magnetic connections")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if v.lower() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.lower()
    
    @validator('magnets')
    def validate_magnets(cls, v, values):
        # Validate that teams referenced in magnets exist
        if 'teams' in values:
            team_names = {team.name for team in values['teams']}
            for magnet in v:
                if magnet.source not in team_names:
                    raise ValueError(f"Source team '{magnet.source}' does not exist")
                if magnet.target not in team_names:
                    raise ValueError(f"Target team '{magnet.target}' does not exist")
        return v
    
    class Config:
        schema_extra = {
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
