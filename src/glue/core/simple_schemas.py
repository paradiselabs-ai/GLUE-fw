# Simple Python classes for smolagents compatibility
# These follow smolagents' preference for lightweight data structures

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class AdhesiveType(str, Enum):
    """Types of adhesive bindings that control tool result persistence"""
    GLUE = "glue"
    VELCRO = "velcro" 
    TAPE = "tape"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                # Try case-insensitive match
                for member in cls:
                    if member.value.lower() == value.lower():
                        return member
        raise ValueError(f"Invalid AdhesiveType: {value}")


class FlowType(str, Enum):
    """Types of information flow between teams"""
    PUSH = "push"
    PULL = "pull"
    BIDIRECTIONAL = "bidirectional"
    REPEL = "repel"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                for member in cls:
                    if member.value.lower() == value.lower():
                        return member
        raise ValueError(f"Invalid FlowType: {value}")


@dataclass
class ToolCall:
    """Simple tool call representation"""
    tool_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool execution - simplified version"""
    
    # For backward compatibility with tests
    tool_name: Optional[str] = None
    result: Optional[Any] = None
    adhesive: Optional[AdhesiveType] = None
    
    # New fields
    tool_call_id: Optional[str] = None
    content: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None  # Using string instead of datetime for simplicity
    
    def __post_init__(self):
        # Basic validation and backward compatibility
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()
            
        # Handle backward compatibility
        if self.tool_name and self.result and not self.tool_call_id:
            self.tool_call_id = f"call_{self.tool_name}"
        if self.tool_call_id and self.content and not self.tool_name:
            self.tool_name = self.tool_call_id.replace("call_", "", 1)
        if self.tool_name and self.result and not self.content:
            self.content = self.result
        if self.tool_call_id and self.content and not self.result:
            self.result = self.content
        if not self.adhesive:
            self.adhesive = AdhesiveType.GLUE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # Convert adhesive if present
        adhesive = data.get("adhesive")
        if adhesive is not None:
            data = dict(data)
            data["adhesive"] = AdhesiveType.from_value(adhesive)
        return cls(**data)


@dataclass
class Message:
    """Simple message representation"""
    role: str
    content: str
    tool_calls: List['ToolCall'] = field(default_factory=list)  # Fix forward reference
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        allowed_roles = {"system", "user", "assistant", "tool", "model", "function"}
        if self.role not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")


@dataclass
class ModelConfig:
    """Simple model configuration - smolagents compatible"""
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    description: str = ""
    api_key: Optional[str] = None
    api_params: Dict[str, Any] = field(default_factory=dict)
    provider_class: Optional[str] = None
    
    def __post_init__(self):
        # Basic validation without pydantic overhead
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        supported_providers = {"openai", "anthropic", "openrouter", "gemini", "custom"}
        if self.provider not in supported_providers:
            raise ValueError(f"Provider must be one of {supported_providers}")


@dataclass
class ToolConfig:
    """Simple tool configuration"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)


@dataclass
class TeamConfig:
    """Simple team configuration"""
    name: str
    description: str
    leader_agent: str
    member_agents: List[str] = field(default_factory=list)
    flow_type: FlowType = FlowType.BIDIRECTIONAL
    adhesives: List[AdhesiveType] = field(default_factory=lambda: [AdhesiveType.GLUE])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        flow_type = data.get("flow_type", FlowType.BIDIRECTIONAL)
        adhesives = data.get("adhesives", [AdhesiveType.GLUE])
        return cls(
            name=data["name"],
            description=data["description"],
            leader_agent=data["leader_agent"],
            member_agents=data.get("member_agents", []),
            flow_type=FlowType.from_value(flow_type),
            adhesives=[AdhesiveType.from_value(a) for a in adhesives],
        )


@dataclass
class AppConfig:
    """Simple application configuration"""
    name: str
    description: str
    models: List[ModelConfig] = field(default_factory=list)
    tools: List[ToolConfig] = field(default_factory=list)
    teams: List[TeamConfig] = field(default_factory=list)


# Helper functions for creating configs from dictionaries
def create_model_config_from_dict(data: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from dictionary with sensible defaults"""
    return ModelConfig(
        name=data["name"],
        provider=data["provider"], 
        model=data.get("model", data["provider"]),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
        description=data.get("description", ""),
        api_key=data.get("api_key"),
        api_params=data.get("api_params", {}),
        provider_class=data.get("provider_class")
    )


def create_tool_config_from_dict(data: Dict[str, Any]) -> ToolConfig:
    """Create ToolConfig from dictionary"""
    return ToolConfig(
        name=data["name"],
        description=data["description"],
        parameters=data.get("parameters", {}),
        required_permissions=data.get("required_permissions", [])
    )


def create_team_config_from_dict(data: Dict[str, Any]) -> TeamConfig:
    """Create TeamConfig from dictionary"""
    return TeamConfig(
        name=data["name"],
        description=data["description"],
        leader_agent=data["leader_agent"],
        member_agents=data.get("member_agents", []),
        flow_type=FlowType(data.get("flow_type", "bidirectional")),
        adhesives=[AdhesiveType(a) for a in data.get("adhesives", ["glue"])]
    )


def create_app_config_from_dict(data: Dict[str, Any]) -> AppConfig:
    """Create AppConfig from dictionary"""
    return AppConfig(
        name=data["name"],
        description=data["description"],
        models=[create_model_config_from_dict(m) for m in data.get("models", [])],
        tools=[create_tool_config_from_dict(t) for t in data.get("tools", [])],
        teams=[create_team_config_from_dict(t) for t in data.get("teams", [])]
    )
