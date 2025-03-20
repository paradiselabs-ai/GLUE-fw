# glue/tools/factory.py
# ==================== Imports ====================
import inspect
import asyncio
from typing import Dict, Any, Optional, Type, Union, Callable, Set
import logging
from pydantic import BaseModel, Field, ConfigDict

from .base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType

# ==================== Constants ====================
logger = logging.getLogger("glue.tools.factory")

# ==================== Class Definitions ====================
class ToolSpec(BaseModel):
    """Specification for dynamic tool creation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str
    function: Union[str, Callable]
    permissions: Set[ToolPermission] = Field(default_factory=lambda: {ToolPermission.READ})
    adhesives: Set[AdhesiveType] = Field(default_factory=lambda: {AdhesiveType.TAPE})
    timeout: float = 30.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DynamicToolFactory:
    """Factory for creating tools dynamically"""
    _registry: Dict[str, Type[Tool]] = {}

    def __init__(self):
        self.created_tools: Dict[str, Tool] = {}

    # ==================== Core Methods ====================
    async def create_tool(self, spec: ToolSpec) -> Tool:
        """Create a new tool from specification"""
        if spec.name in self.created_tools:
            logger.warning(f"Tool {spec.name} already exists")
            return self.created_tools[spec.name]

        try:
            # Create tool config
            config = ToolConfig(
                timeout=spec.timeout,
                required_permissions=spec.permissions,
                adhesive_types=spec.adhesives
            )

            # Create dynamic class
            tool_cls = type(
                f"{spec.name.title()}Tool",
                (Tool,),
                {
                    "__init__": self._create_init(spec),
                    "_execute": self._create_execute(spec),
                    "description": spec.description,
                    "metadata": spec.metadata
                }
            )

            # Create instance
            tool = tool_cls(spec.name, spec.description, config)
            self.created_tools[spec.name] = tool
            
            logger.info(f"Created dynamic tool: {spec.name}")
            return tool

        except Exception as e:
            logger.error(f"Failed to create tool {spec.name}: {str(e)}")
            raise

    async def create_from_code(self, code: str, name: Optional[str] = None) -> Optional[Tool]:
        """Create a tool from Python code string"""
        try:
            # Create restricted namespace
            namespace = {}
            
            # Execute code
            exec(code, namespace)
            
            # Find function
            funcs = [
                (k, v) for k, v in namespace.items()
                if callable(v) and not k.startswith('_')
            ]
            
            if not funcs:
                raise ValueError("No function found in code")
                
            func_name, func = funcs[0] if name is None else next(
                (f for f in funcs if f[0] == name),
                funcs[0]
            )
            
            # Create spec
            spec = ToolSpec(
                name=name or func_name,
                description=func.__doc__ or f"Dynamic tool: {func_name}",
                function=func
            )
            
            return await self.create_tool(spec)
            
        except Exception as e:
            logger.error(f"Failed to create tool from code: {str(e)}")
            return None

    # ==================== Helper Methods ====================
    def _create_init(self, spec: ToolSpec) -> Callable:
        """Create __init__ method for dynamic tool"""
        def __init__(self, name: str, description: str, config: Optional[ToolConfig] = None):
            super().__init__(name, description, config)
            self.spec = spec
        return __init__

    def _create_execute(self, spec: ToolSpec) -> Callable:
        """Create _execute method for dynamic tool"""
        if isinstance(spec.function, str):
            # Create from string
            namespace = {}
            exec(spec.function, namespace)
            func = next(
                v for k, v in namespace.items()
                if callable(v) and not k.startswith('_')
            )
        else:
            func = spec.function

        async def _execute(self, input_data: Any) -> Any:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(input_data)
                return func(input_data)
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                raise

        return _execute

    @classmethod
    def register_tool_class(cls, tool_cls: Type[Tool]) -> None:
        """Register a tool class for later use"""
        cls._registry[tool_cls.__name__] = tool_cls

    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[Tool]]:
        """Get a registered tool class"""
        return cls._registry.get(name)

    # ==================== Natural Language Processing ====================
    async def parse_natural_request(self, request: str, team_name: str) -> Optional[Tool]:
        """Create tool from natural language request"""
        # Simple pattern matching for MVP
        import re
        
        # Extract key information
        tool_match = re.search(r"create (?:a )?tool (?:that |to )?(.*?)(?:\.|$)", request.lower())
        if not tool_match:
            return None
            
        description = tool_match.group(1)
        name = re.sub(r'\s+', '_', description.split()[0].lower())
        
        # Determine permissions
        permissions = {ToolPermission.READ}
        if any(w in request.lower() for w in ["write", "save", "create", "modify"]):
            permissions.add(ToolPermission.WRITE)
        if any(w in request.lower() for w in ["network", "internet", "web", "online"]):
            permissions.add(ToolPermission.NETWORK)
        if any(w in request.lower() for w in ["execute", "run", "code"]):
            permissions.add(ToolPermission.EXECUTE)
        
        # Create basic function template
        function = f"""
async def {name}(input_data: Any) -> Any:
    \"""{description}\"""
    # TODO: Implement {description}
    return {{"status": "not_implemented", "description": "{description}"}}
"""

        # Create spec
        spec = ToolSpec(
            name=name,
            description=description,
            function=function,
            permissions=permissions
        )
        
        return await self.create_tool(spec)

    # ==================== Error Handling ====================
    async def cleanup(self) -> None:
        """Clean up created tools"""
        for tool in self.created_tools.values():
            if hasattr(tool, 'cleanup'):
                await tool.cleanup()
        self.created_tools.clear()
