# glue/tools/base.py
# ==================== Imports ====================
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
from enum import Enum
import inspect
import asyncio
import logging
from pydantic import BaseModel, Field, ConfigDict

from ..core.types import AdhesiveType

# ==================== Constants ====================
logger = logging.getLogger("glue.tools")

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3

# ==================== Class Definitions ====================
class ToolPermission(Enum):
    """Tool permission levels"""
    READ = "read"          # Read-only operations
    WRITE = "write"        # File/state modifications
    NETWORK = "network"    # Network access
    EXECUTE = "execute"    # Code execution

class ToolConfig(BaseModel):
    """Tool configuration with Pydantic validation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)
    max_retries: int = Field(default=MAX_RETRIES, ge=0)
    required_permissions: Set[ToolPermission] = Field(default_factory=set)
    adhesive_types: Set[AdhesiveType] = Field(default_factory=lambda: {AdhesiveType.TAPE})
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Tool(ABC):
    """Base class for all GLUE tools"""
    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[ToolConfig] = None
    ):
        self.name = name
        self.description = description
        self.config = config or ToolConfig()
        
        # Input parameters from execute signature
        self.inputs = self._get_input_parameters()
        
        # Runtime state
        self._initialized = False
        self._instance_id = None

    # ==================== Core Methods ====================
    async def initialize(self, instance_data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize tool with optional instance data"""
        if not self._initialized:
            await self._validate_permissions()
            if instance_data:
                await self._load_instance(instance_data)
            self._initialized = True
            logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, input_data: Any) -> Any:
        """Execute tool with timeout and retries"""
        if not self._initialized:
            await self.initialize()
            
        tries = 0
        max_tries = self.config.max_retries + 1  # Include the initial attempt
        
        while tries < max_tries:
            try:
                async with asyncio.timeout(self.config.timeout):
                    # Execute the tool
                    return await self._execute(input_data)
            except asyncio.TimeoutError:
                tries += 1
                if tries >= max_tries:
                    raise
                logger.warning(f"Tool {self.name} timed out, attempt {tries}")
            except Exception as e:
                # Log and re-raise any other exceptions
                logger.error(f"Tool {self.name} execution failed: {str(e)}")
                raise

    @abstractmethod
    async def _execute(self, input_data: Any) -> Any:
        """Core tool execution logic to be implemented by subclasses"""
        pass

    # ==================== Helper Methods ====================
    def _get_input_parameters(self) -> Dict[str, Any]:
        """Extract parameters from execute method signature"""
        sig = inspect.signature(self._execute)
        params = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = {
                "type": param.annotation if param.annotation != inspect.Parameter.empty else Any,
                "default": None if param.default == inspect.Parameter.empty else param.default,
                "optional": param.default != inspect.Parameter.empty
            }
            params[name] = param_info
            
        return params

    async def _validate_permissions(self) -> None:
        """Validate required permissions are available"""
        for permission in self.config.required_permissions:
            if not await self._check_permission(permission):
                raise PermissionError(f"Missing required permission: {permission}")

    async def _check_permission(self, permission: ToolPermission) -> bool:
        """Check if a specific permission is available"""
        # Default implementation - override for actual checks
        return True

    async def _load_instance(self, data: Dict[str, Any]) -> None:
        """Load tool instance data"""
        self._instance_id = data.get('instance_id')
        # Override for custom instance loading

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "config": self.config.model_dump(),
            "initialized": self._initialized
        }

    # ==================== Error Handling ====================
    async def cleanup(self) -> None:
        """Clean up tool resources"""
        self._initialized = False
        self._instance_id = None
        # Override for custom cleanup

class DynamicTool(Tool):
    """Tool that executes a dynamic function"""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Any,
        config: Optional[ToolConfig] = None
    ):
        super().__init__(name, description, config)
        self.function = function
        
    async def _execute(self, input_data: Any) -> Any:
        """Execute the dynamic function"""
        try:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(input_data)
            else:
                return self.function(input_data)
        except Exception as e:
            logger.debug(f"Error in dynamic function: {type(e).__name__}: {str(e)}")
            raise  # Re-raise to ensure proper error propagation
