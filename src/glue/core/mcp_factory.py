# glue/core/mcp_factory.py
# ==================== Imports ====================
import inspect
import asyncio
import re
from typing import Dict, Any, Optional, Callable
import logging
from pydantic import BaseModel, Field, ConfigDict

from .mcp import MCPServer, MCPTool
from ..tools.tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType
from .sandbox import CodeSandbox, SandboxConfig

# ==================== Constants ====================
logger = logging.getLogger("glue.mcp.factory")


# ==================== Class Definitions ====================
class MCPSpec(BaseModel):
    """Specification for dynamic MCP server creation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    handlers: Dict[str, Callable] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sandbox_enabled: bool = False
    sandbox_config: Optional[SandboxConfig] = None


class DynamicMCPServer(MCPServer):
    """Enhanced MCP server with direct handler execution"""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 8000,
    ):
        super().__init__(name, version, host, port)
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a handler function directly"""
        self.handlers[action] = handler
        logger.info(f"Registered handler {action} with MCP server {self.name}")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request directly using handlers"""
        action = request.get("action")
        if not action:
            raise ValueError("Missing action in request")

        # First try direct handlers
        handler = self.handlers.get(action)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(request)
                else:
                    result = handler(request)

                # If result is a dict, merge with status
                if isinstance(result, dict):
                    if "status" not in result:
                        result["status"] = "success"
                    return result
                else:
                    return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Handler execution failed: {str(e)}")
                return {"status": "error", "error": str(e)}

        # Fall back to tool-based handling
        return await super().handle_request(request)


class DynamicMCPFactory:
    """Factory for creating MCP servers and clients dynamically"""

    def __init__(self):
        self.created_servers: Dict[str, DynamicMCPServer] = {}
        self.default_sandbox_config = SandboxConfig()

    # ==================== Core Methods ====================
    async def create_server(self, spec: MCPSpec) -> DynamicMCPServer:
        """Create a new MCP server from specification"""
        if spec.name in self.created_servers:
            logger.warning(f"MCP server {spec.name} already exists")
            return self.created_servers[spec.name]

        try:
            # Create server
            server = DynamicMCPServer(
                name=spec.name, version=spec.version, host=spec.host, port=spec.port
            )

            # Register handlers directly
            for action_name, handler in spec.handlers.items():
                server.register_handler(action_name, handler)

                # Also create a tool for compatibility
                tool = await self._create_handler_tool(action_name, handler)
                server.register_tool(tool)

            self.created_servers[spec.name] = server
            logger.info(f"Created dynamic MCP server: {spec.name}")
            return server

        except Exception as e:
            logger.error(f"Failed to create MCP server {spec.name}: {str(e)}")
            raise

    async def create_from_code(
        self,
        code: str,
        name: str,
        handler_name: Optional[str] = None,
        action_name: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000,
        sandbox_enabled: bool = False,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> DynamicMCPServer:
        """Create an MCP server from Python code string"""
        try:
            # Create namespace for code execution
            namespace = {}

            if sandbox_enabled:
                # Use sandbox for code execution
                sandbox_config = sandbox_config or self.default_sandbox_config
                sandbox = CodeSandbox(sandbox_config)

                # Execute code in sandbox
                module = await sandbox.execute_code(code, f"mcp_module_{name}")

                # Find handler function
                handler_name = handler_name or next(
                    (
                        name
                        for name, obj in inspect.getmembers(module)
                        if inspect.isfunction(obj) and not name.startswith("_")
                    ),
                    None,
                )

                if not handler_name:
                    raise ValueError("No function found in code")

                handler = getattr(module, handler_name)

                # Wrap handler to execute in sandbox
                original_handler = handler

                async def sandboxed_handler(request):
                    return await sandbox.execute_function(original_handler, request)

                handler = sandboxed_handler
            else:
                # Execute code directly
                exec(code, namespace)

                # Find handler function
                funcs = [
                    (k, v)
                    for k, v in namespace.items()
                    if callable(v) and not k.startswith("_")
                ]

                if not funcs:
                    raise ValueError("No function found in code")

                handler_name = handler_name or funcs[0][0]
                handler = namespace.get(handler_name)

            if not handler:
                raise ValueError(f"Handler {handler_name} not found in code")

            # Create spec
            action = action_name or handler_name
            spec = MCPSpec(
                name=name,
                host=host,
                port=port,
                handlers={action: handler},
                sandbox_enabled=sandbox_enabled,
                sandbox_config=sandbox_config,
            )

            return await self.create_server(spec)

        except Exception as e:
            logger.error(f"Failed to create MCP server from code: {str(e)}")
            raise

    async def create_tool_for_server(
        self, server: MCPServer, tool_name: str, description: str
    ) -> MCPTool:
        """Create an MCP tool that connects to a server"""
        endpoint = f"http://{server.host}:{server.port}/"

        # Create the tool
        tool = MCPTool(name=tool_name, description=description, endpoint=endpoint)

        return tool

    def register_with_team(self, server: MCPServer, team: Any) -> None:
        """Register an MCP server with a team"""
        team.register_mcp(server)
        logger.info(f"Registered MCP server {server.name} with team {team.name}")

    async def create_tool_from_code(
        self,
        code: str,
        tool_name: str,
        function_name: Optional[str] = None,
        description: str = "",
        sandbox_enabled: bool = False,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> Tool:
        """Create a tool from Python code string"""
        try:
            # Create namespace for code execution
            namespace = {}

            if sandbox_enabled:
                # Use sandbox for code execution
                sandbox_config = sandbox_config or self.default_sandbox_config
                sandbox = CodeSandbox(sandbox_config)

                # Execute code in sandbox
                module = await sandbox.execute_code(code, f"tool_module_{tool_name}")

                # Find function
                function_name = function_name or next(
                    (
                        name
                        for name, obj in inspect.getmembers(module)
                        if inspect.isfunction(obj) and not name.startswith("_")
                    ),
                    None,
                )

                if not function_name:
                    raise ValueError("No function found in code")

                func = getattr(module, function_name)

                # Create dynamic tool class with sandboxed execution
                class DynamicSandboxedTool(Tool):
                    def __init__(self, name: str, description: str):
                        config = ToolConfig(
                            timeout=30.0,
                            required_permissions={ToolPermission.READ},
                            adhesive_types={
                                AdhesiveType.TAPE,
                                AdhesiveType.VELCRO,
                                AdhesiveType.GLUE,
                            },
                        )
                        super().__init__(name, description, config)
                        self.sandbox = sandbox
                        self.func = func

                    async def execute(self, parameters: Dict[str, Any] = None) -> Any:
                        """Execute the tool with the given parameters"""
                        # Ensure parameters is always passed to the function
                        if parameters is None:
                            parameters = {}
                        return await self._execute(parameters)

                    async def _execute(self, parameters: Dict[str, Any]) -> Any:
                        # Ensure parameters is always passed to the function
                        if parameters is None:
                            parameters = {}
                        return await self.sandbox.execute_function(
                            self.func, parameters
                        )

                # Create and return the tool
                return DynamicSandboxedTool(tool_name, description)
            else:
                # Execute code directly
                exec(code, namespace)

                # Find function
                funcs = [
                    (k, v)
                    for k, v in namespace.items()
                    if callable(v) and not k.startswith("_")
                ]

                if not funcs:
                    raise ValueError("No function found in code")

                function_name = function_name or funcs[0][0]
                func = namespace.get(function_name)

                if not func:
                    raise ValueError(f"Function {function_name} not found in code")

                # Create dynamic tool class
                class DynamicTool(Tool):
                    def __init__(self, name: str, description: str):
                        config = ToolConfig(
                            timeout=30.0,
                            required_permissions={ToolPermission.READ},
                            adhesive_types={
                                AdhesiveType.TAPE,
                                AdhesiveType.VELCRO,
                                AdhesiveType.GLUE,
                            },
                        )
                        super().__init__(name, description, config)
                        self.func = func

                    async def execute(self, parameters: Dict[str, Any] = None) -> Any:
                        """Execute the tool with the given parameters"""
                        # Ensure parameters is always passed to the function
                        if parameters is None:
                            parameters = {}
                        return await self._execute(parameters)

                    async def _execute(self, parameters: Dict[str, Any]) -> Any:
                        # Ensure parameters is always passed to the function
                        if parameters is None:
                            parameters = {}
                        if asyncio.iscoroutinefunction(self.func):
                            return await self.func(parameters)
                        else:
                            return self.func(parameters)

                # Create and return the tool
                return DynamicTool(tool_name, description)

        except Exception as e:
            logger.error(f"Failed to create tool from code: {str(e)}")
            raise

    # ==================== Helper Methods ====================
    async def _create_handler_tool(self, action_name: str, handler: Callable) -> Tool:
        """Create a tool from a handler function"""
        tool_name = action_name
        description = handler.__doc__ or f"Handler for {action_name}"

        # Create tool config
        config = ToolConfig(
            timeout=30.0,
            required_permissions={ToolPermission.READ},
            adhesive_types={AdhesiveType.TAPE},
        )

        # Create dynamic class
        tool_cls = type(
            f"{action_name.title()}Tool",
            (Tool,),
            {
                "__init__": self._create_init(),
                "_execute": self._create_execute(handler),
                "description": description,
            },
        )

        # Create instance
        tool = tool_cls(tool_name, description, config)
        return tool

    def _create_init(self) -> Callable:
        """Create __init__ method for dynamic tool"""

        def __init__(
            self, name: str, description: str, config: Optional[ToolConfig] = None
        ):
            Tool.__init__(self, name, description, config)

        return __init__

    def _create_execute(self, handler: Callable) -> Callable:
        """Create _execute method for dynamic tool"""

        async def _execute(self, input_data: Any) -> Any:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(input_data)
                return handler(input_data)
            except Exception as e:
                logger.error(f"Handler execution failed: {str(e)}")
                raise

        return _execute

    # ==================== Natural Language Processing ====================
    async def parse_natural_request(
        self, request: str, team_name: str, sandbox_enabled: bool = True
    ) -> DynamicMCPServer:
        """Create an MCP server from a natural language request"""
        # Simple pattern matching for MVP

        # Extract key information
        server_match = re.search(
            r"create (?:an? )?mcp (?:server )?(?:that |to )?(.*?)(?:\.|$)",
            request.lower(),
        )
        if not server_match:
            description = "process data"  # Default
        else:
            description = server_match.group(1)

        name = f"{team_name}_{description.split()[0].lower()}_mcp"

        # Create basic handler template
        handler_code = f"""
from typing import Any, Dict

async def process_data(request: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Process data as requested: {description}\"\"\"
    # Basic implementation
    return {{
        "status": "success",
        "message": "Processed by {name}",
        "request": request
    }}
"""

        # Create server from code
        return await self.create_from_code(
            code=handler_code,
            name=name,
            handler_name="process_data",
            action_name="process_data",
            sandbox_enabled=sandbox_enabled,
        )

    # ==================== Error Handling ====================
    async def cleanup(self) -> None:
        """Clean up created MCP servers"""
        for server in list(self.created_servers.values()):
            if hasattr(server, "cleanup"):
                await server.cleanup()
        self.created_servers.clear()


# ==================== Factory Functions ====================
async def create_dynamic_mcp_server(
    name: str,
    version: str = "1.0.0",
    host: str = "localhost",
    port: int = 8000,
    handlers: Dict[str, Callable] = None,
) -> DynamicMCPServer:
    """Create a dynamic MCP server with the given configuration"""
    factory = DynamicMCPFactory()
    spec = MCPSpec(
        name=name, version=version, host=host, port=port, handlers=handlers or {}
    )
    return await factory.create_server(spec)
