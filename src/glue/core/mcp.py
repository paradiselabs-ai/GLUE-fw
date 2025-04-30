# glue/mcp/client.py
# ==================== Imports ====================
import asyncio
from typing import Dict, Any, Optional
import logging
import httpx
from pydantic import BaseModel, Field, ConfigDict

from glue.tools.tool_base import Tool, ToolConfig

# ==================== Constants ====================
logger = logging.getLogger("glue.mcp")


class MCPConfig(BaseModel):
    """Configuration for MCP client"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: str
    api_key: Optional[str] = None
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Class Definitions ====================
class MCPClient:
    """Client for Model Control Protocol integration"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    async def execute(self, control_message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP control message"""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        tries = 0
        while tries < self.config.max_retries:
            try:
                response = await self.client.post(
                    self.config.endpoint, headers=headers, json=control_message
                )

                if response.status_code == 200:
                    return response.json()

                # Handle specific error codes
                if response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    tries += 1
                    continue

                # Other errors
                raise ValueError(f"MCP error: {response.status_code} {response.text}")

            except Exception:
                tries += 1
                if tries == self.config.max_retries:
                    raise
                await asyncio.sleep(2**tries)  # Exponential backoff

        raise RuntimeError("Failed to execute MCP control message")

    async def close(self):
        """Close the client connection"""
        await self.client.aclose()


class MCPTool(Tool):
    """Tool that integrates with Model Control Protocol"""

    def __init__(
        self,
        name: str,
        description: str,
        endpoint: str,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(name, description, config)
        self.mcp_client = MCPClient(MCPConfig(endpoint=endpoint, api_key=api_key))

    async def _execute(self, input_data: Any) -> Any:
        """Execute tool through MCP"""
        control_message = {
            "action": self.name,
            "parameters": (
                input_data if isinstance(input_data, dict) else {"input": input_data}
            ),
            "metadata": {
                "tool_name": self.name,
                "tool_version": self.config.metadata.get("version", "1.0.0"),
            },
        }

        try:
            result = await self.mcp_client.execute(control_message)
            return result
        except Exception as e:
            logger.error(f"MCP tool execution failed: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up MCP client"""
        await self.mcp_client.close()


# ==================== Factory Functions ====================
def create_mcp_tool(
    name: str,
    description: str,
    endpoint: str,
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> MCPTool:
    """Create a new MCP tool"""
    tool_config = ToolConfig(**(config or {}))
    return MCPTool(
        name=name,
        description=description,
        endpoint=endpoint,
        api_key=api_key,
        config=tool_config,
    )


# ==================== Server Integration ====================
class MCPServer:
    """
    Basic MCP server implementation.
    For more complex servers, use the official MCP server implementation.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 8000,
    ):
        self.name = name
        self.version = version
        self.host = host
        self.port = port
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the server"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool {tool.name} with MCP server {self.name}")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request"""
        action = request.get("action")
        if not action:
            raise ValueError("Missing action in request")

        tool = self.tools.get(action)
        if not tool:
            raise ValueError(f"Unknown tool: {action}")

        parameters = request.get("parameters", {})
        try:
            result = await tool.execute(parameters)
            return {
                "status": "success",
                "result": result,
                "metadata": {
                    "server": self.name,
                    "version": self.version,
                    "tool": action,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "server": self.name,
                    "version": self.version,
                    "tool": action,
                },
            }

    async def serve(self) -> None:
        """Start the MCP server"""
        from aiohttp import web

        async def handle_mcp_request(request):
            try:
                data = await request.json()
                result = await self.handle_request(data)
                return web.json_response(result)
            except Exception as e:
                return web.json_response(
                    {"status": "error", "error": str(e)}, status=500
                )

        app = web.Application()
        app.router.add_post("/", handle_mcp_request)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)

        logger.info(f"Starting MCP server {self.name} on {self.host}:{self.port}")
        await site.start()
