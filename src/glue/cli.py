#!/usr/bin/env python3
"""
GLUE Framework CLI

A command-line interface for the GenAI Linking & Unification Engine
(GLUE) framework, which provides tools for working with multi-model
AI applications with features like adhesive tool bindings and
magnetic information flow.
"""
import os
import sys
import json
import asyncio
import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import framework modules
# Use direct imports to avoid circular imports
from glue.core import GlueApp, AdhesiveType
from glue.dsl import GlueDSLParser, GlueLexer
from glue.tools import SimpleBaseTool, register_tool

# Import from cliHelpers for backward compatibility with tests
from glue.cliHelpers import (
    colorize_agent_output,
    format_agent_message,
    parse_interactive_command,
    get_interactive_help_text,
    format_agent_interactions
)

# Re-export functions for test compatibility
async def interactive_session(app: 'GlueApp'):
    """
    Run an interactive session with the GLUE application.
    
    Args:
        app: The GLUE application to run
    """
    print(f"\nStarting interactive session with {app.name}...")
    print(get_interactive_help_text())
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
                print("Exiting interactive session.")
                break
                
            # Parse as command if it starts with /
            if user_input.startswith("/"):
                command, args = parse_interactive_command(user_input)
                
                if command == "help":
                    display_interactive_help()
                elif command == "status":
                    display_app_status(app)
                elif command == "tools":
                    display_available_tools(app)
                elif command == "teams":
                    display_team_structure(app)
                else:
                    print(f"Unknown command: {command}")
            else:
                # Process as regular input to the app
                if user_input:
                    # This would be handled by the app's input processing
                    print("Processing input...")
                    # Placeholder for actual app interaction
                    print("Response would appear here.")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Use /exit to quit.")
        except Exception as e:
            print(f"Error: {e}")

def display_app_status(app: 'GlueApp'):
    """
    Display the current status of the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print(f"\nApplication: {app.name}")
    print(f"Models: {len(app.models)}")
    print(f"Teams: {len(app.teams)}")
    print(f"Tools: {len(app.tools)}")
    print(f"Development mode: {app.config.get('development', False)}")
    print(f"Sticky mode: {app.config.get('sticky', False)}")

def display_team_structure(app: 'GlueApp'):
    """
    Display the team structure of the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print("\nTeam Structure:")
    for team_name, team in app.teams.items():
        print(f"\n[TEAM] {team_name}")
        print(f"  Lead: {team.lead.name}")
        if team.members:
            print("  Members:")
            for member in team.members:
                print(f"    - {member.name}")
        if team.tools:
            print("  Tools:")
            for tool in team.tools:
                print(f"    - {tool.name}")

def create_tool(name: str, description: str, tool_type: str = "basic"):
    """
    Create a new tool with the given name and description.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        tool_type: Type of tool to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Creating {tool_type} tool: {name}")
    print(f"Description: {description}")
    # Placeholder for actual tool creation
    return True

def list_tools():
    """
    List all available tools in the GLUE framework.
    
    Returns:
        List of tool names
    """
    # Placeholder for actual tool listing
    return ["web_search", "file_handler", "code_interpreter"]

def display_interactive_help():
    """Display help information for interactive mode."""
    print(get_interactive_help_text())

# Version information
__version__ = "0.1.0-alpha"  # Updated for alpha release

# Constants
DEFAULT_ENV_FILE = ".env"
CONFIG_DIR = os.path.expanduser("~/.glue")
LOGS_DIR = os.path.join(CONFIG_DIR, "logs")
TEMPLATES_DIR = os.path.join(CONFIG_DIR, "templates")

# Helper functions for formatting and display
def format_component_name(name: str, component_type: str = "component") -> tuple:
    """Format a component name for display.
    
    Args:
        name: Name of the component
        component_type: Type of component (e.g., "tool", "model", "team")
        
    Returns:
        Tuple of (directory_name, module_name, class_name)
    """
    # Convert to lowercase and handle spaces
    name_lower = name.lower()
    
    # Directory name uses hyphens
    dir_name = name_lower.replace(" ", "-")
    
    # Module name uses underscores
    module_name = name_lower.replace(" ", "_")
    
    # Class name is CamelCase
    class_name = "".join(word.capitalize() for word in name_lower.split())
    
    return dir_name, module_name, class_name

# ==================== Application Functions ====================
async def run_app(config_file: str, interactive: bool = False, input_text: str = None) -> bool:
    """Run a GLUE application from a configuration file.
    
    Args:
        config_file: Path to the GLUE configuration file
        interactive: Whether to run in interactive mode
        input_text: Input text to process (for non-interactive mode)
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger("glue.run_app")
    
    try:
        # Check if file exists
        if not os.path.exists(config_file):
            logger.error(f"Configuration file not found: {config_file}")
            print(f"Error: Configuration file not found: {config_file}")
            return False
        
        logger.info(f"Parsing GLUE file: {config_file}")
        
        # Read the GLUE file
        with open(config_file, 'r') as f:
            glue_content = f.read()
        
        # Create lexer and parser
        lexer = GlueLexer()
        parser = GlueDSLParser()
        
        # Parse the GLUE file
        tokens = lexer.tokenize(glue_content)
        
        # Only print tokens in debug mode
        if logger.level == logging.DEBUG:
            logger.debug("Tokens generated by lexer:")
            for token in tokens:
                logger.debug(f"  {token.type}: '{token.value}' at line {token.line}")
        
        # Parse tokens into AST
        logger.info("Parsing tokens into AST")
        ast = parser.parse(tokens)
        
        # Check if Portkey integration is enabled
        portkey_enabled = False
        portkey_api_key = None
        
        # Check for Portkey configuration in the AST
        if "app" in ast and "config" in ast["app"]:
            app_config = ast["app"]["config"]
            if isinstance(app_config, dict) and "portkey" in app_config and app_config["portkey"]:
                portkey_enabled = True
        
        # If Portkey is enabled, check for API key
        if portkey_enabled:
            logger.info("Portkey integration is enabled")
            portkey_api_key = os.environ.get("PORTKEY_API_KEY")
            if not portkey_api_key:
                logger.warning("Portkey API key not found in environment variables")
                print("Warning: Portkey is enabled but PORTKEY_API_KEY is not set in environment variables")
            else:
                logger.info("Portkey API key found")
                # Configure Portkey here (implementation depends on how Portkey is integrated)
        
        # Build the application
        logger.info("Building GLUE application")
        
        # Create a new GlueApp instance with the parsed config
        app = GlueApp(config=ast)
        
        # Setup the app first
        print("[DEBUG] Before app.setup()")
        await app.setup()
        print("[DEBUG] After app.setup()")
        
        # Display available tools if in interactive mode
        if interactive:
            display_available_tools(app)
            logger.info("Starting interactive session")
            print(f"\nStarting interactive session with {app.name}")
            print("[DEBUG] Before run_interactive_session()")
            await run_interactive_session(app)
            print("[DEBUG] After run_interactive_session()")
            return True
        
        # Non-interactive mode
        if input_text:
            logger.info(f"Running with input: {input_text}")
            response = await app.run(input_text)
            print(response)
            return True
        else:
            logger.error("No input provided for non-interactive mode")
            print("Error: No input provided for non-interactive mode")
            return False
        
    except Exception as e:
        logger.error(f"Error running application: {str(e)}", exc_info=True)
        print(f"Error running application: {str(e)}")
        return False

async def run_interactive_session(app: GlueApp) -> None:
    """Run an interactive session with the GLUE application.
    
    Args:
        app: The GLUE application to run
    """
    logger = logging.getLogger("glue.interactive")
    
    print("[DEBUG] Entered run_interactive_session()")
    print("\nType 'exit' or 'quit' to end the session")
    print("Type 'help' for a list of commands")
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session")
                break
                
            # Check for help command
            elif user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  exit, quit - End the session")
                print("  tools - List available tools")
                print("  clear - Clear the conversation history")
                continue
                
            # Check for tools command
            elif user_input.lower() == "tools":
                display_available_tools(app)
                continue
                
            # Check for clear command
            elif user_input.lower() == "clear":
                # Clear conversation history
                # Implementation depends on how history is stored in the app
                print("Conversation history cleared")
                continue
                
            # Process regular input
            logger.info(f"Processing user input: {user_input}")
            response = await app.run(user_input)
            print(f"\n{response}")
                
        except KeyboardInterrupt:
            print("\nSession interrupted")
            break
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")

def display_available_tools(app: GlueApp) -> None:
    """Display the available tools in the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print("\nAvailable tools:")
    # Implementation depends on how tools are stored in the app
    for tool_name, tool in app.tools.items():
        if isinstance(tool, dict):
            description = tool.get("description", "No description available")
        else:
            description = getattr(tool, "description", "No description available")
        print(f"  {tool_name} - {description}")

# ==================== Logging Setup ====================
def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    # Ensure log directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOGS_DIR, "glue.log"), mode='a')
        ]
    )
    
    # Create logger
    logger = logging.getLogger("glue")
    logger.setLevel(level)
    
    return logger

# ... (rest of the code remains the same)

# ==================== Project Management Functions ====================
def get_template_content(template: str, project_name: str) -> str:
    """Get the content for a template GLUE file.
    
    Args:
        template: Template type (basic, research, chat, or agent)
        project_name: Name of the project
        
    Returns:
        Content for the GLUE file
    """
    if template == "basic":
        return f"""glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define models
model assistant {{
    provider = openrouter
    role = "Help the user with their tasks"
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }}
}}

// Define teams
magnetize {{
    main {{
        lead = assistant
    }}
}}

apply glue
"""
    elif template == "research":
        return f"""glue app {{
    name = "{project_name}"
    config {{
        development = true
        sticky = true
    }}
}}

// Define tools
tool web_search {{
    provider = serp
}}

tool file_handler {{
    config {{
        base_path = "./workspace"
    }}
}}

// Define models
model researcher {{
    provider = openrouter
    role = "You are a research assistant that helps the user find and analyze information."
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.5
    }}
}}

// Define teams
magnetize {{
    research_team {{
        lead = researcher
        tools = [web_search, file_handler]
    }}
}}

apply glue
"""
    elif template == "chat":
        return f"""glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define models
model chat_assistant {{
    provider = openrouter
    role = "You are a helpful, friendly chat assistant."
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.8
    }}
}}

// Define teams
magnetize {{
    chat {{
        lead = chat_assistant
    }}
}}

apply glue
"""
    elif template == "agent":
        return f"""glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define tools
tool web_search {{
    provider = serp
}}

tool file_handler {{
    config {{
        base_path = "./workspace"
    }}
}}

tool code_interpreter {{}}

// Define models
model agent {{
    provider = openrouter
    role = "You are an autonomous agent that can use tools to accomplish tasks."
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }}
}}

// Define teams
magnetize {{
    agent_team {{
        lead = agent
        tools = [web_search, file_handler, code_interpreter]
    }}
}}

apply glue
"""
    else:
        return f"""glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define models
model assistant {{
    provider = openrouter
    role = "Help the user with their tasks"
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }}
}}

// Define teams
magnetize {{
    main {{
        lead = assistant
    }}
}}

apply glue
"""

def create_new_project(project_name: str, template: str = "basic"):
    """Create a new GLUE project with template files.
    
    Args:
        project_name: Name of the project to create
        template: Template type to use (basic, research, chat, or agent)
    """
    logger = logging.getLogger("glue.create_project")
    
    # Create project directory
    project_dir = Path(project_name)
    if project_dir.exists():
        logger.error(f"Project directory '{project_name}' already exists")
        print(f"Error: Project directory '{project_name}' already exists")
        return False
    
    # Create project structure
    try:
        logger.info(f"Creating project '{project_name}' with template '{template}'")
        project_dir.mkdir(parents=True)
        
        # Create app.glue file
        app_content = get_template_content(template, project_name)
        (project_dir / "app.glue").write_text(app_content)
        
        # Create workspace directory
        (project_dir / "workspace").mkdir()
        
        # Create README.md
        readme_content = f"""# {project_name}

A GLUE framework application.

## Getting Started

1. Install the GLUE framework:
   ```
   pip install glue-framework
   ```

2. Run the application:
   ```
   glue run app.glue
   ```

## Configuration

Edit the `app.glue` file to customize your application.
"""
        (project_dir / "README.md").write_text(readme_content)
        
        # Create .env file with example settings
        env_content = """# API Keys for model providers
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# OPENROUTER_API_KEY=your_openrouter_key

# API Keys for tools
# SERP_API_KEY=your_serp_key
# TAVILY_API_KEY=your_tavily_key

# Portkey integration (optional)
# PORTKEY_ENABLED=true
# PORTKEY_API_KEY=your_portkey_key
"""
        (project_dir / ".env").write_text(env_content)
        
        # Create .gitignore
        gitignore_content = """# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Workspace files (may contain sensitive data)
workspace/
"""
        (project_dir / ".gitignore").write_text(gitignore_content)
        
        logger.info(f"Project '{project_name}' created successfully")
        print(f"Project '{project_name}' created successfully in '{project_dir.absolute()}'")
        print(f"To run your application: cd {project_name} && glue run app.glue")
        return True
        
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        print(f"Error creating project: {str(e)}")
        return False

# ... (rest of the code remains the same)

# ==================== Forge Functions ====================
def forge_tool(name: str, description: str, template: str = "basic"):
    """Create a new custom tool.
    
    Args:
        name: Name of the tool
        description: Description of what the tool does
        template: Template type to use (basic, api, or data)
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger("glue.forge_tool")
    
    # Validate name
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        logger.error(f"Invalid tool name: {name}. Must start with a letter and contain only letters, numbers, and underscores.")
        print(f"Error: Invalid tool name. Must start with a letter and contain only letters, numbers, and underscores.")
        return False
    
    # Create tools directory if it doesn't exist
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)
    
    # Create tool file
    tool_file = tools_dir / f"{name}.py"
    if tool_file.exists():
        logger.error(f"Tool file '{tool_file}' already exists")
        print(f"Error: Tool file '{tool_file}' already exists")
        return False
    
    # Generate tool code based on template
    if template == "basic":
        tool_code = f"""from glue.tools import SimpleBaseTool

class {name.capitalize()}Tool(SimpleBaseTool):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        super().__init__(
            name="{name}",
            description="{description}",
            parameters={{
                "type": "object",
                "properties": {{
                    "query": {{
                        "type": "string",
                        "description": "The input for the tool"
                    }}
                }},
                "required": ["query"]
            }}
        )
    
    async def _execute(self, query: str, **kwargs):
        \"\"\"
        Execute the tool with the given query.
        
        Args:
            query: The input for the tool
            **kwargs: Additional arguments
            
        Returns:
            The result of the tool execution
        \"\"\"
        # Implement your tool logic here
        return f"Executed {{self.name}} with query: {{query}}"

# Register the tool with the framework
def get_tool():
    return {name.capitalize()}Tool()
"""
    elif template == "api":
        tool_code = f"""import aiohttp
from glue.tools import SimpleBaseTool

class {name.capitalize()}Tool(SimpleBaseTool):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        super().__init__(
            name="{name}",
            description="{description}",
            parameters={{
                "type": "object",
                "properties": {{
                    "query": {{
                        "type": "string",
                        "description": "The query to send to the API"
                    }}
                }},
                "required": ["query"]
            }}
        )
        self.api_key = None
        self.base_url = "https://api.example.com"  # Replace with actual API URL
    
    async def setup(self):
        \"\"\"Set up the tool with API key from environment variables if not provided.\"\"\"
        import os
        self.api_key = os.environ.get("{name.upper()}_API_KEY")
        if not self.api_key:
            print(f"Warning: {name.upper()}_API_KEY not found in environment variables")
    
    async def _execute(self, query: str, **kwargs):
        \"\"\"
        Execute the tool by making an API request.
        
        Args:
            query: The query to send to the API
            **kwargs: Additional arguments
            
        Returns:
            The API response
        \"\"\"
        if not self.api_key:
            return "Error: API key not configured"
        
        headers = {{"Authorization": f"Bearer {{self.api_key}}"}}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{{self.base_url}}/search",
                params={{"q": query}},
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    return f"Error: API returned status {{response.status}}"

# Register the tool with the framework
def get_tool():
    return {name.capitalize()}Tool()
"""
    elif template == "data":
        tool_code = f"""import pandas as pd
from glue.tools import SimpleBaseTool

class {name.capitalize()}Tool(SimpleBaseTool):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        super().__init__(
            name="{name}",
            description="{description}",
            parameters={{
                "type": "object",
                "properties": {{
                    "data_path": {{
                        "type": "string",
                        "description": "Path to the data file"
                    }},
                    "query": {{
                        "type": "string",
                        "description": "Query to filter or analyze the data"
                    }}
                }},
                "required": ["data_path", "query"]
            }}
        )
    
    async def _execute(self, data_path: str, query: str, **kwargs):
        \"\"\"
        Execute the tool by analyzing data.
        
        Args:
            data_path: Path to the data file
            query: Query to filter or analyze the data
            **kwargs: Additional arguments
            
        Returns:
            Analysis results
        \"\"\"
        try:
            # Load data based on file extension
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                return f"Unsupported file format: {{data_path}}"
            
            # Basic query processing - in a real tool, you'd implement more sophisticated analysis
            if "count" in query.lower():
                return f"Row count: {{len(df)}}"
            elif "columns" in query.lower():
                return f"Columns: {{', '.join(df.columns.tolist())}}"
            elif "summary" in query.lower() or "describe" in query.lower():
                return df.describe().to_string()
            else:
                # Default to returning the first few rows
                return df.head().to_string()
                
        except Exception as e:
            return f"Error processing data: {{str(e)}}"

# Register the tool with the framework
def get_tool():
    return {name.capitalize()}Tool()
"""
    else:
        logger.error(f"Unknown tool template: {template}")
        print(f"Error: Unknown tool template: {template}")
        return False
    
    # Write tool file
    tool_file.write_text(tool_code)
    
    # Create __init__.py if it doesn't exist
    init_file = tools_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Custom tools package\n")
    
    logger.info(f"Tool '{name}' created successfully")
    print(f"Tool '{name}' created successfully in '{tool_file}'")
    print(f"To use this tool, add the following to your app.glue file:")
    print(f"""
tool {name} {{
    custom = true
}}
""")
    return True

def forge_mcp(name: str, description: str):
    """Create a new MCP (Model Control Protocol) integration.
    
    Args:
        name: Name of the MCP integration
        description: Description of the MCP integration
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger("glue.forge_mcp")
    
    # Validate name
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        logger.error(f"Invalid MCP name: {name}. Must start with a letter and contain only letters, numbers, and underscores.")
        print(f"Error: Invalid MCP name. Must start with a letter and contain only letters, numbers, and underscores.")
        return False
    
    # Create mcps directory if it doesn't exist
    mcps_dir = Path("mcps")
    mcps_dir.mkdir(exist_ok=True)
    
    # Create MCP file
    mcp_file = mcps_dir / f"{name}_mcp.py"
    if mcp_file.exists():
        logger.error(f"MCP file '{mcp_file}' already exists")
        print(f"Error: MCP file '{mcp_file}' already exists")
        return False
    
    # Generate MCP code
    mcp_code = f"""from typing import Dict, List, Any, Optional
from glue.core.mcp import BaseMCP

class {name.capitalize()}MCP(BaseMCP):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="{name}", config=config or {{}})
        self.base_url = self.config.get("base_url", "https://api.example.com")
        self.api_key = self.config.get("api_key")
    
    async def setup(self):
        \"\"\"Set up the MCP with API key from environment variables if not provided.\"\"\"
        if not self.api_key:
            import os
            self.api_key = os.environ.get("{name.upper()}_API_KEY")
            if not self.api_key:
                print(f"Warning: {name.upper()}_API_KEY not found in environment variables")
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        \"\"\"
        Execute a function through this MCP.
        
        Args:
            function_name: Name of the function to execute
            arguments: Arguments to pass to the function
            
        Returns:
            The result of the function execution
        \"\"\"
        # Implement your MCP function execution logic here
        # This is a placeholder implementation
        return {{
            "result": f"Executed {{function_name}} with arguments: {{arguments}}",
            "status": "success"
        }}
    
    async def get_available_functions(self) -> List[Dict[str, Any]]:
        \"\"\"
        Get a list of functions available through this MCP.
        
        Returns:
            List of function definitions
        \"\"\"
        # Return a list of function definitions
        # This is a placeholder implementation
        return [
            {{
                "name": "example_function",
                "description": "An example function",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "param1": {{
                            "type": "string",
                            "description": "First parameter"
                        }},
                        "param2": {{
                            "type": "integer",
                            "description": "Second parameter"
                        }}
                    }},
                    "required": ["param1"]
                }}
            }}
        ]

# Register the MCP with the framework
def get_mcp():
    return {name.capitalize()}MCP()
"""
    
    # Write MCP file
    mcp_file.write_text(mcp_code)
    
    # Create __init__.py if it doesn't exist
    init_file = mcps_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Custom MCPs package\n")
    
    logger.info(f"MCP '{name}' created successfully")
    print(f"MCP '{name}' created successfully in '{mcp_file}'")
    print(f"To use this MCP, add the following to your app.glue file:")
    print(f"""
mcp {name} {{
    custom = true
}}
""")
    return True

def forge_api(name: str, description: str):
    """Create a new API integration.
    
    Args:
        name: Name of the API integration
        description: Description of the API integration
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger("glue.forge_api")
    
    # Validate name
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        logger.error(f"Invalid API name: {name}. Must start with a letter and contain only letters, numbers, and underscores.")
        print(f"Error: Invalid API name. Must start with a letter and contain only letters, numbers, and underscores.")
        return False
    
    # Create apis directory if it doesn't exist
    apis_dir = Path("apis")
    apis_dir.mkdir(exist_ok=True)
    
    # Create API file
    api_file = apis_dir / f"{name}_api.py"
    if api_file.exists():
        logger.error(f"API file '{api_file}' already exists")
        print(f"Error: API file '{api_file}' already exists")
        return False
    
    # Generate API code
    api_code = f"""import aiohttp
from typing import Dict, Any, Optional

class {name.capitalize()}API:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.example.com"  # Replace with actual API URL
    
    async def setup(self):
        \"\"\"Set up the API client with API key from environment variables if not provided.\"\"\"
        if not self.api_key:
            import os
            self.api_key = os.environ.get("{name.upper()}_API_KEY")
            if not self.api_key:
                print(f"Warning: {name.upper()}_API_KEY not found in environment variables")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response
        \"\"\"
        if not self.api_key:
            return {{"error": "API key not configured"}}
        
        headers = {{"Authorization": f"Bearer {{self.api_key}}"}}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{{self.base_url}}/{{endpoint}}",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {{
                        "error": f"API returned status {{response.status}}",
                        "status": response.status
                    }}
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            API response
        \"\"\"
        if not self.api_key:
            return {{"error": "API key not configured"}}
        
        headers = {{
            "Authorization": f"Bearer {{self.api_key}}",
            "Content-Type": "application/json"
        }}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{{self.base_url}}/{{endpoint}}",
                json=data,
                headers=headers
            ) as response:
                if response.status in (200, 201):
                    return await response.json()
                else:
                    return {{
                        "error": f"API returned status {{response.status}}",
                        "status": response.status
                    }}

# Create an instance of the API client
def get_api_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    return {name.capitalize()}API(api_key, base_url)
"""
    
    # Write API file
    api_file.write_text(api_code)
    
    # Create __init__.py if it doesn't exist
    init_file = apis_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Custom APIs package\n")
    
    logger.info(f"API '{name}' created successfully")
    print(f"API '{name}' created successfully in '{api_file}'")
    print(f"To use this API in your application:")
    print(f"""
from apis.{name}_api import get_api_client

# In an async function:
api_client = get_api_client()
await api_client.setup()
result = await api_client.get("endpoint", {{"param": "value"}})
""")
    return True

def run_forge_command():
    """Run the GLUE Forge interactive CLI to create custom components using Google Gemini.
    
    This function implements an interactive CLI that guides users through setting up
    their API key for Google Gemini and using it to create custom components.
    """
    import os
    import re
    import json
    import getpass
    from pathlib import Path
    
    logger = logging.getLogger("glue.forge")
    
    # ASCII art banner
    banner = """
  ██████  ██      ██    ██ ███████     ███████  ██████  ██████   ██████  ███████ 
 ██       ██      ██    ██ ██          ██      ██    ██ ██   ██ ██       ██      
 ██   ███ ██      ██    ██ █████       █████   ██    ██ ██████  ██   ███ █████   
 ██    ██ ██      ██    ██ ██          ██      ██    ██ ██   ██ ██    ██ ██      
  ██████  ███████  ██████  ███████     ██       ██████  ██   ██  ██████  ███████ 
                                                                                 
"""
    print(banner)
    print("Welcome to GLUE Forge - Create custom components with AI assistance")
    print("=" * 70)
    print("\nGLUE Forge uses Google Gemini 2.5 Pro, which is currently free to use.")
    print("You'll need an API key from either Google AI Studio or OpenRouter.")
    
    # Check if API key is already configured
    config_dir = Path(os.path.expanduser("~/.glue"))
    config_dir.mkdir(exist_ok=True)
    
    forge_config_file = config_dir / "forge_config.json"
    api_key = None
    api_source = None
    model_name = None
    
    if forge_config_file.exists():
        try:
            with open(forge_config_file, "r") as f:
                config = json.load(f)
                api_key = config.get("api_key")
                api_source = config.get("api_source")
                model_name = config.get("model_name")
                
            print(f"\nFound existing configuration: Using {api_source} API key")
            use_existing = input("Would you like to use this configuration? (Y/n): ").strip().lower()
            
            if use_existing != "n":
                print(f"Using existing {api_source} API key")
            else:
                api_key = None
                api_source = None
                model_name = None
        except Exception as e:
            logger.error(f"Error reading forge config: {str(e)}")
            print("Error reading existing configuration. Setting up new configuration.")
            api_key = None
            api_source = None
            model_name = None
    
    # If no API key is configured, prompt for one
    if not api_key:
        print("\nYou need an API key to use GLUE Forge. Choose your API key source:")
        print("1. Google AI Studio (https://makersuite.google.com/app/apikey)")
        print("2. OpenRouter (https://openrouter.ai/keys)")
        
        choice = None
        while choice not in ["1", "2"]:
            choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            api_source = "Google AI Studio"
            model_name = "gemini-1.5-pro"
            print(f"\nGet your API key from: https://makersuite.google.com/app/apikey")
        else:
            api_source = "OpenRouter"
            model_name = "google/gemini-2.5-pro-exp-03-25:free"
            print(f"\nGet your API key from: https://openrouter.ai/keys")
        
        api_key = getpass.getpass(f"\nEnter your {api_source} API key: ").strip()
        
        # Validate API key format
        if not api_key or len(api_key) < 10:
            print("Invalid API key. Please provide a valid API key.")
            return False
        
        # Save configuration
        config = {
            "api_key": api_key,
            "api_source": api_source,
            "model_name": model_name
        }
        
        with open(forge_config_file, "w") as f:
            json.dump(config, f)
        
        print(f"\nAPI key saved. Using {api_source} with model: {model_name}")
    
    # Set the appropriate environment variable based on the API source
    if api_source == "Google AI Studio":
        os.environ["GOOGLE_API_KEY"] = api_key
    else:  # OpenRouter
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Now run the forge command
    print("\nWhat would you like to create with GLUE Forge?")
    print("1. Custom Tool")
    print("2. Custom MCP Integration")
    print("3. Custom API Integration")
    
    forge_choice = None
    while forge_choice not in ["1", "2", "3"]:
        forge_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if forge_choice == "1":
        name = input("\nEnter a name for your tool (letters, numbers, underscores only): ").strip()
        description = input("Enter a brief description of what your tool does: ").strip()
        
        print("\nChoose a template:")
        print("1. Basic Tool (simple function)")
        print("2. API Tool (makes external API calls)")
        print("3. Data Tool (processes data files)")
        
        template_choice = None
        while template_choice not in ["1", "2", "3"]:
            template_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        template = "basic" if template_choice == "1" else "api" if template_choice == "2" else "data"
        
        print(f"\nCreating {template} tool: {name}")
        forge_tool(name, description, template)
        
    elif forge_choice == "2":
        name = input("\nEnter a name for your MCP integration (letters, numbers, underscores only): ").strip()
        description = input("Enter a brief description of what your MCP integration does: ").strip()
        
        print(f"\nCreating MCP integration: {name}")
        forge_mcp(name, description)
        
    elif forge_choice == "3":
        name = input("\nEnter a name for your API integration (letters, numbers, underscores only): ").strip()
        description = input("Enter a brief description of what your API integration does: ").strip()
        
        print(f"\nCreating API integration: {name}")
        forge_api(name, description)
    
    print("\nThank you for using GLUE Forge!")
    return True

# ==================== Main CLI Entry Point ====================
def main():
    """Main CLI entry point"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="GLUE Framework CLI - GenAI Linking & Unification Engine"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a GLUE application")
    run_parser.add_argument("config", help="Path to GLUE config file")
    run_parser.add_argument("--input", "-i", help="Input text for the app")
    run_parser.add_argument("--interactive", "-I", action="store_true", 
                         help="Run in interactive mode")
    run_parser.add_argument("--verbose", "-v", action="store_true",
                         help="Enable verbose logging")
    run_parser.add_argument("--env", "-e", help="Path to .env file")
    
    # New command
    new_parser = subparsers.add_parser("new", help="Create a new GLUE project")
    new_parser.add_argument("project", help="Project name")
    new_parser.add_argument("--template", "-t", 
                          choices=["basic", "research", "chat", "agent"],
                          default="basic",
                          help="Project template to use")
    
    # Forge command (for creating custom components)
    forge_parser = subparsers.add_parser("forge", help="Create custom components with AI assistance")
    forge_subparsers = forge_parser.add_subparsers(dest="forge_type", help="Type of component to forge")
    
    # Forge tool
    forge_tool_parser = forge_subparsers.add_parser("tool", help="Create a custom tool")
    forge_tool_parser.add_argument("name", help="Tool name")
    forge_tool_parser.add_argument("--description", "-d", required=True, help="Tool description")
    forge_tool_parser.add_argument("--template", "-t", 
                                choices=["basic", "api", "data"],
                                default="basic",
                                help="Tool template to use")
    
    # Forge MCP
    forge_mcp_parser = forge_subparsers.add_parser("mcp", help="Create a custom MCP integration")
    forge_mcp_parser.add_argument("name", help="MCP name")
    forge_mcp_parser.add_argument("--description", "-d", required=True, help="MCP description")
    
    # Forge API
    forge_api_parser = forge_subparsers.add_parser("api", help="Create a custom API integration")
    forge_api_parser.add_argument("name", help="API name")
    forge_api_parser.add_argument("--description", "-d", required=True, help="API description")
    
    # List tools command
    subparsers.add_parser("list-tools", help="List available tools")
    
    # List models command
    subparsers.add_parser("list-models", help="List available models")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a GLUE file")
    validate_parser.add_argument("file", help="Path to GLUE file to validate")
    
    # Version command
    subparsers.add_parser("version", help="Show GLUE version")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    logger = setup_logging(log_level)
    
    # Process commands
    try:
        if args.command == "run":
            # Load environment variables if specified
            if args.env:
                from dotenv import load_dotenv
                load_dotenv(args.env)
            elif os.path.exists(DEFAULT_ENV_FILE):
                # Load default .env file if it exists
                from dotenv import load_dotenv
                load_dotenv(DEFAULT_ENV_FILE)
            
            # Run the application
            asyncio.run(run_app(args.config, args.interactive, args.input))
            
        elif args.command == "new":
            create_new_project(args.project, args.template)
            
        elif args.command == "forge":
            if not args.forge_type:
                # Run interactive forge command if no subcommand is specified
                run_forge_command()
            elif args.forge_type == "tool":
                import re  # Import here to avoid unnecessary import
                forge_tool(args.name, args.description, args.template)
            elif args.forge_type == "mcp":
                import re  # Import here to avoid unnecessary import
                forge_mcp(args.name, args.description)
            elif args.forge_type == "api":
                import re  # Import here to avoid unnecessary import
                forge_api(args.name, args.description)
            else:
                forge_parser.print_help()
            
        elif args.command == "list-tools":
            list_tools()
            
        elif args.command == "list-models":
            list_models()
            
        elif args.command == "validate":
            validate_glue_file(args.file)
            
        elif args.command == "version":
            print(f"GLUE Framework version {__version__}")
            
        else:
            # If no command or unrecognized command, show help
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
