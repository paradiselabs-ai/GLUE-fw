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
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import framework modules
# Use direct imports to avoid circular imports
from glue.core import GlueApp, AdhesiveType
from glue.dsl import GlueDSLParser, GlueLexer
from glue.tools import SimpleBaseTool, register_tool

# Version information
__version__ = "0.1.0"  # Placeholder version

# Constants
DEFAULT_ENV_FILE = ".env"

# Helper functions for formatting and display
def format_component_name(name: str, component_type: str = "component") -> str:
    """Format a component name for display.
    
    Args:
        name: Name of the component
        component_type: Type of component (e.g., "tool", "model", "team")
        
    Returns:
        Formatted component name
    """
    return f"[{component_type.upper()}] {name}"

def create_tool(name: str, description: str, function_code: str) -> bool:
    """Create a new tool with the given name and description.
    
    Args:
        name: Name of the tool
        description: Description of what the tool does
        function_code: Python code implementing the tool function
        
    Returns:
        True if the tool was created successfully, False otherwise
    """
    # In a real implementation, this would create a new tool file
    # and register it with the tool registry
    return True

def list_tools() -> List[Dict[str, str]]:
    """List all available tools.
    
    Returns:
        List of dictionaries containing tool information
    """
    # In a real implementation, this would query the tool registry
    return [
        {"name": "search", "description": "Search for information"},
        {"name": "calculate", "description": "Perform calculations"},
        {"name": "summarize", "description": "Summarize text"}
    ]

def display_interactive_help() -> None:
    """Display help information for interactive mode."""
    print("\nGLUE Interactive Mode Help:")
    print("  /help       - Display this help message")
    print("  /exit       - Exit interactive mode")
    print("  /clear      - Clear conversation history")
    print("  /tools      - List available tools")
    print("  /status     - Display application status")
    print("  /teams      - Display team structure")
    print("  /verbose    - Toggle verbose mode")
    print("  /step       - Toggle step-by-step execution")
    print("  /color      - Toggle colored output")

def display_app_status(app: GlueApp) -> None:
    """Display the current status of the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print(f"\nApplication: {app.app_config.name}")
    print(f"Teams: {len(app.teams)}")
    print(f"Tools: {len(app.tools)}")
    print(f"Conversations: {len(app.conversations)}")

def display_available_tools(app: GlueApp) -> None:
    """Display the available tools in the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print("\nAvailable Tools:")
    for name, tool in app.tools.items():
        print(f"  {name} - {tool.description}")

def display_team_structure(app: GlueApp) -> None:
    """Display the team structure of the GLUE application.
    
    Args:
        app: The GLUE application
    """
    print("\nTeam Structure:")
    for name, team in app.teams.items():
        print(f"  {name}:")
        for member in team.get('members', []):
            print(f"    {member}")
BUILT_IN_TOOLS = ["web_search", "file_handler", "code_interpreter"]
MAGNETIC_TYPES = ["push", "pull", "bidirectional", "repel"]
ADHESIVE_TYPES = ["glue", "velcro", "tape"]

# ==================== Logging Setup ====================
def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.expanduser("~/.glue/logs/glue.log"), 
                               mode='a')
        ]
    )
    # Ensure log directory exists
    os.makedirs(os.path.expanduser("~/.glue/logs"), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("glue")
    logger.setLevel(level)
    
    return logger

# ==================== App Running Functions ====================
async def run_app(config_file: str, input_text: str = None, interactive: bool = False):
    """Run a GLUE app with the specified configuration
    
    Args:
        config_file: Path to GLUE configuration file
        input_text: Optional initial input text
        interactive: Whether to run in interactive mode
    """
    logger = logging.getLogger("glue.run_app")
    logger.info(f"Initializing app from {config_file}")
    
    # Create and setup app
    app = GlueApp(config_file=config_file)
    app.setup()
    
    try:
        if interactive:
            await interactive_session(app)
        elif input_text:
            logger.info(f"Processing input: {input_text}")
            response = await app.run(input_text)
            print(f"Response: {response}")
        else:
            # Default input
            default_greeting = "Hello! What can you help me with today?"
            logger.info(f"Using default greeting: {default_greeting}")
            response = await app.run(default_greeting)
            print(f"Response: {response}")
    finally:
        logger.info("Closing app")
        await app.close()

from .cliHelpers import (
    parse_interactive_command,
    format_agent_message,
    format_agent_interactions,
    colorize_agent_output,
    get_interactive_help_text
)

async def interactive_session(app: GlueApp):
    """Run an interactive session with the app
    
    Args:
        app: Initialized GLUE application
    """
    logger = logging.getLogger("glue.interactive")
    logger.info("Starting interactive session")
    
    # Display app info
    print(f"=== Welcome to {app.app_config.name} ===")
    print("Type '/exit' or '/quit' to end the session")
    print("Type '/help' to see available commands")
    
    # Use a consistent conversation ID for the session
    conv_id = "interactive"
    
    # Feature flags
    verbose_mode = False
    step_mode = False
    color_enabled = False
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            # Parse commands
            command, args = parse_interactive_command(user_input)
            
            # Handle special commands
            if command in ('exit', 'quit'):
                print("Exiting session...")
                break
            elif command == 'help':
                print(get_interactive_help_text())
                continue
            elif command == 'status':
                display_app_status(app)
                continue
            elif command == 'tools':
                display_available_tools(app)
                continue
            elif command == 'teams':
                display_team_structure(app)
                continue
            elif command == 'clear':
                app.clear_memory()
                print("Conversation memory cleared.")
                continue
            elif command == 'verbose':
                verbose_mode = not verbose_mode
                print(f"Verbose mode {'enabled. Showing agent interactions.' if verbose_mode else 'disabled.'}")
                continue
            elif command == 'step':
                if step_mode:
                    # Exit step mode if already in it
                    step_mode = False
                    await app.end_step_execution()
                    print("Step-by-step mode disabled.")
                else:
                    step_mode = True
                    print("Step-by-step mode enabled. Use /next to advance.")
                continue
            elif command == 'next':
                if not step_mode:
                    print("Not in step-by-step mode. Use /step to enable it first.")
                    continue
                    
                # Get the next step in execution
                step_result = await app.next_step()
                if step_result:
                    agent = step_result.get("agent", "system")
                    message = step_result.get("message", "No message")
                    formatted = format_agent_message(agent, message, color_enabled)
                    print(formatted)
                else:
                    print("\n[System] Execution complete.")
                continue
            elif command == 'color':
                if args and args[0].lower() in ('on', 'true', 'yes', 'enable'):
                    color_enabled = True
                    print("Color output enabled.")
                elif args and args[0].lower() in ('off', 'false', 'no', 'disable'):
                    color_enabled = False
                    print("Color output disabled.")
                else:
                    print("Usage: /color [on|off]")
                continue
                
            # Process regular input
            print("\nProcessing...")
            
            if step_mode:
                # Start step-by-step execution
                await app.begin_step_execution(user_input, conv_id=conv_id)
                print("\n[System] Started step-by-step execution. Use /next to advance.")
            else:
                # Normal execution
                response = await app.run(user_input, conv_id=conv_id)
                
                # Handle response based on mode
                if verbose_mode and isinstance(response, dict):
                    # Display agent interactions in verbose mode
                    final_response = response.get("final_response", "No response")
                    interactions = response.get("agent_interactions", [])
                    
                    # Print each interaction
                    for interaction in interactions:
                        agent = interaction.get("agent", "unknown")
                        message = interaction.get("message", "")
                        formatted = format_agent_message(agent, message, color_enabled)
                        print(formatted)
                    
                    # Print final response
                    formatted = format_agent_message("assistant", final_response, color_enabled)
                    print(formatted)
                else:
                    # Standard output
                    if isinstance(response, dict) and "final_response" in response:
                        response = response["final_response"]
                    
                    if color_enabled:
                        response = colorize_agent_output("assistant", response)
                    
                    print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nSession interrupted!")
            break
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}", exc_info=True)
            print(f"\nError: {str(e)}")

def display_interactive_help():
    """Display help for interactive session commands"""
    print("\nAvailable Commands:")
    print("  help    - Show this help message")
    print("  status  - Show app status and configuration")
    print("  tools   - Show available tools and their configurations")
    print("  teams   - Show team structure and information flow")
    print("  exit    - Exit the session")
    print("  quit    - Exit the session")

def display_app_status(app: GlueApp):
    """Display current app status
    
    Args:
        app: Current GLUE application
    """
    print("\n=== App Status ===")
    print(f"Name: {app.app_config.name}")
    print(f"Development mode: {app.app_config.development}")
    print(f"Sticky: {getattr(app.app_config, 'sticky', False)}")
    print(f"Models: {len(app.models)} configured")
    print(f"Teams: {len(app.teams)} configured")
    
def display_available_tools(app: GlueApp):
    """Display available tools in the app
    
    Args:
        app: Current GLUE application
    """
    print("\n=== Available Tools ===")
    
    # Group tools by team
    team_tools = {}
    for team_name, team in app.teams.items():
        team_tools[team_name] = team.tools
    
    for team_name, tools in team_tools.items():
        print(f"\nTeam: {team_name}")
        for tool_name in tools:
            adhesives = []
            for model in app.models.values():
                if hasattr(model, "adhesives") and tool_name in model.tools:
                    adhesives.extend(model.adhesives)
            adhesives = list(set(adhesives))  # Remove duplicates
            
            print(f"  - {tool_name}")
            if adhesives:
                print(f"    Adhesives: {', '.join(adhesives)}")

def display_team_structure(app: GlueApp):
    """Display team structure and information flow
    
    Args:
        app: Current GLUE application
    """
    print("\n=== Team Structure ===")
    
    for team_name, team in app.teams.items():
        print(f"\nTeam: {team_name}")
        print(f"  Lead: {team.lead}")
        
        if hasattr(team, "members") and team.members:
            print(f"  Members: {', '.join(team.members)}")
        else:
            print("  Members: None")
            
        if hasattr(team, "tools") and team.tools:
            print(f"  Tools: {', '.join(team.tools)}")
        else:
            print("  Tools: None")
    
    # Display information flow if available
    if hasattr(app, "magnetic_field") and app.magnetic_field:
        print("\nInformation Flow:")
        for source, targets in app.magnetic_field.flow.items():
            for target, flow_type in targets.items():
                flow_symbol = "→" if flow_type == "push" else "←" if flow_type == "pull" else "↔" if flow_type == "bidirectional" else "⊥"
                print(f"  {source} {flow_symbol} {target} ({flow_type})")

# ==================== Project Management Functions ====================
def create_new_project(project_name: str, template: str = "basic"):
    """Create a new GLUE project with template files
    
    Args:
        project_name: Name of the project to create
        template: Template type to use (basic, research, or chat)
    """
    logger = logging.getLogger("glue.new_project")
    
    # Check if directory already exists
    if os.path.exists(project_name):
        print(f"Error: Directory '{project_name}' already exists")
        return
    
    # Create project directory
    os.makedirs(project_name)
    
    # Create subdirectories
    os.makedirs(os.path.join(project_name, "models"), exist_ok=True)
    os.makedirs(os.path.join(project_name, "tools"), exist_ok=True)
    os.makedirs(os.path.join(project_name, "configs"), exist_ok=True)
    
    # Create app.glue based on template
    app_glue = get_template_content(template, project_name)
    
    with open(os.path.join(project_name, "app.glue"), "w") as f:
        f.write(app_glue)
    
    # Create .env.example
    env_example = '''# Required API Keys
OPENROUTER_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Web Search API Keys (Optional)
SERP_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
'''
    
    with open(os.path.join(project_name, ".env.example"), "w") as f:
        f.write(env_example)
    
    # Create README.md
    readme_content = f'''# {project_name}

A GLUE framework application for AI orchestration.

## Setup

1. Copy `.env.example` to `.env`
2. Add your API keys to `.env`
3. Run the application with:

```
glue run app.glue
```

## Project Structure

- `app.glue`: Main application configuration
- `models/`: Custom model definitions
- `tools/`: Custom tool implementations
- `configs/`: Additional configuration files
'''
    
    with open(os.path.join(project_name, "README.md"), "w") as f:
        f.write(readme_content)
    
    logger.info(f"Created new project: {project_name}")
    print(f"Created new GLUE project: {project_name}")
    print(f"Edit {project_name}/app.glue to customize your application")
    print("Set up your API keys according to the .env.example file")
    print(f"Run your app with: glue run {project_name}/app.glue")

def get_template_content(template: str, project_name: str) -> str:
    """Get template content based on template type
    
    Args:
        template: Template type (basic, research, or chat)
        project_name: Name of the project
        
    Returns:
        str: Template content for app.glue
    """
    if template == "research":
        return f'''glue app {{
    name = "{project_name}"
    config {{
        development = true
        sticky = true  // Enable persistence
    }}
}}

// Define tools
tool web_search {{
    provider = serp  // Will use SERP_API_KEY from environment
}}

tool file_handler {{
    config {{
        base_path = "./workspace"  // Set base path for file operations
    }}
}}

tool code_interpreter {{}}

// Models define their tool relationships
model researcher {{
    provider = openrouter
    role = "Research different topics and subjects online."
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }}
}}

model assistant {{
    provider = openrouter
    role = "Help with research and coding tasks."
    adhesives = [glue, velcro]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.5
    }}
}}

model writer {{
    provider = openrouter
    role = "Write documentation summarizing findings."
    adhesives = [tape]
    config {{
        model = "mistralai/mistral-7b-instruct:free"
        temperature = 0.3
    }}
}}

// Workflow defines model interactions and memory
magnetize {{
    research {{
        lead = researcher
        members = [assistant]
        tools = [web_search, code_interpreter]
    }}

    docs {{
        lead = writer
        tools = [web_search, file_handler]
    }}

    flow {{
        research -> docs   // Research flows to documentation
        docs <- pull      // Docs can pull from research when needed
    }}
}}

apply glue
'''
    elif template == "chat":
        return f'''glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define tools
tool web_search {{
    provider = serp
}}

tool file_handler {{}}

tool code_interpreter {{}}

// Define models
model chat_assistant {{
    provider = openrouter
    role = "You are a helpful AI assistant."
    adhesives = [glue, velcro, tape]
    config {{
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }}
}}

// Define teams
magnetize {{
    chat {{
        lead = chat_assistant
        tools = [web_search, file_handler, code_interpreter]
    }}
}}

apply glue
'''
    else:  # Default basic template
        return f'''glue app {{
    name = "{project_name}"
    config {{
        development = true
    }}
}}

// Define tools
tool web_search {{
    provider = serp
}}

tool code_interpreter {{}}

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
        tools = [web_search, code_interpreter]
    }}
}}

apply glue
'''

# ==================== Tool Management Functions ====================
def list_tools():
    """List all available tools in the framework"""
    print("\n=== Available Tools ===")
    
    # Built-in tools
    print("\nBuilt-in Tools:")
    print("-" * 50)
    
    for tool_name in BUILT_IN_TOOLS:
        print(f"\n{tool_name}")
        if tool_name == "web_search":
            print("  Providers: serp, tavily")
            print("  Description: Search the web for information")
            print("  Example:")
            print("  ```")
            print("  tool web_search {")
            print("      provider = serp  // Uses SERP_API_KEY from environment")
            print("  }")
            print("  ```")
        elif tool_name == "file_handler":
            print("  Description: Read and write files in the workspace")
            print("  Example:")
            print("  ```")
            print("  tool file_handler {")
            print("      config {")
            print("          base_path = \"./workspace\"")
            print("      }")
            print("  }")
            print("  ```")
        elif tool_name == "code_interpreter":
            print("  Description: Execute code in a sandboxed environment")
            print("  Example:")
            print("  ```")
            print("  tool code_interpreter {")
            print("      config {")
            print("          languages = [\"python\", \"javascript\"]")
            print("          sandbox = true")
            print("      }")
            print("  }")
            print("  ```")
    
    # Custom tools
    custom_tools = list_custom_tools()
    if custom_tools:
        print("\nCustom Tools:")
        print("-" * 50)
        
        for tool_name in custom_tools:
            print(f"\n{tool_name}")
            print(f"  Path: {custom_tools[tool_name]}")
            print("  Example:")
            print("  ```")
            print(f"  tool {tool_name} {{")
            print("      // Custom configuration")
            print("  }")
            print("  ```")

def list_custom_tools() -> Dict[str, str]:
    """List custom tools in the project
    
    Returns:
        Dict[str, str]: Dictionary of tool names to their file paths
    """
    custom_tools = {}
    
    # Check current directory for tools
    tools_dir = Path("tools")
    if tools_dir.exists() and tools_dir.is_dir():
        for tool_file in tools_dir.glob("**/*.py"):
            # Extract tool name from filename
            tool_name = tool_file.stem.replace("_", "-")
            custom_tools[tool_name] = str(tool_file)
    
    return custom_tools

# ==================== Model Management Functions ====================
def list_models():
    """List available model providers and models"""
    print("\n=== Available Model Providers ===")
    
    # Basic provider information
    providers = {
        "openrouter": {
            "models": ["meta-llama/llama-3.1-8b-instruct:free", "mistralai/mistral-7b-instruct:free", 
                      "anthropic/claude-3-haiku:free", "google/gemma-7b-it:free"],
            "adhesives": ["glue", "velcro", "tape"],
            "description": "Multiple models through OpenRouter API (FREE tier)"
        },
        "anthropic": {
            "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
            "adhesives": ["glue", "velcro", "tape"],
            "description": "Claude models from Anthropic"
        },
        "openai": {
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "adhesives": ["glue", "velcro", "tape"],
            "description": "OpenAI models (planned for future release)"
        }
    }
    
    for provider, info in providers.items():
        print(f"\n{provider}")
        print(f"  Description: {info['description']}")
        print(f"  Supported adhesives: {', '.join(info['adhesives'])}")
        print("  Available models:")
        for model in info["models"]:
            print(f"    - {model}")
        
        print("\n  Example:")
        print("  ```")
        print(f"  model my_model {{")
        print(f"      provider = {provider}")
        print("      role = \"Your role description\"")
        print("      adhesives = [glue, velcro]")
        print("      config {{")
        print(f"          model = \"{info['models'][0]}\"")
        print("          temperature = 0.7")
        print("      }}")
        print("  }")
        print("  ```")

# ==================== DSL Validation Functions ====================
def validate_glue_file(file_path: str) -> bool:
    """Validate a GLUE file for syntax errors
    
    Args:
        file_path: Path to GLUE file
        
    Returns:
        bool: True if file is valid, False otherwise
        
    Raises:
        ValueError: If syntax errors are found
    """
    logger = logging.getLogger("glue.validate")
    
    try:
        # Read the file
        with open(file_path, "r") as f:
            content = f.read()
        
        # Create lexer and parser
        lexer = GlueLexer(content)
        tokens = lexer.tokenize()
        
        # Create parser and parse
        parser = GlueDSLParser(tokens)
        ast = parser.parse()
        
        # If no exceptions are raised, the file is valid
        print(f"File {file_path} is valid!")
        print(f"Found {len(ast)} top-level declarations")
        return True
        
    except Exception as e:
        logger.error(f"Validation error in {file_path}: {str(e)}")
        print(f"Error validating {file_path}: {str(e)}")
        raise ValueError(f"Syntax error: {str(e)}")

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
                          choices=["basic", "research", "chat"],
                          default="basic",
                          help="Project template to use")
    
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
            
            # Run the application
            asyncio.run(run_app(args.config, args.input, args.interactive))
            
        elif args.command == "new":
            create_new_project(args.project, args.template)
            
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
            
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()