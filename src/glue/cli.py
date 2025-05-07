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
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Rich UI components
from rich import print
from rich.console import Console, Group
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.layout import Layout
from rich.traceback import install as install_rich_traceback
from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree
from rich.rule import Rule
from rich.theme import Theme
from rich.status import Status

# Import framework modules
# Use direct imports to avoid circular imports
from glue.core import GlueApp
from glue.dsl.parser import GlueParser
from glue.dsl.lexer import GlueLexer

# Import new utilities
from .utils.json_utils import extract_json
from .cliHelpers import parse_interactive_command, get_interactive_help_text
from .utils.ui_utils import display_warning, set_cli_config

# Constants for tools
def get_available_tools():
    """retrieve available tools from the tool registry."""

    # Fallback if dynamic retrieval fails
    return {
        "web_search": {
            "description": "Search the web for information",
            "parameters": {"query": "string"},
        },
        "file_handler": {
            "description": "Read and write files",
            "parameters": {"path": "string", "content": "string (optional)"},
        },
        "code_interpreter": {
            "description": "Execute Python code",
            "parameters": {"code": "string"},
        },
    }


# Replace the hardcoded constant with a function call
AVAILABLE_TOOLS = get_available_tools()

# Version information
__version__ = "0.1.0-alpha"  # Updated for alpha release

# Constants
DEFAULT_ENV_FILE = ".env"
CONFIG_DIR = os.path.expanduser("~/.glue")
LOGS_DIR = os.path.join(CONFIG_DIR, "logs")
TEMPLATES_DIR = os.path.join(CONFIG_DIR, "templates")
WORKSPACE_DIR = os.path.join(CONFIG_DIR, "workspace")

# Define custom Rich theme
GLUE_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "danger": "bold red",
        "success": "bold green",
        "command": "bold magenta",
        "heading": "bold blue",
        "filename": "cyan",
        "filepath": "bold cyan",
        "prompt": "bold cyan",
        "input": "white",
        "muted": "dim white",
        "title": "bold green",
        "subtitle": "dim green",
        "url": "underline cyan",
        "error": "red",
        "debug": "dim",
        "team": "bold blue",
        "tool": "bold yellow",
        "code": "green",
        "json": "yellow",
        "highlight": "bold white on blue",
        "app.name": "bold cyan",
        "app.value": "white",
    }
)

# CLI Configuration
CLI_CONFIG = {
    "theme": {
        "app_name": "app.name",
        "section_title": "heading",
        "success": "success",
        "error": "danger",
        "warning": "warning",
        "info": "info",
        "command": "command",
        "prompt": "prompt",
        "input": "input",
        "tool_call": "tool",
        "team_style": "team",
        "danger_style": "error",
    },
    "layout": {
        "show_header": True,
        "show_footer": True,
        "compact_mode": False,
        "use_animations": True,
        "loading_animations": {
            "dots": "dots",
            "dots2": "dots2",
            "dots3": "dots3",
            "dots4": "dots4",
            "line": "line",
            "bounce": "bounce",
            "pulse": "pulse",
        },
    },
    "display": {
        "show_timestamps": True,
        "verbose_errors": False,
        "color_enabled": True,
        "show_emoji": True,
        "use_unicode_symbols": True,
        "box_style": ROUNDED,
        "table_box": ROUNDED,
        "panel_box": ROUNDED,
        "interactive_mode": {
            "show_thinking": False,
            "step_by_step": False,
            "history_length": 10,
            "show_typing_animation": True,
        },
    },
    "emoji": {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "tool": "🔧",
        "model": "🧠",
        "team": "👥",
        "config": "⚙️",
        "file": "📄",
        "folder": "📁",
        "app": "🚀",
        "search": "🔍",
        "loading": "⏳",
        "run": "▶️",
        "stop": "⏹️",
    },
}


# Add a function to return the CLI_CONFIG
def get_cli_config():
    """Return the CLI configuration.

    Returns:
        The CLI configuration dictionary
    """
    return CLI_CONFIG


# Export the CLI_CONFIG to the ui_utils module
set_cli_config(CLI_CONFIG)

# ASCII art logo with gradient color styling
GLUE_LOGO = r"""[bold blue]
   ██████╗  ██╗      ██╗   ██╗ ███████╗
  ██╔════╝  ██║      ██║   ██║ ██╔════╝
  ██║  ███╗ ██║      ██║   ██║ █████╗  
  ██║   ██║ ██║      ██║   ██║ ██╔══╝  
  ╚██████╔╝ ███████╗ ╚██████╔╝ ███████╗
   ╚═════╝  ╚══════╝  ╚═════╝  ╚══════╝
[/bold blue][bold cyan]                                 
  GenAI Linking & Unification Engine v{__version__}
[/bold cyan]"""

# Command categories for help display
COMMAND_CATEGORIES = {
    "core": ["run", "new", "validate", "version"],
    "dev": ["forge", "list-tools", "list-models"],
    "interactive": [
        "help",
        "status",
        "tools",
        "teams",
        "clear",
        "verbose",
        "step",
        "color",
        "exit",
    ],
}

# Template types with descriptions
TEMPLATES = {
    "basic": "Simple app with a single model",
    "research": "Research-focused app with web search tools",
    "chat": "Chat application with multiple collaborating models",
    "agent": "Full agent system with autonomy and complex workflows",
}

# Install rich traceback handler with custom settings
install_rich_traceback(show_locals=True, width=None, suppress=[re])


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


def display_section_header(
    console: Console, title: str, emoji: Optional[str] = None
) -> None:
    """Display a section header.

    Args:
        console: Rich console to use for display
        title: Section title
        emoji: Optional emoji to display before the title
    """
    if CLI_CONFIG["display"]["show_emoji"] and emoji and emoji in CLI_CONFIG["emoji"]:
        emoji_str = f"{CLI_CONFIG['emoji'][emoji]} "
    elif emoji:
        emoji_str = f"{emoji} "
    else:
        emoji_str = ""

    rule = Rule(f"{emoji_str}{title}", style=CLI_CONFIG["theme"]["section_title"])
    console.print(rule)


# Add a helper for enhanced prompts with help and choices
def prompt_with_help(
    prompt_text,
    default=None,
    help_text=None,
    icon=None,
    console=None,
    choices=None,
    style="cyan",
    help_style="blue",
    choice_style="yellow",
):
    """
    Enhanced prompt for user input with rich visual styling, help text, icons, and choices.

    Args:
        prompt_text: The main prompt/question (str)
        default: Default value (optional)
        help_text: Help or tooltip to show below the prompt (optional)
        icon: Emoji or icon to show before the prompt (optional)
        console: Rich Console to use (optional)
        choices: List of choices to show (optional)
        style: Color style for the main prompt text (default: "cyan")
        help_style: Color style for the help panel (default: "blue")
        choice_style: Color style for the choice numbers (default: "yellow")

    Returns:
        User input (str), or default if input is empty
    """
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.box import ROUNDED, HEAVY
    from rich import get_console

    if console is None:
        console = get_console()

    # Show choices as a styled table if provided
    if choices:
        from rich.table import Table
        from rich.padding import Padding

        # Create a more visually appealing table with better spacing
        table = Table(show_header=False, box=ROUNDED, expand=False, padding=(0, 1))
        table.add_column("#", style=f"bold {choice_style}", justify="right", width=4)
        table.add_column("Option", style=f"bold {style}")

        for i, choice in enumerate(choices, 1):
            table.add_row(str(i), str(choice))

        # Add padding around the table for better visual spacing
        padded_table = Padding(table, (1, 0))
        console.print(padded_table)

    # Compose the prompt line with improved styling
    prompt_line = Text()

    # Add icon with proper spacing if provided
    if icon:
        prompt_line.append(f"{icon} ", style="bold")

    # Add the main prompt text with specified style
    prompt_line.append(prompt_text, style=f"bold {style}")

    # Add styled default value in the exact format requested
    if default is not None:
        default_text = f" [default: {default}]"
        prompt_line.append(default_text, style=f"dim {style}")

    # Print help text with enhanced styling if provided
    if help_text:
        from rich.markdown import Markdown

        # Handle help text as markdown for better formatting options
        if help_text.startswith("#") or "*" in help_text:
            help_content = Markdown(help_text)
        else:
            help_content = Text(help_text, style="dim")

        help_panel = Panel(
            help_content,
            title="[dim]Help[/dim]",
            border_style=help_style,
            box=HEAVY,  # More prominent box style
            padding=(1, 2),  # Better internal padding
        )
        console.print(help_panel)

    # Ask for input with a visual separator before
    console.print("─" * console.width, style=f"dim {style}")

    # Use show_default=False to prevent Rich from adding its own default display format
    user_input = Prompt.ask(
        prompt_line, default=default, console=console, show_default=False
    )
    console.print("─" * console.width, style=f"dim {style}")

    return user_input


# ==================== Application Functions ====================
async def run_app(
    config_file: str, interactive: bool = False, input_text: str = None, engine: str = 'agno'
) -> bool:
    """Run a GLUE application from a configuration file.

    Args:
        config_file: Path to the GLUE configuration file
        interactive: Whether to run in interactive mode
        input_text: Input text to process (for non-interactive mode)
        engine: The execution engine to use ('glue' or 'agno')

    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger("glue.run_app")
    console = (
        get_console()
    )  # Use get_console() instead of Console() to ensure theme is applied

    if engine == 'agno':
        logger.info(f"Agno engine selected. Config file: {config_file}")
        if not os.path.exists(config_file):
            logger.error(f"Agno engine: Configuration file not found: {config_file}")
            print(f"[bold red]Error (Agno):[/bold red] Configuration file not found: {config_file}")
            return False
        
        try:
            # Read the GLUE file
            with open(config_file, "r") as f:
                glue_content = f.read()

            # Create lexer and parser
            lexer = GlueLexer()
            parser = GlueParser()
            
            # Parse the GLUE DSL into an AST
            tokens = lexer.tokenize(glue_content)
            ast = parser.parse(tokens)
            
            if not ast:
                logger.error("Agno engine: Failed to parse GLUE DSL")
                print(f"[bold red]Error (Agno):[/bold red] Failed to parse GLUE DSL")
                return False
            
            # Translate the AST to Agno configuration
            from glue.core.adapters.agno import GlueDSLAgnoTranslator, GlueAgnoAdapter
            
            translator = GlueDSLAgnoTranslator()
            agno_config = translator.translate(ast)
            
            logger.info(f"Translated GLUE DSL to Agno configuration")
            
            # Create and run the Agno adapter
            adapter = GlueAgnoAdapter()
            result = adapter.run(agno_config)
            
            if result is None:
                logger.error("Agno engine: Failed to run Agno workflow")
                print(f"[bold red]Error (Agno):[/bold red] Failed to run Agno workflow")
                return False
                
            logger.info("Agno engine: Successfully ran Agno workflow")
            print(f"[bold green]Success (Agno):[/bold green] Successfully ran Agno workflow")
            return True
            
        except Exception as e:
            logger.error(f"Agno engine: Error running Agno workflow: {str(e)}")
            print(f"[bold red]Error (Agno):[/bold red] {str(e)}")
            return False

    # Existing GLUE engine logic starts here
    logger.info(f"GLUE engine selected. Parsing GLUE file: {config_file}")
    print(f"[green]Using GLUE engine. Loading configuration: {config_file}[/green]")

    try:
        # Check if file exists
        if not os.path.exists(config_file):
            print(f"[bold red]Error:[/bold red] Configuration file not found: {config_file}")
            return False

        logger.info(f"Parsing GLUE file: {config_file}")

        # Read the GLUE file
        with open(config_file, "r") as f:
            glue_content = f.read()

        # Create lexer and parser
        lexer = GlueLexer()
        parser = GlueParser()

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
            if (
                isinstance(app_config, dict)
                and "portkey" in app_config
                and app_config["portkey"]
            ):
                portkey_enabled = True

        # If Portkey is enabled, check for API key
        if portkey_enabled:
            logger.info("Portkey integration is enabled")
            portkey_api_key = os.environ.get("PORTKEY_API_KEY")
            if not portkey_api_key:
                logger.warning("Portkey API key not found in environment variables")
                print(
                    "Warning: Portkey is enabled but PORTKEY_API_KEY is not set in environment variables"
                )
            else:
                logger.info("Portkey API key found")
                # Configure Portkey here (implementation depends on how Portkey is integrated)

        # Build the application
        logger.info("Building GLUE application")

        # Create a new GlueApp instance with the parsed config
        app = GlueApp(config=ast)

        # Set the interactive flag on the app instance
        app.interactive = interactive
        logger.debug(f"Set app.interactive to {interactive}")

        # Setup the app first
        logger.debug("Setting up application")
        await app.setup()
        logger.debug("Application setup complete")

        # Automatically start agent loops for all teams so members process tasks as soon as they're assigned
        import asyncio

        for team in app.teams.values():
            # Pass initial input_text so leads can start delegation and members can fetch tasks
            asyncio.create_task(team.start_agent_loops(initial_input=input_text))
            logger.info(f"Auto-started agent loops for team {team.name}")

        # Display available tools if in interactive mode
        if interactive:
            logger.info("Starting interactive session")
            print(f"\nStarting interactive session with {app.name}")
            logger.debug("Entering interactive session")
            await run_interactive_session(app)
            logger.debug("Interactive session complete")
            return True

        # Non-interactive mode
        elif input_text:
            logger.info(
                "Running non-interactive agentic workflow with programmatic orchestrator"
            )
            # Use the programmatic TeamLeadAgentLoop for proper asynchronous delegation
            from glue.core.agent_loop import TeamLeadAgentLoop

            first_team_name = next(iter(app.teams))
            team = app.teams[first_team_name]
            # Prepare the delegate_task tool executable
            delegate_tool_exec = team._tools.get("delegate_task")
            if hasattr(delegate_tool_exec, "execute"):
                delegate_tool_exec = delegate_tool_exec.execute
            # Instantiate and run the lead orchestrator loop
            lead_loop = TeamLeadAgentLoop(
                team=team, delegate_tool=delegate_tool_exec, agent_llm=team.lead
            )
            # Run orchestration with the initial goal
            final_json = await lead_loop.start(
                parent_task_id=first_team_name, goal_description=input_text
            )
            # Display the final JSON response
            from rich.syntax import Syntax
            from rich.align import Align

            syntax = Syntax(final_json, "json", theme="monokai", line_numbers=False)
            console.print(
                Panel(
                    syntax,
                    title="Final JSON Response",
                    border_style="success",
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2),
                )
            )
            console.print(
                Panel(
                    Align.center("Thank you for using GLUE Framework!", style="cyan"),
                    border_style="success",
                    box=CLI_CONFIG["display"]["panel_box"],
                )
            )
            return True
        else:
            logger.error("No input provided for non-interactive mode")
            console.print(
                "[bold red]Error:[/bold red] No input provided for non-interactive mode"
            )
            return False

    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        console.print(f"[bold red]Error running application:[/bold red] {e}")
        return False


async def run_interactive_session(app: GlueApp) -> None:
    """Run an enhanced interactive session with the GLUE application.

    This function provides a command-line interface for interacting with GLUE applications,
    allowing users to interact with AI teams and models, execute commands, and manage
    the conversation flow.

    Args:
        app: The GLUE application to run
    """
    logger = logging.getLogger("glue.interactive")
    console = get_console()

    # State management for interactive session with better comments
    state = {
        "history": [],  # Complete message history (user & AI)
        "verbose_mode": False,  # Show detailed processing messages
        "step_mode": False,  # Enable step-by-step execution
        "color_enabled": True,  # Enable/disable colored output
        "thinking_visible": False,  # Show model reasoning process
        "current_team": None,  # Active team context (if any)
        "current_model": None,  # Active model context (if any)
        "awaiting_next_step": False,  # Step mode pause state
        "last_command": None,  # Track last command for repeats
        "auto_refresh": True,  # Auto-refresh display after operations
    }

    # Track conversation history indices per team to print new messages in background
    state["history_indices"] = {team_name: 0 for team_name in app.teams}

    async def background_printer():
        import asyncio
        from rich.panel import Panel

        while True:
            for team_name, team in app.teams.items():
                hist = team.conversation_history
                last_idx = state["history_indices"].get(team_name, 0)
                for msg in hist[last_idx:]:
                    title = f"{team_name} | {getattr(msg, 'name', msg.role)}"
                    console.print(
                        Panel(msg.content, title=title, border_style="magenta")
                    )
                state["history_indices"][team_name] = len(hist)
            await asyncio.sleep(0.5)

    # Launch the background printer task
    import asyncio

    bg_task = asyncio.create_task(background_printer())

    # Display welcome banner with improved styling
    display_logo(console)

    # Create welcome message with better fallback handling
    app.name or "Anonymous Application"
    welcome_lines = [
        (
            "Welcome to the interactive GLUE session for:"
            if app.name
            else "Welcome to the interactive GLUE session!"
        ),
        app.name if app.name else "",
    ]

    # Create enhanced tip text with more helpful information
    from rich.text import Text

    tip_text = Text("Type ", style="")
    tip_text.append("/help", style="green bold")
    tip_text.append(" for commands • ", style="")
    tip_text.append("/exit", style="green bold")
    tip_text.append(" to quit • ", style="")
    tip_text.append("/status", style="green bold")
    tip_text.append(" for system info", style="")

    # Create enhanced welcome panel with better padding and styling
    welcome_panel = Panel(
        Group(
            Align.center(welcome_lines[0], style="cyan"),
            (
                Align.center(welcome_lines[1], style="bold cyan")
                if welcome_lines[1]
                else None
            ),
            Rule(style="dim"),
            Align.center(tip_text),
        ),
        title="🤖 Interactive Mode",
        title_align="center",
        border_style="green",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 3),
    )
    console.print(welcome_panel)

    # Display comprehensive session information
    emoji_enabled = CLI_CONFIG["display"]["show_emoji"]
    emoji = "info" if emoji_enabled else None
    display_section_header(console, "Session Information", emoji=emoji)

    # Show improved application structure with equal column sizing
    columns = Columns(
        [create_status_panel(app, state), create_model_tree(app.teams)],
        equal=True,
        expand=True,
    )
    console.print(columns)

    # Show available tools with better filtering
    if app.tools:
        emoji = "tool" if emoji_enabled else None
        display_section_header(console, "Available Tools", emoji=emoji)
        # Filter out internal communication tools for cleaner display
        filtered_tools = {
            name: tool for name, tool in app.tools.items() if name not in ["_internal"]
        }
        console.print(create_tool_table(filtered_tools))

    # Enhanced initial instructions
    console.print(
        "\n[info]Type your message or command below (commands start with '/'):[/info]"
    )

    # Command history with improved navigation
    command_history = []

    # Main interaction loop with enhanced error handling
    while True:
        try:
            # Create prompt based on current context with better styling
            prompt_style = CLI_CONFIG["theme"]["prompt"]
            CLI_CONFIG["theme"]["input"]

            if state["current_team"]:
                team_style = "team"
                prompt_text = f"[{prompt_style}][{team_style}]{state['current_team']}[/{team_style}]>[/{prompt_style}] "
            elif state["current_model"]:
                model_style = CLI_CONFIG["theme"]["model"].get(
                    state["current_model"], CLI_CONFIG["theme"]["model"]["default"]
                )
                prompt_text = f"[{prompt_style}][{model_style}]{state['current_model']}[/{model_style}]>[/{prompt_style}] "
            else:
                prompt_text = f"[{prompt_style}]glue>[/{prompt_style}] "

            # Get user input using Rich's Prompt.ask for styled prompt
            user_input = await asyncio.to_thread(
                Prompt.ask, prompt_text, console=console, show_default=False
            )

            # More robust exit command handling
            if user_input.lower() in ["quit", "exit", "/exit", "/quit", "q", ":q"]:
                console.print(
                    f"[{CLI_CONFIG['theme']['success']}]Exiting interactive session.[/{CLI_CONFIG['theme']['success']}]"
                )
                # Cancel background printer on exit
                bg_task.cancel()
                break

            # Skip empty input but handle special case for repeating last command
            if not user_input:
                if state["last_command"]:
                    user_input = state["last_command"]
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Repeating last command: {user_input}[/{CLI_CONFIG['theme']['info']}]"
                    )
                else:
                    continue

            # Smart command history management - avoid duplicates
            if user_input not in command_history:
                command_history.append(user_input)
                len(command_history)
            elif command_history and command_history[-1] != user_input:
                # Move repeated command to end for better history navigation
                command_history.remove(user_input)
                command_history.append(user_input)
                len(command_history)

            # Process command if it starts with / with enhanced parsing
            if user_input.startswith("/"):
                # Save last command for potential repeat
                state["last_command"] = user_input

                # Parse command with better arg handling
                command, args = parse_interactive_command(user_input)

                # Enhanced command parsing and handling with better organization
                if command == "help":
                    show_interactive_help(console)
                    continue

                elif command == "status":
                    # Display enhanced status information
                    status_panel = create_status_panel(app, state)
                    console.print(status_panel)
                    continue

                elif command == "tools":
                    # Display available tools with better categorization
                    emoji = "tool" if emoji_enabled else None
                    display_section_header(console, "Available Tools", emoji=emoji)
                    filtered_tools = {
                        name: tool
                        for name, tool in app.tools.items()
                        if name not in ["_internal"]
                    }
                    console.print(create_tool_table(filtered_tools))
                    continue

                elif command == "teams":
                    # Enhanced team structure display
                    emoji = "team" if emoji_enabled else None
                    display_section_header(console, "Team Structure", emoji=emoji)
                    console.print(create_team_table(app.teams))
                    continue

                elif command == "models":
                    # Enhanced model structure display
                    emoji = "model" if emoji_enabled else None
                    display_section_header(console, "Model Structure", emoji=emoji)
                    console.print(create_model_tree(app.teams))
                    continue

                elif command == "clear":
                    # Clear conversation history with confirmation
                    if len(state["history"]) > 0:
                        console.print(
                            f"[{CLI_CONFIG['theme']['warning']}]Clearing {len(state['history'])} message(s) from history.[/{CLI_CONFIG['theme']['warning']}]"
                        )
                    state["history"] = []
                    console.print(
                        f"[{CLI_CONFIG['theme']['success']}]Conversation history cleared.[/{CLI_CONFIG['theme']['success']}]"
                    )
                    continue

                elif command == "verbose":
                    # Toggle verbose mode with better feedback
                    state["verbose_mode"] = not state["verbose_mode"]
                    mode = "enabled" if state["verbose_mode"] else "disabled"
                    icon = "✅" if state["verbose_mode"] else "❌"

                    # Adjust logger level based on verbose mode
                    if state["verbose_mode"]:
                        logger.setLevel(logging.DEBUG)
                    else:
                        logger.setLevel(logging.INFO)

                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Verbose mode {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                    )
                    continue

                elif command == "step":
                    # Toggle step-by-step mode with better feedback
                    state["step_mode"] = not state["step_mode"]
                    mode = "enabled" if state["step_mode"] else "disabled"
                    icon = "✅" if state["step_mode"] else "❌"

                    # Reset awaiting flag when turning off step mode
                    if not state["step_mode"]:
                        state["awaiting_next_step"] = False

                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Step-by-step mode {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                    )
                    continue

                elif (
                    command == "next"
                    and state["step_mode"]
                    and state["awaiting_next_step"]
                ):
                    # Proceed to next step in step mode with better feedback
                    state["awaiting_next_step"] = False
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Proceeding to next step...[/{CLI_CONFIG['theme']['info']}]"
                    )
                    # Execution will continue below

                elif command == "color":
                    # Toggle color output with improved argument handling
                    if len(args) > 0 and args[0].lower() in [
                        "on",
                        "off",
                        "true",
                        "false",
                        "1",
                        "0",
                    ]:
                        # Support more input formats
                        state["color_enabled"] = args[0].lower() in ["on", "true", "1"]
                        mode = "enabled" if state["color_enabled"] else "disabled"
                        icon = "✅" if state["color_enabled"] else "❌"
                        console.print(
                            f"[{CLI_CONFIG['theme']['info']}]Color output {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                        )
                    else:
                        # Toggle if no args provided
                        if len(args) == 0:
                            state["color_enabled"] = not state["color_enabled"]
                            mode = "enabled" if state["color_enabled"] else "disabled"
                            icon = "✅" if state["color_enabled"] else "❌"
                            console.print(
                                f"[{CLI_CONFIG['theme']['info']}]Color output {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                            )
                        else:
                            console.print(
                                f"[{CLI_CONFIG['theme']['warning']}]Usage: /color [on|off][/{CLI_CONFIG['theme']['warning']}]"
                            )
                    continue

                elif command == "team":
                    # Switch to a specific team with better feedback
                    if len(args) > 0:
                        team_name = args[0]
                        if team_name in app.teams:
                            state["current_team"] = team_name
                            state["current_model"] = None
                            console.print(
                                f"[{CLI_CONFIG['theme']['success']}]Switched to team: [team]{team_name}[/team][/{CLI_CONFIG['theme']['success']}]"
                            )

                            # Show team info for better context
                            team = app.teams[team_name]
                            if hasattr(team, "lead") and team.lead:
                                console.print(
                                    f"[{CLI_CONFIG['theme']['info']}]Team lead: [model]{team.lead.name}[/model][/{CLI_CONFIG['theme']['info']}]"
                                )
                            if hasattr(team, "members") and team.members:
                                members = ", ".join(
                                    [f"[model]{m.name}[/model]" for m in team.members]
                                )
                                console.print(
                                    f"[{CLI_CONFIG['theme']['info']}]Team members: {members}[/{CLI_CONFIG['theme']['info']}]"
                                )

                        else:
                            # Show available teams when not found
                            console.print(
                                f"[{CLI_CONFIG['theme']['error']}]Team not found: {team_name}[/{CLI_CONFIG['theme']['error']}]"
                            )
                            console.print(
                                f"[{CLI_CONFIG['theme']['info']}]Available teams: {', '.join(app.teams.keys())}[/{CLI_CONFIG['theme']['info']}]"
                            )
                    else:
                        # List available teams if no argument provided
                        if app.teams:
                            console.print(
                                f"[{CLI_CONFIG['theme']['info']}]Available teams: {', '.join(app.teams.keys())}[/{CLI_CONFIG['theme']['info']}]"
                            )
                            console.print(
                                f"[{CLI_CONFIG['theme']['warning']}]Usage: /team [team_name][/{CLI_CONFIG['theme']['warning']}]"
                            )
                        else:
                            console.print(
                                f"[{CLI_CONFIG['theme']['warning']}]No teams available in this application.[/{CLI_CONFIG['theme']['warning']}]"
                            )
                    continue

                elif command == "model":
                    # Switch to a specific model with better search logic
                    if len(args) > 0:
                        model_name = args[0]
                        # More efficient model lookup
                        model_found = False
                        model_team = None

                        # First try exact match
                        for team_name, team in app.teams.items():
                            if (
                                hasattr(team, "lead")
                                and team.lead
                                and team.lead.name == model_name
                            ):
                                model_found = True
                                model_team = team_name
                                break
                            if hasattr(team, "members"):
                                for member in team.members:
                                    if member.name == model_name:
                                        model_found = True
                                        model_team = team_name
                                        break
                                if model_found:
                                    break

                        # If not found, try case-insensitive match
                        if not model_found:
                            model_name_lower = model_name.lower()
                            for team_name, team in app.teams.items():
                                if (
                                    hasattr(team, "lead")
                                    and team.lead
                                    and team.lead.name.lower() == model_name_lower
                                ):
                                    model_name = team.lead.name  # Use correct case
                                    model_found = True
                                    model_team = team_name
                                    break
                                if hasattr(team, "members"):
                                    for member in team.members:
                                        if member.name.lower() == model_name_lower:
                                            model_name = member.name  # Use correct case
                                            model_found = True
                                            model_team = team_name
                                            break
                                    if model_found:
                                        break

                        if model_found:
                            state["current_model"] = model_name
                            state["current_team"] = None
                            model_style = CLI_CONFIG["theme"]["model"].get(
                                model_name, CLI_CONFIG["theme"]["model"]["default"]
                            )
                            console.print(
                                f"[{CLI_CONFIG['theme']['success']}]Switched to model: [{model_style}]{model_name}[/{model_style}][/{CLI_CONFIG['theme']['success']}]"
                            )
                            console.print(
                                f"[{CLI_CONFIG['theme']['info']}]Part of team: [team]{model_team}[/team][/{CLI_CONFIG['theme']['info']}]"
                            )
                        else:
                            # Show models when not found
                            console.print(
                                f"[{CLI_CONFIG['theme']['error']}]Model not found: {model_name}[/{CLI_CONFIG['theme']['error']}]"
                            )
                            # List available models
                            models = []
                            for team_name, team in app.teams.items():
                                if hasattr(team, "lead") and team.lead:
                                    models.append(team.lead.name)
                                if hasattr(team, "members"):
                                    models.extend([m.name for m in team.members])
                            if models:
                                console.print(
                                    f"[{CLI_CONFIG['theme']['info']}]Available models: {', '.join(models)}[/{CLI_CONFIG['theme']['info']}]"
                                )
                    else:
                        # List available models if no argument provided
                        models = []
                        for team_name, team in app.teams.items():
                            if hasattr(team, "lead") and team.lead:
                                models.append(team.lead.name)
                            if hasattr(team, "members"):
                                models.extend([m.name for m in team.members])
                        if models:
                            console.print(
                                f"[{CLI_CONFIG['theme']['info']}]Available models: {', '.join(models)}[/{CLI_CONFIG['theme']['info']}]"
                            )
                            console.print(
                                f"[{CLI_CONFIG['theme']['warning']}]Usage: /model [model_name][/{CLI_CONFIG['theme']['warning']}]"
                            )
                        else:
                            console.print(
                                f"[{CLI_CONFIG['theme']['warning']}]No models available in this application.[/{CLI_CONFIG['theme']['warning']}]"
                            )
                    continue

                elif command == "thinking":
                    # Toggle model thinking visibility
                    state["thinking_visible"] = not state["thinking_visible"]
                    mode = "visible" if state["thinking_visible"] else "hidden"
                    icon = "✅" if state["thinking_visible"] else "❌"
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Model thinking is now {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                    )
                    continue

                elif command == "refresh":
                    # Enhanced refresh display with more info
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Refreshing display...[/{CLI_CONFIG['theme']['info']}]"
                    )
                    # Show application structure with improved layout
                    columns = Columns(
                        [create_status_panel(app, state), create_model_tree(app.teams)],
                        equal=True,
                        expand=True,
                    )
                    console.print(columns)

                    # Show conversation stats
                    if state["history"]:
                        user_messages = sum(
                            1 for msg in state["history"] if msg.get("role") == "user"
                        )
                        ai_messages = sum(
                            1 for msg in state["history"] if msg.get("role") != "user"
                        )
                        console.print(
                            f"[{CLI_CONFIG['theme']['info']}]Conversation: {user_messages} user message(s), {ai_messages} AI response(s)[/{CLI_CONFIG['theme']['info']}]"
                        )
                    continue

                elif command == "auto-refresh":
                    # Toggle auto-refresh setting
                    if len(args) > 0 and args[0].lower() in [
                        "on",
                        "off",
                        "true",
                        "false",
                        "1",
                        "0",
                    ]:
                        state["auto_refresh"] = args[0].lower() in ["on", "true", "1"]
                    else:
                        state["auto_refresh"] = not state["auto_refresh"]

                    mode = "enabled" if state["auto_refresh"] else "disabled"
                    icon = "✅" if state["auto_refresh"] else "❌"
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Auto-refresh {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]"
                    )
                    continue

                elif command == "history":
                    # Display conversation history
                    if not state["history"]:
                        console.print(
                            f"[{CLI_CONFIG['theme']['info']}]No conversation history yet.[/{CLI_CONFIG['theme']['info']}]"
                        )
                        continue

                    # Parse optional limit argument
                    limit = None
                    if args and args[0].isdigit():
                        limit = int(args[0])

                    history_to_show = (
                        state["history"][-limit:] if limit else state["history"]
                    )

                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Conversation history ({len(history_to_show)} of {len(state['history'])} messages):[/{CLI_CONFIG['theme']['info']}]"
                    )

                    for i, msg in enumerate(history_to_show):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        role_style = "green" if role == "user" else "blue"
                        index = i + (len(state["history"]) - len(history_to_show))

                        console.print(
                            f"[{role_style}]{index + 1}. {role.upper()}:[/{role_style}]"
                        )
                        # Truncate very long messages
                        if len(content) > 500:
                            console.print(
                                f"{content[:500]}... [dim](truncated, {len(content)} chars)[/dim]"
                            )
                        else:
                            console.print(content)
                        console.print("")
                    continue

                else:
                    console.print(
                        f"[{CLI_CONFIG['theme']['error']}]Unknown command: {command}[/{CLI_CONFIG['theme']['error']}]"
                    )
                    console.print(
                        f"[{CLI_CONFIG['theme']['info']}]Type /help to see available commands[/{CLI_CONFIG['theme']['info']}]"
                    )
                    continue

            # Process regular input (non-command)
            logger.info(f"User input: {user_input}")
            state["history"].append({"role": "user", "content": user_input})

            # Get appropriate spinner based on config
            spinner = CLI_CONFIG["layout"]["loading_animations"].get("dots", "dots")

            # Improved response handling with better error detection
            try:
                # Run with or without status indicator based on verbosity level
                if state["verbose_mode"] or logger.level <= logging.INFO:
                    # In verbose mode, don't show spinner to avoid interfering with log output
                    logger.debug(
                        "Processing input without status indicator (verbose mode)"
                    )
                    response = await process_message(app, user_input, state)
                else:
                    # In normal mode, show spinner status indicator with improved messaging
                    with Status(
                        "[info]Processing message...[/info]",
                        spinner=spinner,
                        console=console,
                    ):
                        response = await process_message(app, user_input, state)

                # Determine the source (current team, model or app name)
                source = (
                    state.get("current_team") or state.get("current_model") or app.name
                )

                # Enhanced warning detection and processing
                if isinstance(response, str):
                    # Enhanced warning patterns with better categorization
                    warning_patterns = [
                        ("does not support tool use", "Tool Error"),
                        ("Warning:", "Warning"),
                        ("Error:", "Error"),
                        ("Retrying with", "Retry"),
                        ("Exception:", "Exception"),
                        ("Failed to", "Failure"),
                    ]

                    # Check if any pattern is in the response with improved detection
                    warning_found = False
                    for pattern, warning_type in warning_patterns:
                        if pattern in response:
                            warning_found = True
                            break

                    if warning_found:
                        # Process multi-line response to extract warnings with better categorization
                        lines = response.split("\n")
                        warning_lines = []
                        content_lines = []

                        for line in lines:
                            line = line.strip()
                            is_warning = False

                            for pattern, warning_type in warning_patterns:
                                if pattern in line:
                                    # Clean up the warning line with better formatting
                                    clean_line = line
                                    prefix = (
                                        pattern
                                        if pattern != "Warning:" and pattern != "Error:"
                                        else ""
                                    )
                                    if "Warning:" in line:
                                        clean_line = line[
                                            line.find("Warning:") + 8 :
                                        ].strip()
                                    elif "Error:" in line:
                                        clean_line = line[
                                            line.find("Error:") + 6 :
                                        ].strip()

                                    warning_lines.append(
                                        f"{warning_type}: {prefix}{clean_line}"
                                    )
                                    is_warning = True
                                    break

                            if not is_warning and line:  # Skip empty lines and warnings
                                content_lines.append(line)

                        # Display all warnings in a single panel with better styling
                        if warning_lines:
                            combined_warning = "\n".join(warning_lines)
                            display_warning(console, combined_warning)

                        # Update response to only include content
                        if content_lines:
                            response = "\n".join(content_lines)
                        else:
                            response = "I'll try to process your request differently."

                # Format and display the response with enhanced styling
                display_response(console, response, source, state)

                # Add response to history
                state["history"].append({"role": "assistant", "content": response})

                # In step mode, wait for user to continue with better UI
                if state["step_mode"]:
                    state["awaiting_next_step"] = True
                    next_step_panel = Panel(
                        Group(
                            Text("Type [green]/next[/green] to continue"),
                            Text("Or enter a new input to proceed"),
                        ),
                        title="Step Completed",
                        title_align="center",
                        border_style="info",
                        box=CLI_CONFIG["display"]["panel_box"],
                        padding=(1, 2),
                    )
                    console.print(next_step_panel)

                # Auto-refresh display if enabled
                if state["auto_refresh"] and not state["step_mode"]:
                    # Update status silently
                    status_panel = create_status_panel(app, state)
                    console.print("")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                console.print(
                    f"[{CLI_CONFIG['theme']['error']}]Error processing message: {str(e)}[/{CLI_CONFIG['theme']['error']}]"
                )

        except KeyboardInterrupt:
            console.print(
                f"\n[{CLI_CONFIG['theme']['warning']}]Interrupted. Press Ctrl+C again to exit or Enter to continue.[/{CLI_CONFIG['theme']['warning']}]"
            )
            try:
                # Give user a chance to cancel the exit
                if await asyncio.to_thread(lambda: input()) == "":
                    continue
                else:
                    break
            except KeyboardInterrupt:
                # Double interrupt = definite exit
                break
        except Exception as e:
            # Improved error handling with more context
            logger.error(f"Error during interactive session: {e}", exc_info=True)
            if CLI_CONFIG["display"]["verbose_errors"]:
                console.print_exception()
            else:
                error_panel = Panel(
                    str(e),
                    title="Error Occurred",
                    title_align="center",
                    border_style="error",
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2),
                )
                console.print(error_panel)

            # Offer recovery options
            console.print(
                f"[{CLI_CONFIG['theme']['info']}]Type /refresh to refresh the display or /help for commands.[/{CLI_CONFIG['theme']['info']}]"
            )

    # Graceful shutdown
    logger.debug("Interactive session ended.")
    console.print(
        f"[{CLI_CONFIG['theme']['success']}]Interactive session ended. Thank you for using GLUE![/{CLI_CONFIG['theme']['success']}]"
    )


# Helper function for processing messages with better error handling
async def process_message(app: GlueApp, user_input: str, state: dict) -> str:
    """Process a user message and route it to the appropriate handler.

    Args:
        app: The GLUE application
        user_input: The user's input text
        state: Current session state

    Returns:
        The response from the model or team
    """
    if state["current_team"]:
        # Direct message to specific team
        return await app.teams[state["current_team"]].process_message(user_input)
    elif state["current_model"]:
        # Direct message to specific model
        model = find_model_in_teams(app, state["current_model"])

        if model:
            return await model.generate_response(
                messages=[{"role": "user", "content": user_input}]
            )
        else:
            return f"Error: Model {state['current_model']} not found or not accessible."
    else:
        # Default routing through app
        return await app.run(user_input)


# Helper function to find model in teams with improved search
def find_model_in_teams(app: GlueApp, model_name: str):
    """Find a model by name across all teams.

    Args:
        app: The GLUE application
        model_name: Name of the model to find

    Returns:
        The model object or None if not found
    """
    # Check team leads first (more efficient)
    for team_name, team in app.teams.items():
        if hasattr(team, "lead") and team.lead and team.lead.name == model_name:
            return team.lead

    # Then check team members
    for team_name, team in app.teams.items():
        if hasattr(team, "members"):
            for member in team.members:
                if member.name == model_name:
                    return member

    # Not found
    return None


def display_response(
    console: Console, response: Any, source: str, state: Dict[str, Any]
) -> None:
    """Display a response from the application with enhanced visuals.

    Args:
        console: Rich console to use for display
        response: The response to display
        source: Source of the response (team or model name)
        state: The current session state
    """
    # Create a panel for the response
    console.print("")  # Add space before response

    # Determine the appropriate styling based on the source
    if state.get("current_team"):
        panel_style = CLI_CONFIG["theme"]["team_style"]
        emoji = (
            CLI_CONFIG["emoji"]["team"] if CLI_CONFIG["display"]["show_emoji"] else ""
        )
    elif state.get("current_model"):
        model_name = state.get("current_model")
        panel_style = CLI_CONFIG["theme"]["model"].get(
            model_name, CLI_CONFIG["theme"]["model"]["default"]
        )
        emoji = (
            CLI_CONFIG["emoji"]["model"] if CLI_CONFIG["display"]["show_emoji"] else ""
        )
    else:
        panel_style = CLI_CONFIG["theme"]["app_name"]
        emoji = (
            CLI_CONFIG["emoji"]["app"] if CLI_CONFIG["display"]["show_emoji"] else ""
        )

    title = f"{emoji} Response from {source}" if emoji else f"Response from {source}"

    # Format the response based on its type
    if isinstance(response, str):
        # Check for JSON tool call using the utility function
        tool_call_data = extract_json(response)

        if tool_call_data:
            # Format JSON for tool calls
            json_str = json.dumps(tool_call_data, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)

            # Add tool emoji if enabled
            tool_emoji = (
                CLI_CONFIG["emoji"]["tool"]
                if CLI_CONFIG["display"]["show_emoji"]
                else ""
            )
            tool_title = f"{tool_emoji} Tool Call" if tool_emoji else "Tool Call"

            # Create nested panels for tool calls
            tool_panel = Panel(
                syntax,
                title=tool_title,
                border_style="tool",
                box=CLI_CONFIG["display"]["panel_box"],
            )

            response_panel = Panel(
                tool_panel,
                title=title,
                border_style=panel_style,
                box=CLI_CONFIG["display"]["panel_box"],
                padding=(1, 2),
            )

            console.print(response_panel)

            # Save to history
            state["history"].append(
                {"role": "assistant", "content": response, "is_tool_call": True}
            )
        else:
            # For plain text, try to detect and render markdown
            try:
                md = Markdown(response)
                # Print directly without animation, even if animation is enabled in config
                response_panel = Panel(
                    md,
                    title=title,
                    border_style=panel_style,
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2),
                )
                console.print(response_panel)
            except Exception:
                # Fall back to plain text if markdown rendering fails
                response_panel = Panel(
                    response,
                    title=title,
                    border_style=panel_style,
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2),
                )
                console.print(response_panel)

            # Save to history
            state["history"].append({"role": "assistant", "content": response})

    elif isinstance(response, dict) or isinstance(response, list):
        # Pretty print structured data
        json_str = json.dumps(response, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)

        response_panel = Panel(
            syntax,
            title=title,
            border_style=panel_style,
            box=CLI_CONFIG["display"]["panel_box"],
            padding=(1, 2),
        )

        console.print(response_panel)

        # Save to history
        state["history"].append({"role": "assistant", "content": response})

    else:
        # Handle any other type of response
        response_panel = Panel(
            str(response),
            title=title,
            border_style=panel_style,
            box=CLI_CONFIG["display"]["panel_box"],
            padding=(1, 2),
        )

        console.print(response_panel)

        # Save to history
        state["history"].append({"role": "assistant", "content": str(response)})


# ==================== Logging Setup ====================
def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    # Ensure log directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Set up main logging configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "glue.log"), mode="a")],
    )

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define a simpler format for console output
    if level <= logging.INFO:
        # Detailed format for verbose mode
        console_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(console_fmt)
        console_handler.setFormatter(formatter)
    else:
        # In normal mode, use a custom formatter that renders warnings in Rich style
        class RichWarningFormatter(logging.Formatter):
            def format(self, record):
                import io
                from rich.console import Console
                from rich.text import Text
                from rich.panel import Panel
                from rich.align import Align

                if record.levelno >= logging.WARNING:
                    # Get color based on level
                    if record.levelno >= logging.ERROR:
                        color = "red"
                        prefix = "✗ Error: "
                        style = "bold red"
                        title = (
                            "❌ Error"
                            if CLI_CONFIG["display"]["show_emoji"]
                            else "Error"
                        )
                    elif record.levelno >= logging.WARNING:
                        color = "yellow"
                        prefix = "⚠ Warning: "
                        style = "bold yellow"
                        title = (
                            "⚠️ Warning"
                            if CLI_CONFIG["display"]["show_emoji"]
                            else "Warning"
                        )
                    else:
                        color = "blue"
                        prefix = "ℹ Info: "
                        style = "blue"
                        title = (
                            "ℹ️ Info" if CLI_CONFIG["display"]["show_emoji"] else "Info"
                        )

                    # Format the message
                    message = record.getMessage()

                    # Create output buffer
                    string_io = io.StringIO()
                    temp_console = Console(file=string_io, highlight=False)

                    # Special highlighting for certain messages
                    if "does not support tool use on OpenRouter" in message:
                        # Create a panel for OpenRouter warnings (special case)
                        warning_panel = Panel(
                            f"[{style}]{message}[/{style}]",
                            title=(
                                "⚠️ OpenRouter Model Limitation"
                                if CLI_CONFIG["display"]["show_emoji"]
                                else "OpenRouter Model Limitation"
                            ),
                            border_style="yellow",
                            box=CLI_CONFIG["display"]["panel_box"],
                            width=100,
                        )
                        # Add spacing and center the panel
                        temp_console.print()  # Empty line before
                        temp_console.print(Align.center(warning_panel))
                        temp_console.print()  # Empty line after
                    elif record.levelno >= logging.WARNING:
                        # For other warnings/errors, create a panel with appropriate styling
                        panel = Panel(
                            f"[{style}]{message}[/{style}]",
                            title=title,
                            border_style=color,
                            box=CLI_CONFIG["display"]["panel_box"],
                            width=100,
                        )
                        # Add spacing and center the panel
                        temp_console.print()  # Empty line before
                        temp_console.print(Align.center(panel))
                        temp_console.print()  # Empty line after
                    else:
                        # For info messages, use a simpler format
                        text = Text()
                        text.append(prefix, style=f"bold {color}")
                        text.append(message, style="dim")
                        temp_console.print(text)

                    return string_io.getvalue().strip()

                # For non-warnings, use simple text
                return f"[{record.levelname}] {record.getMessage()}"

        console_handler.setFormatter(RichWarningFormatter())

    # Create deduplication filter to prevent duplicate warnings
    class DuplicateFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self._logged = set()

        def filter(self, record):
            # Create a unique key from the message and name
            key = (record.getMessage(), record.name)
            # Only log new messages, suppress duplicates
            if key in self._logged:
                return False
            self._logged.add(key)
            return True

    # Add filter to console handler
    console_handler.addFilter(DuplicateFilter())

    # Configure GLUE logger
    glue_logger = logging.getLogger("glue")
    glue_logger.setLevel(level)
    glue_logger.addHandler(console_handler)
    glue_logger.propagate = False  # Prevent double logging

    # Configure third-party loggers - always set to WARNING unless in DEBUG mode
    third_party_level = logging.DEBUG if level == logging.DEBUG else logging.WARNING
    for logger_name in ["httpcore", "httpx", "openai", "markdown_it", "rich"]:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(third_party_level)
        # Only log to file, not console
        for handler in third_party_logger.handlers:
            third_party_logger.removeHandler(handler)
        third_party_logger.addHandler(
            logging.FileHandler(os.path.join(LOGS_DIR, "glue.log"), mode="a")
        )
        third_party_logger.propagate = False  # Prevent double logging

    return glue_logger


# ==================== Project Management Functions ====================


def create_new_project(
    project_name: Optional[str] = None,
    template: Optional[str] = None,
    force: bool = False,
) -> None:
    """Create a new GLUE project.

    Args:
        project_name: Name of the project to create
        template: Template to use (ignored - interactive mode is used)
        force: Whether to overwrite an existing project
    """
    from rich.panel import Panel

    console = get_console()

    # Display logo and welcome header
    console.clear()
    display_logo(console, show_version=True)
    console.print()

    welcome_panel = Panel(
        "[bold]Welcome to the GLUE Project Builder![/bold]\n\n"
        "This wizard will guide you through creating a new GLUE project with a customized "
        "configuration for your multi-agent application.\n\n"
        "[dim]You'll define models, tools, teams, and how they work together.[/dim]",
        title="🚀 Create New GLUE Project",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(welcome_panel)
    console.print()

    # If no project name provided, prompt for one
    if not project_name:
        project_name = prompt_with_help(
            "Enter project name", default="my_glue_app", icon="📁", console=console
        )

        # Sanitize project name
        project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name)

    # Check if project directory already exists
    project_dir = os.path.join(os.getcwd(), project_name)
    glue_file = os.path.join(project_dir, "app.glue")

    if os.path.exists(project_dir) and not force:
        error_panel = Panel(
            f"[bold red]Project '{project_name}' already exists.[/bold red]\n\n"
            "To create a new project with this name, use the --force flag:\n"
            f"[dim]glue new {project_name} --force[/dim]",
            title="❌ Error",
            border_style="red",
            padding=(1, 2),
        )
        console.print(error_panel)
        return

    # Create project directory structure
    try:
        # Create directories
        os.makedirs(project_dir, exist_ok=True)
        workspace_dir = os.path.join(project_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        console.print("[cyan]Building GLUE configuration...[/cyan]")

        # Generate GLUE file content
        try:
            # Generate content with the interactive builder
            content = get_template_content(template, project_name)

            # Debug: Check content before writing
            console.print("[dim]DEBUG: Writing GLUE file content...[/dim]")
            for line in content.splitlines():
                if "members =" in line or "tools =" in line:
                    console.print(f"[dim]WRITE DEBUG: {line}[/dim]")

            # Write the GLUE file
            with open(glue_file, "w") as f:
                f.write(content)

            # Create README
            readme_path = os.path.join(project_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"# {project_name}\n\n")
                f.write("A GLUE project for multi-agent applications.\n\n")
                f.write("## Getting Started\n\n")
                f.write("To start this GLUE application:\n\n")
                f.write(f"```\ncd {project_name}\nglue run app.glue\n```\n")

            # Create other necessary files like .env
            env_path = os.path.join(project_dir, ".env")
            with open(env_path, "w") as f:
                f.write("# API Keys for providers\n")
                f.write("# OPENAI_API_KEY=your_openai_key\n")
                f.write("# ANTHROPIC_API_KEY=your_anthropic_key\n")
                f.write("# OPENROUTER_API_KEY=your_openrouter_key\n")

            # Success message
            success_panel = Panel(
                f"[bold green]Project '{project_name}' created successfully![/bold green]\n\n"
                f"Your new GLUE project is ready at: [bold]{project_dir}[/bold]\n\n"
                f"To start your application:\n"
                f"[dim]cd {project_name}[/dim]\n"
                f"[dim]glue run app.glue[/dim]",
                title="✅ Success",
                border_style="green",
                padding=(1, 2),
            )
            console.print()
            console.print(success_panel)

        except Exception as e:
            console.print(
                f"[bold red]Failed to create project configuration: {str(e)}[/bold red]"
            )
            return

    except Exception as e:
        console.print(
            f"[bold red]Failed to create project directory: {str(e)}[/bold red]"
        )
        return


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
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        logger.error(
            f"Invalid tool name: {name}. Must start with a letter and contain only letters, numbers, and underscores."
        )
        print(
            "Error: Invalid tool name. Must start with a letter and contain only letters, numbers, and underscores."
        )
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
    print("To use this tool, add the following to your app.glue file:")
    print(
        f"""
tool {name} {{
    custom = true
}}
"""
    )
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
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        logger.error(
            f"Invalid MCP name: {name}. Must start with a letter and contain only letters, numbers, and underscores."
        )
        print(
            "Error: Invalid MCP name. Must start with a letter and contain only letters, numbers, and underscores."
        )
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
    print("To use this MCP, add the following to your app.glue file:")
    print(
        f"""
mcp {name} {{
    custom = true
}}
"""
    )
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
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        logger.error(
            f"Invalid API name: {name}. Must start with a letter and contain only letters, numbers, and underscores."
        )
        print(
            "Error: Invalid API name. Must start with a letter and contain only letters, numbers, and underscores."
        )
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
    print("To use this API in your application:")
    print(
        f"""
from apis.{name}_api import get_api_client

# In an async function:
api_client = get_api_client()
await api_client.setup()
result = await api_client.get("endpoint", {{"param": "value"}})
"""
    )
    return True


def run_forge_command():
    """Run the GLUE Forge interactive CLI to create custom components using Google Gemini.

    This function implements an interactive CLI that guides users through setting up
    their API key for Google Gemini and using it to create custom components.
    """
    import os
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
            use_existing = (
                prompt_with_help(
                    "Would you like to use this configuration?",
                    default="Y",
                    choices=["Y", "n"],
                    icon="🔑",
                )
                .strip()
                .lower()
            )

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
            choice = prompt_with_help(
                "Enter your choice",
                choices=["1", "2"],
                icon="🔑",
            ).strip()

        if choice == "1":
            api_source = "Google AI Studio"
            model_name = "gemini-1.5-pro"
            print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        else:
            api_source = "OpenRouter"
            model_name = "google/gemini-2.5-pro-exp-03-25:free"
            print("\nGet your API key from: https://openrouter.ai/keys")

        api_key = getpass.getpass(f"\nEnter your {api_source} API key: ").strip()

        # Validate API key format
        if not api_key or len(api_key) < 10:
            print("Invalid API key. Please provide a valid API key.")
            return False

        # Save configuration
        config = {
            "api_key": api_key,
            "api_source": api_source,
            "model_name": model_name,
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
        forge_choice = prompt_with_help(
            "Enter your choice",
            choices=["1", "2", "3"],
            icon="🔨",
        ).strip()

    if forge_choice == "1":
        name = prompt_with_help(
            "Enter a name for your tool (letters, numbers, underscores only)",
            icon="🔧",
        ).strip()
        description = prompt_with_help(
            "Enter a brief description of what your tool does",
            icon="📝",
        ).strip()

        print("\nChoose a template:")
        print("1. Basic Tool (simple function)")
        print("2. API Tool (makes external API calls)")
        print("3. Data Tool (processes data files)")

        template_choice = None
        while template_choice not in ["1", "2", "3"]:
            template_choice = prompt_with_help(
                "Enter your choice",
                choices=["1", "2", "3"],
                icon="📁",
            ).strip()

        template = (
            "basic"
            if template_choice == "1"
            else "api"
            if template_choice == "2"
            else "data"
        )

        print(f"\nCreating {template} tool: {name}")
        forge_tool(name, description, template)

    elif forge_choice == "2":
        name = prompt_with_help(
            "Enter a name for your MCP integration (letters, numbers, underscores only)",
            icon="🔌",
        ).strip()
        description = prompt_with_help(
            "Enter a brief description of what your MCP integration does",
            icon="📝",
        ).strip()

        print(f"\nCreating MCP integration: {name}")
        forge_mcp(name, description)

    elif forge_choice == "3":
        name = prompt_with_help(
            "Enter a name for your API integration (letters, numbers, underscores only)",
            icon="🔗",
        ).strip()
        description = prompt_with_help(
            "Enter a brief description of what your API integration does",
            icon="📝",
        ).strip()

        print(f"\nCreating API integration: {name}")
        forge_api(name, description)

    print("\nThank you for using GLUE Forge!")
    return True


# ==================== Utility Functions ====================
def list_models():
    """List available models."""
    # Import provider modules dynamically to get available models
    from glue.core.providers import get_available_providers, get_provider_models

    print("Available models:")

    try:
        # Get all available providers
        providers = get_available_providers()

        # For each provider, get and display their available models
        for provider_name in providers:
            models = get_provider_models(provider_name)
            if models:
                print(f"\n{provider_name.capitalize()} Models:")
                for model in models:
                    print(f"  - {model}")
    except Exception as e:
        print(f"Error retrieving models: {str(e)}")
        # Fallback to basic info if dynamic retrieval fails
        FALLBACK_MODELS = {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "google": ["gemini-pro", "gemini-ultra"],
        }
        print("\nFallback model information:")
        for provider, models in FALLBACK_MODELS.items():
            print(f"\n{provider.capitalize()} Models:")
            for model in models:
                print(f"  - {model}")


def display_tools() -> int:
    """Display available tools.

    Returns:
        Exit code.
    """
    tools = get_available_tools()

    if not tools:
        print("No tools are currently available.")
        return 1

    print("Available tools:")
    for tool_name, tool_info in tools.items():
        description = tool_info.get("description", "No description available")

        print(f"\n{tool_name}:")
        print(f"  Description: {description}")

    return 0


def validate_glue_file(config_file: str, strict: bool = False) -> None:
    """Validate a GLUE file for syntax and semantic correctness.

    Args:
        config_file: Path to the GLUE configuration file
        strict: Whether to enable strict validation
    """
    logger = logging.getLogger("glue.validate")
    console = Console()

    try:
        # Check if file exists
        if not os.path.exists(config_file):
            console.print(
                f"[{CLI_CONFIG['theme']['error']}]Error: File not found: {config_file}[/{CLI_CONFIG['theme']['error']}]"
            )
            sys.exit(1)

        # Read the file
        with open(config_file, "r") as f:
            content = f.read()

        # Display file info
        file_stats = os.stat(config_file)
        file_size = file_stats.st_size
        file_modified = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        display_section_header(console, "File Information")
        console.print(f"File: [cyan]{config_file}[/cyan]")
        console.print(f"Size: {file_size:,} bytes")
        console.print(f"Last Modified: {file_modified}")
        console.print(f"Lines: {len(content.splitlines())}")
        console.print("")

        # Parse the file with lexer
        lexer = GlueLexer()
        parser = GlueParser()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Lexical analysis
            task1 = progress.add_task("[cyan]Performing lexical analysis...", total=1)
            try:
                tokens = lexer.tokenize(content)
                # Convert tokens to a list so they can be iterated multiple times
                tokens_list = list(tokens)
                progress.update(
                    task1, advance=1, description="[green]Lexical analysis completed"
                )
                syntax_errors = []
            except Exception as e:
                progress.update(
                    task1, advance=1, description="[red]Lexical analysis failed"
                )
                console.print(
                    f"[{CLI_CONFIG['theme']['error']}]Lexical error: {e}[/{CLI_CONFIG['theme']['error']}]"
                )
                sys.exit(1)

            # Syntax analysis
            task2 = progress.add_task("[cyan]Performing syntax analysis...", total=1)
            try:
                ast = parser.parse(iter(tokens_list))
                progress.update(
                    task2, advance=1, description="[green]Syntax analysis completed"
                )
            except Exception as e:
                progress.update(
                    task2, advance=1, description="[red]Syntax analysis failed"
                )
                console.print(
                    f"[{CLI_CONFIG['theme']['error']}]Syntax error: {e}[/{CLI_CONFIG['theme']['error']}]"
                )
                sys.exit(1)

            # Semantic analysis
            task3 = progress.add_task(
                "[cyan]Performing semantic validation...", total=1
            )
            semantic_errors = []
            warnings = []

            # Check for app section
            if "app" not in ast:
                semantic_errors.append("Missing 'app' section")
            else:
                app_config = ast["app"]
                # Check for app name
                if "name" not in app_config:
                    semantic_errors.append("Missing app name in 'app' section")

            # Check for models
            if "models" not in ast:
                warnings.append("No models defined")
            else:
                models = ast["models"]
                for model_name, model_config in models.items():
                    # Check for required model fields
                    if "provider" not in model_config:
                        semantic_errors.append(
                            f"Model '{model_name}' is missing required 'provider' field"
                        )

            # Check for teams (magnetize section)
            if "magnetize" not in ast:
                warnings.append("No teams defined (missing 'magnetize' section)")
            else:
                teams = ast["magnetize"]
                for team_name, team_config in teams.items():
                    # Check for lead model
                    if "lead" not in team_config:
                        semantic_errors.append(
                            f"Team '{team_name}' is missing required 'lead' field"
                        )

            # Check for adhesives
            if "apply" not in ast:
                warnings.append("No adhesives applied (missing 'apply' section)")

            # Additional strict checks if enabled
            if strict:
                # Check model configs
                if "models" in ast:
                    for model_name, model_config in ast["models"].items():
                        if "config" in model_config:
                            config = model_config["config"]
                            # Check if model specified for OpenAI
                            if (
                                model_config.get("provider") == "openai"
                                and "model" not in config
                            ):
                                semantic_errors.append(
                                    f"OpenAI model '{model_name}' is missing 'model' in config"
                                )
                        else:
                            warnings.append(
                                f"Model '{model_name}' has no configuration section"
                            )

                # Check tool configs
                if "tools" in ast:
                    for tool_name, tool_config in ast["tools"].items():
                        if tool_name not in AVAILABLE_TOOLS:
                            warnings.append(f"Unknown tool: '{tool_name}'")

            if semantic_errors:
                progress.update(
                    task3, advance=1, description="[red]Semantic validation failed"
                )
            else:
                progress.update(
                    task3, advance=1, description="[green]Semantic validation completed"
                )

        # Display validation results
        display_section_header(console, "Validation Results")

        if not syntax_errors and not semantic_errors:
            console.print(
                f"[{CLI_CONFIG['theme']['success']}]✓ File is valid![/{CLI_CONFIG['theme']['success']}]"
            )

        if syntax_errors:
            console.print(
                f"[{CLI_CONFIG['theme']['error']}]Syntax Errors:[/{CLI_CONFIG['theme']['error']}]"
            )
            for i, error in enumerate(syntax_errors, 1):
                console.print(f"  {i}. {error}")

        if semantic_errors:
            console.print(
                f"[{CLI_CONFIG['theme']['error']}]Semantic Errors:[/{CLI_CONFIG['theme']['error']}]"
            )
            for i, error in enumerate(semantic_errors, 1):
                console.print(f"  {i}. {error}")

        if warnings:
            console.print(
                f"[{CLI_CONFIG['theme']['warning']}]Warnings:[/{CLI_CONFIG['theme']['warning']}]"
            )
            for i, warning in enumerate(warnings, 1):
                console.print(f"  {i}. {warning}")

        # Display AST if no errors and strict mode
        if strict and not syntax_errors and not semantic_errors:
            display_section_header(console, "AST Structure")
            json_str = json.dumps(ast, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            console.print(syntax)

        # Exit with appropriate code
        if syntax_errors or semantic_errors:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error validating file: {e}", exc_info=True)
        console.print_exception()
        console.print(
            f"[{CLI_CONFIG['theme']['error']}]Error validating file: {e}[/{CLI_CONFIG['theme']['error']}]"
        )
        sys.exit(1)


# ==================== Display Helpers ====================
def get_console(theme: bool = True) -> Console:
    """Get a rich console with the GLUE theme.

    Args:
        theme: Whether to apply the GLUE theme

    Returns:
        A configured Console object
    """
    if theme:
        return Console(theme=GLUE_THEME, highlight=True)
    return Console(highlight=True)


def display_logo(console: Console, show_version: bool = True) -> None:
    """Display the GLUE framework logo.

    Args:
        console: Rich console to use for display
        show_version: Whether to include version information
    """
    logo = GLUE_LOGO.format(__version__=__version__ if show_version else "")
    align = Align.center(logo)
    console.print(align)
    console.print(Rule(style="dim cyan"))


def create_app_layout() -> Layout:
    """Create a rich layout for the application display.

    Returns:
        A rich Layout object
    """
    layout = Layout()

    # Split into header, main, and footer
    layout.split(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )

    # Split main area into sidebar and content
    layout["main"].split_row(Layout(name="sidebar", size=30), Layout(name="content"))

    # Add sub-layouts in content area
    layout["content"].split(
        Layout(name="output", ratio=3), Layout(name="input", size=5)
    )

    return layout


def create_tool_table(tools: Dict[str, Any]) -> Table:
    """Create a table displaying the available tools.

    Args:
        tools: Dictionary of tools with name as key

    Returns:
        A rich Table object
    """
    table = Table(
        title="Available Tools",
        show_header=True,
        header_style="bold yellow",
        box=CLI_CONFIG["display"]["table_box"],
        title_style="tool",
    )

    # Add emoji if enabled
    tool_emoji = (
        CLI_CONFIG["emoji"]["tool"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    )

    table.add_column(f"{tool_emoji}Tool", style="tool")
    table.add_column("Description")

    for name, tool in tools.items():
        # Extract tool information
        if isinstance(tool, dict):
            description = tool.get("description", "No description available")
        else:
            description = getattr(tool, "description", "No description available")

        table.add_row(name, description)

    return table


def create_team_table(teams: Dict[str, Any]) -> Table:
    """Create a table displaying the team structure.

    Args:
        teams: Dictionary of teams

    Returns:
        A rich Table object
    """
    table = Table(
        title="Team Structure",
        show_header=True,
        header_style="bold blue",
        box=CLI_CONFIG["display"]["table_box"],
        title_style="team",
    )

    # Add emoji if enabled
    team_emoji = (
        CLI_CONFIG["emoji"]["team"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    )
    model_emoji = (
        CLI_CONFIG["emoji"]["model"] + " "
        if CLI_CONFIG["display"]["show_emoji"]
        else ""
    )
    tool_emoji = (
        CLI_CONFIG["emoji"]["tool"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    )

    table.add_column(f"{team_emoji}Team", style="team")
    table.add_column(f"{model_emoji}Lead", style="green")
    table.add_column("Members", style="cyan")
    table.add_column(f"{tool_emoji}Tools", style="tool")

    for team_name, team in teams.items():
        lead_name = team.lead.name if hasattr(team, "lead") and team.lead else "None"

        # Get members from team.models instead of team.members
        members_list = []
        if hasattr(team, "models") and team.models:
            for model_name, model in team.models.items():
                # Skip the lead model which is displayed separately
                if not (hasattr(team, "lead") and team.lead and model == team.lead):
                    members_list.append(model_name)
        members = ", ".join(members_list) if members_list else "None"

        tools = ", ".join([t for t in team.tools]) if hasattr(team, "tools") else "None"

        table.add_row(team_name, lead_name, members, tools)

    return table


def create_model_tree(teams: Dict[str, Any]) -> Tree:
    """Create a tree view of models organized by teams.

    Args:
        teams: Dictionary of teams

    Returns:
        A rich Tree object
    """
    # Add emoji if enabled
    team_emoji = (
        CLI_CONFIG["emoji"]["team"] if CLI_CONFIG["display"]["show_emoji"] else "●"
    )
    model_emoji = (
        CLI_CONFIG["emoji"]["model"] if CLI_CONFIG["display"]["show_emoji"] else "○"
    )

    # Create root tree
    tree = Tree("[bold]Model Structure[/bold]")

    # Add teams and models
    for team_name, team in teams.items():
        team_node = tree.add(f"[team]{team_emoji} {team_name}[/team]")

        # Add lead model - always use green for leads
        if hasattr(team, "lead") and team.lead:
            team_node.add(f"[green]{model_emoji} {team.lead.name} (Lead)[/green]")

        # Add member models - always use cyan for members
        if hasattr(team, "models") and team.models:
            for model_name, model in team.models.items():
                # Skip the lead model which was already displayed
                if hasattr(team, "lead") and team.lead and model == team.lead:
                    continue

                team_node.add(f"[cyan]{model_emoji} {model_name}[/cyan]")

    return tree


def display_model_info(console: Console, model: Dict[str, Any]) -> None:
    """Display detailed information about a model.

    Args:
        console: Rich console to use for display
        model: Model information dictionary
    """
    # Add emoji if enabled
    model_emoji = (
        CLI_CONFIG["emoji"]["model"] + " "
        if CLI_CONFIG["display"]["show_emoji"]
        else ""
    )

    # Determine model style
    model_name = model.get("name", "Unknown")
    style = CLI_CONFIG["theme"]["model"].get(
        model_name, CLI_CONFIG["theme"]["model"]["default"]
    )

    # Create model info
    model_info = Group(
        Text(f"Provider: {model.get('provider', 'Unknown')}", style="muted"),
        Text(f"Base Model: {model.get('base_model', 'Unknown')}", style="app.value"),
        Text(f"Temperature: {model.get('temperature', 0.7)}", style="muted"),
        Rule(style="dim"),
        Markdown(f"**Role:** {model.get('role', 'No role defined')}"),
        Text(f"Adhesives: {', '.join(model.get('adhesives', []))}", style="code"),
    )

    panel = Panel(
        model_info,
        title=f"{model_emoji}{model_name}",
        title_align="left",
        border_style=style,
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2),
    )

    console.print(panel)


def show_interactive_help(console: Console) -> None:
    """Display help information for interactive mode.

    Args:
        console: Rich console to use for display
    """
    help_text = get_interactive_help_text()

    # Create the help content with sections
    commands_section = Table.grid(padding=(0, 2))
    commands_section.add_column(style="command")
    commands_section.add_column()

    # Add command explanations from help text
    for line in help_text.strip().split("\n"):
        if line.startswith("/"):
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                cmd, desc = parts
                commands_section.add_row(cmd.strip(), desc.strip())

    # Create a panel with the help information
    panel = Panel(
        Group(
            Text(
                "Use these commands to interact with the GLUE application:",
                style="info",
            ),
            commands_section,
            Text(
                "\nTip: You can type a message directly to interact with the active model or team.",
                style="muted",
            ),
        ),
        title="Interactive Mode Help",
        title_align="center",
        border_style="blue",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2),
    )

    console.print(panel)


def create_status_panel(app: Any, state: Dict[str, Any]) -> Panel:
    """Create a panel showing the current application status.

    Args:
        app: The GLUE application
        state: Current state dictionary

    Returns:
        A rich Panel object
    """
    # Build status information
    status_table = Table.grid(padding=(0, 1))
    status_table.add_column(style="muted", justify="right")
    status_table.add_column(style="app.value")

    # Add app details
    status_table.add_row(
        "App Name:", Text(app.name, style="app.name" if app.name else "Undefined")
    )
    status_table.add_row("Teams:", str(len(app.teams)))

    # Calculate total models correctly (leads + members)
    total_models = 0
    for team in app.teams.values():
        # Count lead if it exists
        if hasattr(team, "lead") and team.lead is not None:
            total_models += 1
        # Count members if they exist - fix the bug by checking team.models instead of team.members
        if hasattr(team, "models") and team.models is not None:
            # Count all models except the lead (which was already counted)
            for model_name, model in team.models.items():
                if not (hasattr(team, "lead") and team.lead and model == team.lead):
                    total_models += 1

    status_table.add_row("Models:", str(total_models))
    status_table.add_row("Tools:", str(len(app.tools) - 1))

    # Add mode settings
    status_table.add_row(
        "Verbose Mode:", "✅ Enabled" if state.get("verbose_mode") else "❌ Disabled"
    )
    status_table.add_row(
        "Step Mode:", "✅ Enabled" if state.get("step_mode") else "❌ Disabled"
    )
    status_table.add_row(
        "Color Output:", "✅ Enabled" if state.get("color_enabled") else "❌ Disabled"
    )

    # Add context info
    if state.get("current_team"):
        status_table.add_row("Active Team:", Text(state["current_team"], style="team"))
    elif state.get("current_model"):
        style = CLI_CONFIG["theme"]["model"].get(
            state["current_model"], CLI_CONFIG["theme"]["model"]["default"]
        )
        status_table.add_row("Active Model:", Text(state["current_model"], style=style))

    # Create panel
    panel = Panel(
        status_table,
        title="Application Status",
        title_align="center",
        border_style="blue",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2),
    )

    return panel


def animate_typing(console: Console, text: str, speed: float = 0.03) -> None:
    """Animate typing text to the console character by character.

    Args:
        console: Rich console to use for display
        text: Text to animate
        speed: Delay between characters in seconds
    """
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(speed)
    console.print()  # Print a newline at the end


# ==================== Main CLI Entry Point ====================
import click

@click.group()
@click.version_option(version=__version__)
def cli():
    """GLUE Framework CLI"""
    pass

@cli.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "-i", help="Input text for the app")
@click.option("--interactive", "-I", is_flag=True, help="Run in interactive mode")
@click.option("--verbose", "-v", count=True, help="Enable verbose logging")
@click.option("--env", "-e", help="Path to .env file")
@click.option("--engine", type=click.Choice(["glue", "agno"]), default="agno", help="Specify the execution engine")
def run_command(config, input, interactive, verbose, env, engine):
    """Run a GLUE application"""
    # Load environment variables if specified
    if env:
        from dotenv import load_dotenv
        load_dotenv(env)
    elif os.path.exists(DEFAULT_ENV_FILE):
        # Load default .env file if it exists
        from dotenv import load_dotenv
        load_dotenv(DEFAULT_ENV_FILE)

    # Set up logging
    if verbose > 1:
        log_level = logging.DEBUG
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logger = setup_logging(log_level)

    # Run the application
    asyncio.run(run_app(config, interactive, input, engine))

@cli.command("new")
@click.argument("project", required=False)
@click.option("--template", "-t", default="interactive", help="Ignored - interactive mode is always used")
def new_command(project, template):
    """Create a new GLUE project"""
    create_new_project(project, template)

@cli.group("forge", invoke_without_command=True)
@click.pass_context
def forge_group(ctx):
    """Create custom components with AI assistance"""
    if ctx.invoked_without_command:
        # Run interactive forge command if no subcommand is specified
        run_forge_command()

@forge_group.command("tool")
@click.argument("name")
@click.option("--description", "-d", required=True, help="Tool description")
@click.option("--template", "-t", type=click.Choice(["basic", "api", "data"]), default="basic", help="Tool template to use")
def forge_tool_command(name, description, template):
    """Create a custom tool"""
    forge_tool(name, description, template)

@forge_group.command("mcp")
@click.argument("name")
@click.option("--description", "-d", required=True, help="MCP description")
def forge_mcp_command(name, description):
    """Create a custom MCP integration"""
    forge_mcp(name, description)

@forge_group.command("api")
@click.argument("name")
@click.option("--description", "-d", required=True, help="API description")
def forge_api_command(name, description):
    """Create a custom API integration"""
    forge_api(name, description)

@cli.command("list-tools")
def list_tools_command():
    """List available tools"""
    display_tools()

@cli.command("list-models")
def list_models_command():
    """List available models"""
    list_models()

@cli.command("validate")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def validate_command(file):
    """Validate a GLUE file"""
    validate_glue_file(file)

if __name__ == "__main__":
    cli()
