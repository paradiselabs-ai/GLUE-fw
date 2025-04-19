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
import uuid
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Rich UI components
from rich import print
from rich.console import Console, Group
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.live import Live
from rich.traceback import install as install_rich_traceback
from rich.align import Align
from rich.box import ROUNDED, DOUBLE, HEAVY, Box
from rich.columns import Columns
from rich.highlighter import ReprHighlighter
from rich.style import Style
from rich.text import Text
from rich.tree import Tree
from rich.rule import Rule
from rich.theme import Theme
from rich.emoji import Emoji
from rich.segment import Segment
from rich.status import Status

# Import framework modules
# Use direct imports to avoid circular imports
from glue.core import GlueApp
from glue.dsl import GlueDSLParser, GlueLexer

# Import new utilities
from .utils.json_utils import extract_json
from .cliHelpers import parse_interactive_command, colorize_agent_output, format_agent_message, get_interactive_help_text
from .utils.ui_utils import display_warning, set_cli_config

# Constants for tools
def get_available_tools():
    """retrieve available tools from the tool registry."""
        
    # Fallback if dynamic retrieval fails
    return {
    "web_search": {
        "description": "Search the web for information",
        "parameters": {"query": "string"}
    },
    "file_handler": {
        "description": "Read and write files",
        "parameters": {"path": "string", "content": "string (optional)"}
    },
    "code_interpreter": {
        "description": "Execute Python code",
        "parameters": {"code": "string"}
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
GLUE_THEME = Theme({
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
})

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
            "pulse": "pulse"
        }
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
            "show_typing_animation": True
        }
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
    }
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
    "interactive": ["help", "status", "tools", "teams", "clear", "verbose", "step", "color", "exit"]
}

# Template types with descriptions
TEMPLATES = {
    "basic": "Simple app with a single model",
    "research": "Research-focused app with web search tools",
    "chat": "Chat application with multiple collaborating models",
    "agent": "Full agent system with autonomy and complex workflows"
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

def display_section_header(console: Console, title: str, emoji: Optional[str] = None) -> None:
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
    console = get_console()  # Use get_console() instead of Console() to ensure theme is applied
    
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
        logger.debug("Setting up application")
        await app.setup()
        logger.debug("Application setup complete")
        
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
            logger.info(f"Running non-interactive agentic workflow with initial input: {input_text}")
            current_input = input_text
            max_turns = 10  # Set a limit to prevent infinite loops
            turn_count = 0
            final_response = None

            # Determine the lead model (assuming the first team's lead is the entry point)
            if not app.teams:
                logger.error("No teams defined in the application.")
                console.print(Panel(
                    "[bold red]No teams defined in the application.[/bold red]",
                    title="Error",
                    border_style="error",
                    box=CLI_CONFIG["display"]["panel_box"]
                ))
                return False
            first_team_name = next(iter(app.teams))
            if not hasattr(app.teams[first_team_name], 'lead') or not app.teams[first_team_name].lead:
                logger.error(f"Lead model not found for the first team '{first_team_name}'. Check team config.")
                console.print(Panel(
                    f"[bold red]Lead model not found for team '{first_team_name}'.[/bold red]",
                    title="Error",
                    border_style="error",
                    box=CLI_CONFIG["display"]["panel_box"]
                ))
                return False
            lead_model = app.teams[first_team_name].lead
            logger.info(f"Using lead model '{lead_model.name}' from team '{first_team_name}' as entry point.")

            # Create a state dictionary similar to interactive mode for display helpers
            state = {
                "history": [],
                "current_team": first_team_name,
                "current_model": lead_model.name,
                "color_enabled": True
            }

            # Display welcome banner
            display_logo(console)
            
            welcome_panel = Panel(
                Group(
                    Align.center(f"Running non-interactive mode for:", style="cyan"),
                    Align.center(app.name, style="bold cyan") if app.name else None,
                    Rule(style="dim"),
                    Align.center(Text(f"Using {lead_model.name} from team {first_team_name}", style="cyan"))
                ),
                title="GLUE App",
                border_style="success",
                box=CLI_CONFIG["display"]["panel_box"],
                padding=(1, 2)
            )
            console.print(welcome_panel)

            # Show initial input
            if CLI_CONFIG["display"]["show_emoji"]:
                display_section_header(console, "Initial Input", emoji="info")
            else:
                display_section_header(console, "Initial Input")
            
            input_panel = Panel(
                Markdown(current_input),
                title="User Input",
                border_style="bright_blue",
                box=CLI_CONFIG["display"]["panel_box"],
                padding=(1, 2)
            )
            console.print(input_panel)

            messages = [{"role": "user", "content": current_input}]

            if CLI_CONFIG["display"]["show_emoji"]:
                display_section_header(console, "Agent Interactions", emoji="model")
            else:
                display_section_header(console, "Agent Interactions")

            while turn_count < max_turns:
                turn_count += 1
                
                # Display turn header
                console.print(Rule(f"Turn {turn_count}", style="dim"))

                # --- Generate response ---
                response_content = None
                try:
                    logger.debug(f"Calling model {lead_model.name} with {len(messages)} messages.")
                    response_content = await lead_model.generate_response(messages=messages, tools=app.tools)
                    
                    # Use display_response function for consistent styling with interactive mode
                    display_response(console, response_content, lead_model.name, state)

                    # --- Append Assistant Response to History ---
                    if response_content is not None:
                        messages.append({"role": "assistant", "content": str(response_content)})

                except Exception as e:
                    logger.error(f"Error during model generation: {e}", exc_info=True)
                    console.print(Panel(
                        f"[bold red]Error during model generation: {e}[/bold red]",
                        title="Error",
                        border_style="error",
                        box=CLI_CONFIG["display"]["panel_box"]
                    ))
                    break

                # --- Process potential tool calls in the response ---
                tool_calls_found_in_response = False
                tool_results_for_next_turn = []

                if isinstance(response_content, str):
                    # Use extract_json to detect tool calls
                    potential_tool_call = extract_json(response_content)
                    if potential_tool_call and isinstance(potential_tool_call, dict) and "tool_name" in potential_tool_call and "arguments" in potential_tool_call:
                        tool_name = potential_tool_call["tool_name"]
                        arguments = potential_tool_call["arguments"]
                        tool_call_id = f"call_{uuid.uuid4()}"
                        logger.info(f"Extracted tool call: {tool_name} (ID: {tool_call_id}) with args {arguments}")
                        tool_calls_found_in_response = True

                        # Display tool call notification
                        tool_emoji = CLI_CONFIG["emoji"]["tool"] if CLI_CONFIG["display"]["show_emoji"] else ""
                        tool_title = f"{tool_emoji} Tool Call: {tool_name}" if tool_emoji else f"Tool Call: {tool_name}"
                        
                        json_str = json.dumps(arguments, indent=2)
                        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                        
                        console.print(Panel(
                            syntax,
                            title=tool_title,
                            border_style="tool",
                            box=CLI_CONFIG["display"]["panel_box"],
                            padding=(1, 2)
                        ))

                        # --- Execute Tool with Error Handling ---
                        tool_result_message = None
                        try:
                            if tool_name in app.tools:
                                tool = app.tools[tool_name]
                                if hasattr(tool, 'execute') and callable(tool.execute):
                                    # Display executing notification
                                    with console.status(f"[yellow]Executing tool {tool_name}...[/yellow]"):
                                        tool_result = await app.execute_tool(tool_name, arguments)
                                    
                                    
                                    content_for_llm = str(tool_result)
                                    if isinstance(tool_result, dict) and 'success' in tool_result and 'response' in tool_result:
                                        if tool_result['success']:
                                            content_for_llm = tool_result['response']
                                        else:
                                            content_for_llm = f"Tool execution failed: {tool_result.get('response', 'Unknown error')}"
                                    elif tool_result is None:
                                        content_for_llm = "Tool executed successfully with no return value."

                                    # Display tool result
                                    if isinstance(tool_result, (dict, list)):
                                        json_str = json.dumps(tool_result, indent=2)
                                        result_content = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                                    else:
                                        result_content = str(tool_result)
                                    
                                    console.print(Panel(
                                        result_content,
                                        title=f"Tool Result: {tool_name}",
                                        border_style="success",
                                        box=CLI_CONFIG["display"]["panel_box"],
                                        padding=(1, 2)
                                    ))

                                    tool_result_message = {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": content_for_llm
                                    }
                                else:
                                    logger.error(f"Tool {tool_name} exists but has no execute method.")
                                    error_content = f"Tool execution failed: '{tool_name}' is not properly initialized."
                                    
                                    console.print(Panel(
                                        f"[bold red]{error_content}[/bold red]",
                                        title="Tool Error",
                                        border_style="error",
                                        box=CLI_CONFIG["display"]["panel_box"]
                                    ))
                                    
                                    tool_result_message = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": error_content, "is_error": True}
                            else:
                                logger.error(f"Tool {tool_name} not found in app.tools.")
                                error_content = f"Tool execution failed: '{tool_name}' not found."
                                
                                console.print(Panel(
                                    f"[bold red]{error_content}[/bold red]",
                                    title="Tool Error",
                                    border_style="error",
                                    box=CLI_CONFIG["display"]["panel_box"]
                                ))
                                
                                tool_result_message = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": error_content, "is_error": True}
                        except Exception as e:
                            logger.error(f"Error executing tool '{tool_name}' (ID: {tool_call_id}): {e}", exc_info=True)
                            error_content = f"Error executing tool '{tool_name}': {e}"
                            
                            console.print(Panel(
                                f"[bold red]{error_content}[/bold red]",
                                title="Tool Error",
                                border_style="error",
                                box=CLI_CONFIG["display"]["panel_box"]
                            ))
                            
                            tool_result_message = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": error_content, "is_error": True}

                        if tool_result_message:
                            tool_results_for_next_turn.append(tool_result_message)
                    else:
                        logger.debug("No valid tool call JSON found in response.")

                # --- Append Tool Results and Decide Next Step ---
                if tool_calls_found_in_response:
                    if tool_results_for_next_turn:
                        logger.info(f"Adding {len(tool_results_for_next_turn)} tool results to conversation history.")
                        messages.extend(tool_results_for_next_turn)
                        continue
                    else:
                        logger.warning("Tool call detected but no results generated.")
                        continue
                else:
                    logger.info("No tool calls found in the latest response. Ending agentic loop.")
                    final_response = response_content
                    break

            # --- After the loop ---
            if turn_count >= max_turns:
                logger.warning(f"Agentic loop reached maximum turns ({max_turns}). Returning last response.")
                console.print(Panel(
                    f"[yellow]Reached maximum turns ({max_turns}). Returning last response.[/yellow]",
                    title="Warning",
                    border_style="warning",
                    box=CLI_CONFIG["display"]["panel_box"]
                ))

            # Print the final response
            if CLI_CONFIG["display"]["show_emoji"]:
                display_section_header(console, "Final Response", emoji="success")
            else:
                display_section_header(console, "Final Response")
            
            if isinstance(final_response, str):
                final_json = extract_json(final_response)
                if final_json:
                    json_str = json.dumps(final_json, indent=2)
                    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                    console.print(Panel(
                        syntax,
                        title="Final JSON Response",
                        border_style="success",
                        box=CLI_CONFIG["display"]["panel_box"],
                        padding=(1, 2)
                    ))
                else:
                    try:
                        md = Markdown(final_response)
                        console.print(Panel(
                            md,
                            title="Final Response",
                            border_style="success",
                            box=CLI_CONFIG["display"]["panel_box"],
                            padding=(1, 2)
                        ))
                    except Exception:
                        console.print(Panel(
                            final_response,
                            title="Final Response",
                            border_style="success",
                            box=CLI_CONFIG["display"]["panel_box"],
                            padding=(1, 2)
                        ))
            elif isinstance(final_response, (dict, list)):
                json_str = json.dumps(final_response, indent=2)
                syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                console.print(Panel(
                    syntax,
                    title="Final JSON Response",
                    border_style="success",
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2)
                ))
            else:
                console.print(Panel(
                    str(final_response),
                    title="Final Response",
                    border_style="success",
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2)
                ))
            
            # Final success message
            console.print(Panel(
                Align.center("Thank you for using GLUE Framework!", style="cyan"),
                border_style="success",
                box=CLI_CONFIG["display"]["panel_box"]
            ))
                 
            return True
        else:
            logger.error("No input provided for non-interactive mode")
            console.print("[bold red]Error:[/bold red] No input provided for non-interactive mode")
            return False
        
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        console.print(f"[bold red]Error running application:[/bold red] {e}")
        return False

async def run_interactive_session(app: GlueApp) -> None:
    """Run an interactive session with the GLUE application.
    
    Args:
        app: The GLUE application to run
    """
    logger = logging.getLogger("glue.interactive")
    console = get_console()
    
    # State management for interactive session
    state = {
        "history": [],                           # Message history
        "verbose_mode": False,                   # Show internal messages 
        "step_mode": False,                      # Step by step execution
        "color_enabled": True,                   # Color output
        "thinking_visible": False,               # Show model thinking
        "current_team": None,                    # Currently selected team
        "current_model": None,                   # Currently selected model
        "awaiting_next_step": False              # Waiting for next step in step mode
    }
    
    # Display welcome banner
    display_logo(console)
    
    # Handle the case where app name might be None or empty
    if app.name:
        welcome_line1 = "Welcome to the interactive GLUE session for:"
        welcome_line2 = app.name
    else:
        welcome_line1 = "Welcome to the interactive GLUE session!"
        welcome_line2 = ""
    
    # Create tip text with styled commands
    from rich.text import Text
    
    tip_text = Text("Type ", style="")
    tip_text.append("/help", style="green bold")
    tip_text.append(" to see available commands or ", style="")
    tip_text.append("/exit", style="green bold")
    tip_text.append(" to quit.", style="")
    
    welcome_panel = Panel(
        Group(
            Align.center(welcome_line1, style="cyan"),
            Align.center(welcome_line2, style="bold cyan") if welcome_line2 else None,
            Rule(style="dim"),
            Align.center(tip_text)
        ),
        title="Interactive Mode",
        border_style="green",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2)
    )
    console.print(welcome_panel)
    
    # Display session information
    if CLI_CONFIG["display"]["show_emoji"]:
        display_section_header(console, "Session Information", emoji="info")
    else:
        display_section_header(console, "Session Information")
    
    # Show application structure
    columns = Columns([
        create_status_panel(app, state),
        create_model_tree(app.teams)
    ], equal=True, expand=True)
    console.print(columns)
    
    # Show available tools
    if app.tools:
        if CLI_CONFIG["display"]["show_emoji"]:
            display_section_header(console, "Available Tools", emoji="tool")
        else:
            display_section_header(console, "Available Tools")
        filtered_tools = {name: tool for name, tool in app.tools.items() if name != "communicate"}
        console.print(create_tool_table(filtered_tools))
    
    # Display initial instructions
    console.print("\n[info]Type your message or command below:[/info]")
    
    # Command history for arrow key navigation
    command_history = []
    history_index = 0
    
    while True:
        try:
            # Create prompt based on current context
            prompt_style = CLI_CONFIG["theme"]["prompt"]
            input_style = CLI_CONFIG["theme"]["input"]
            
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
            
            # Get user input
            user_input = await asyncio.to_thread(
                lambda: Prompt.ask(prompt_text, console=console)
            )
            
            # Process exit commands
            if user_input.lower() in ["quit", "exit", "/exit", "/quit"]:
                console.print(f"[{CLI_CONFIG['theme']['success']}]Exiting interactive session.[/{CLI_CONFIG['theme']['success']}]")
                break
                
            # Skip empty input
            if not user_input:
                continue
                
            # Add to command history
            if user_input not in command_history:
                command_history.append(user_input)
                history_index = len(command_history)
            
            # Process command if it starts with /
            if user_input.startswith('/'):
                command, args = parse_interactive_command(user_input)
                
                # Command parsing and handling
                if command == "help":
                    show_interactive_help(console)
                    continue
                    
                elif command == "status":
                    # Display current status information
                    status_panel = create_status_panel(app, state)
                    console.print(status_panel)
                    continue
                    
                elif command == "tools":
                    # Display available tools
                    if CLI_CONFIG["display"]["show_emoji"]:
                        display_section_header(console, "Available Tools", emoji="tool")
                    else:
                        display_section_header(console, "Available Tools")
                    filtered_tools = {name: tool for name, tool in app.tools.items() if name != "communicate"}
                    console.print(create_tool_table(filtered_tools))
                    continue
                    
                elif command == "teams":
                    # Display team structure
                    if CLI_CONFIG["display"]["show_emoji"]:
                        display_section_header(console, "Team Structure", emoji="team")
                    else:
                        display_section_header(console, "Team Structure")
                    console.print(create_team_table(app.teams))
                    continue
                    
                elif command == "models":
                    # Display model structure as a tree
                    if CLI_CONFIG["display"]["show_emoji"]:
                        display_section_header(console, "Model Structure", emoji="model")
                    else: 
                        display_section_header(console, "Model Structure")
                    console.print(create_model_tree(app.teams))
                    continue
                    
                elif command == "clear":
                    # Clear conversation history
                    state["history"] = []
                    console.print(f"[{CLI_CONFIG['theme']['success']}]Conversation history cleared.[/{CLI_CONFIG['theme']['success']}]")
                    continue
                    
                elif command == "verbose":
                    # Toggle verbose mode
                    state["verbose_mode"] = not state["verbose_mode"]
                    mode = "enabled" if state["verbose_mode"] else "disabled"
                    icon = "✅" if state["verbose_mode"] else "❌"
                    console.print(f"[{CLI_CONFIG['theme']['info']}]Verbose mode {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]")
                    continue
                    
                elif command == "step":
                    # Toggle step-by-step mode
                    state["step_mode"] = not state["step_mode"]
                    mode = "enabled" if state["step_mode"] else "disabled"
                    icon = "✅" if state["step_mode"] else "❌"
                    console.print(f"[{CLI_CONFIG['theme']['info']}]Step-by-step mode {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]")
                    continue
                    
                elif command == "next" and state["step_mode"] and state["awaiting_next_step"]:
                    # Proceed to next step in step mode
                    state["awaiting_next_step"] = False
                    console.print(f"[{CLI_CONFIG['theme']['info']}]Proceeding to next step...[/{CLI_CONFIG['theme']['info']}]")
                    # Execution will continue below
                
                elif command == "color":
                    # Toggle color output
                    if len(args) > 0 and args[0].lower() in ["on", "off"]:
                        state["color_enabled"] = args[0].lower() == "on"
                        mode = "enabled" if state["color_enabled"] else "disabled"
                        icon = "✅" if state["color_enabled"] else "❌"
                        console.print(f"[{CLI_CONFIG['theme']['info']}]Color output {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]")
                    else:
                        console.print(f"[{CLI_CONFIG['theme']['warning']}]Usage: /color [on|off][/{CLI_CONFIG['theme']['warning']}]")
                    continue
                
                elif command == "team":
                    # Switch to a specific team
                    if len(args) > 0:
                        team_name = args[0]
                        if team_name in app.teams:
                            state["current_team"] = team_name
                            state["current_model"] = None
                            console.print(f"[{CLI_CONFIG['theme']['success']}]Switched to team: [team]{team_name}[/team][/{CLI_CONFIG['theme']['success']}]")
                        else:
                            console.print(f"[{CLI_CONFIG['theme']['error']}]Team not found: {team_name}[/{CLI_CONFIG['theme']['error']}]")
                    else:
                        console.print(f"[{CLI_CONFIG['theme']['warning']}]Usage: /team [team_name][/{CLI_CONFIG['theme']['warning']}]")
                    continue
                
                elif command == "model":
                    # Switch to a specific model
                    if len(args) > 0:
                        model_name = args[0]
                        # Check if model exists in any team
                        model_found = False
                        for team_name, team in app.teams.items():
                            if hasattr(team, "lead") and team.lead and team.lead.name == model_name:
                                model_found = True
                                break
                            if hasattr(team, "members"):
                                for member in team.members:
                                    if member.name == model_name:
                                        model_found = True
                                        break
                        
                        if model_found:
                            state["current_model"] = model_name
                            state["current_team"] = None
                            model_style = CLI_CONFIG["theme"]["model"].get(
                                model_name, CLI_CONFIG["theme"]["model"]["default"]
                            )
                            console.print(f"[{CLI_CONFIG['theme']['success']}]Switched to model: [{model_style}]{model_name}[/{model_style}][/{CLI_CONFIG['theme']['success']}]")
                        else:
                            console.print(f"[{CLI_CONFIG['theme']['error']}]Model not found: {model_name}[/{CLI_CONFIG['theme']['error']}]")
                    else:
                        console.print(f"[{CLI_CONFIG['theme']['warning']}]Usage: /model [model_name][/{CLI_CONFIG['theme']['warning']}]")
                    continue
                    
                elif command == "thinking":
                    # Toggle model thinking visibility
                    state["thinking_visible"] = not state["thinking_visible"]
                    mode = "visible" if state["thinking_visible"] else "hidden"
                    icon = "✅" if state["thinking_visible"] else "❌"
                    console.print(f"[{CLI_CONFIG['theme']['info']}]Model thinking is now {icon} {mode}.[/{CLI_CONFIG['theme']['info']}]")
                    continue
                
                elif command == "refresh":
                    # Refresh the display
                    console.print(f"[{CLI_CONFIG['theme']['info']}]Refreshing display...[/{CLI_CONFIG['theme']['info']}]")
                    # Show application structure
                    columns = Columns([
                        create_status_panel(app, state),
                        create_model_tree(app.teams)
                    ], equal=True, expand=True)
                    console.print(columns)
                    continue
                
                else:
                    console.print(f"[{CLI_CONFIG['theme']['error']}]Unknown command: {command}[/{CLI_CONFIG['theme']['error']}]")
                    continue
            
            # Process regular input (non-command)
            logger.info(f"User input: {user_input}")
            state["history"].append({"role": "user", "content": user_input})
            
            # Get appropriate spinner based on verbosity
            spinner = CLI_CONFIG["layout"]["loading_animations"].get("dots", "dots")
            
            # Run with or without status indicator based on verbosity level
            if logger.level <= logging.INFO:
                # In verbose mode, don't show spinner to avoid interfering with log output
                logger.debug("Processing input without status indicator (verbose mode)")
                if state["current_team"]:
                    # Direct message to specific team
                    response = await app.teams[state["current_team"]].process_message(user_input)
                elif state["current_model"]:
                    # Direct message to specific model
                    model = None
                    # Find model in teams
                    for team_name, team in app.teams.items():
                        if hasattr(team, "lead") and team.lead and team.lead.name == state["current_model"]:
                            model = team.lead
                            break
                        if hasattr(team, "members"):
                            for member in team.members:
                                if member.name == state["current_model"]:
                                    model = member
                                    break
                    
                    if model:
                        response = await model.generate_response(messages=[{"role": "user", "content": user_input}])
                    else:
                        console.print(f"[{CLI_CONFIG['theme']['error']}]Error: Model {state['current_model']} not found or not accessible.[/{CLI_CONFIG['theme']['error']}]")
                        continue
                else:
                    # Default routing through app
                    response = await app.run(user_input)
            else:
                # In normal mode, show spinner status indicator
                with Status(f"[info]Processing...[/info]", spinner=spinner, console=console):
                    if state["current_team"]:
                        # Direct message to specific team
                        response = await app.teams[state["current_team"]].process_message(user_input)
                    elif state["current_model"]:
                        # Direct message to specific model
                        model = None
                        # Find model in teams
                        for team_name, team in app.teams.items():
                            if hasattr(team, "lead") and team.lead and team.lead.name == state["current_model"]:
                                model = team.lead
                                break
                            if hasattr(team, "members"):
                                for member in team.members:
                                    if member.name == state["current_model"]:
                                        model = member
                                        break
                        
                        if model:
                            response = await model.generate_response(messages=[{"role": "user", "content": user_input}])
                        else:
                            console.print(f"[{CLI_CONFIG['theme']['error']}]Error: Model {state['current_model']} not found or not accessible.[/{CLI_CONFIG['theme']['error']}]")
                            continue
                    else:
                        # Default routing through app
                        response = await app.run(user_input)
            
            # Determine the source (current team, model or app name)
            source = state.get("current_team") or state.get("current_model") or app.name
            
            # Check for warnings in the response if it's a string
            if isinstance(response, str):
                # Look for different warning patterns
                warning_patterns = [
                    "does not support tool use", 
                    "Warning:", 
                    "Error:",
                    "Retrying with"
                ]
                
                # Check if any pattern is in the response
                has_warning = any(pattern in response for pattern in warning_patterns)
                
                if has_warning:
                    # Process multi-line response to extract warnings
                    lines = response.split('\n')
                    warning_lines = []
                    content_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if any(pattern in line for pattern in warning_patterns):
                            # Clean up the warning line
                            clean_line = line
                            if "Warning:" in line:
                                clean_line = line[line.find("Warning:") + 8:].strip()
                            warning_lines.append(clean_line)
                        elif line:  # Skip empty lines
                            content_lines.append(line)
                    
                    # Display all warnings in a single panel
                    if warning_lines:
                        combined_warning = "\n".join(warning_lines)
                        display_warning(console, combined_warning)
                    
                    # Update response to only include content
                    if content_lines:
                        response = "\n".join(content_lines)
                    else:
                        response = "I'll try to process your request."
            
            # Format and display the response
            display_response(console, response, source, state)
            
            # In step mode, wait for user to continue
            if state["step_mode"]:
                state["awaiting_next_step"] = True
                next_step_panel = Panel(
                    "Type [command]/next[/command] to continue or enter a new input to proceed",
                    title="Step Completed",
                    border_style="info",
                    box=CLI_CONFIG["display"]["panel_box"]
                )
                console.print(next_step_panel)
                
        except KeyboardInterrupt:
            console.print(f"\n[{CLI_CONFIG['theme']['warning']}]Interrupted. Exiting session.[/{CLI_CONFIG['theme']['warning']}]")
            break
        except Exception as e:
            logger.error(f"Error during interactive session: {e}", exc_info=True)
            if CLI_CONFIG["display"]["verbose_errors"]:
                console.print_exception()
            else:
                error_panel = Panel(
                    str(e),
                    title="Error Occurred",
                    border_style="error",
                    box=CLI_CONFIG["display"]["panel_box"]
                )
                console.print(error_panel)

    logger.debug("Interactive session ended.")
    console.print(f"[{CLI_CONFIG['theme']['success']}]Interactive session ended. Thank you for using GLUE![/{CLI_CONFIG['theme']['success']}]")

def display_response(console: Console, response: Any, source: str, state: Dict[str, Any]) -> None:
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
        emoji = CLI_CONFIG["emoji"]["team"] if CLI_CONFIG["display"]["show_emoji"] else ""
    elif state.get("current_model"):
        model_name = state.get("current_model")
        panel_style = CLI_CONFIG["theme"]["model"].get(model_name, CLI_CONFIG["theme"]["model"]["default"])
        emoji = CLI_CONFIG["emoji"]["model"] if CLI_CONFIG["display"]["show_emoji"] else ""
    else:
        panel_style = CLI_CONFIG["theme"]["app_name"]
        emoji = CLI_CONFIG["emoji"]["app"] if CLI_CONFIG["display"]["show_emoji"] else ""
    
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
            tool_emoji = CLI_CONFIG["emoji"]["tool"] if CLI_CONFIG["display"]["show_emoji"] else ""
            tool_title = f"{tool_emoji} Tool Call" if tool_emoji else "Tool Call"
            
            # Create nested panels for tool calls
            tool_panel = Panel(
                syntax,
                title=tool_title,
                border_style="tool",
                box=CLI_CONFIG["display"]["panel_box"]
            )
            
            response_panel = Panel(
                tool_panel,
                title=title,
                border_style=panel_style,
                box=CLI_CONFIG["display"]["panel_box"],
                padding=(1, 2)
            )
            
            console.print(response_panel)
            
            # Save to history
            state["history"].append({"role": "assistant", "content": response, "is_tool_call": True})
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
                    padding=(1, 2)
                )
                console.print(response_panel)
            except Exception:
                # Fall back to plain text if markdown rendering fails
                response_panel = Panel(
                    response,
                    title=title,
                    border_style=panel_style,
                    box=CLI_CONFIG["display"]["panel_box"],
                    padding=(1, 2)
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
            padding=(1, 2)
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
            padding=(1, 2)
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
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, "glue.log"), mode='a')
        ]
    )
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Define a simpler format for console output
    if level <= logging.INFO:
        # Detailed format for verbose mode
        console_fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
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
                        title = "❌ Error" if CLI_CONFIG["display"]["show_emoji"] else "Error"
                    elif record.levelno >= logging.WARNING:
                        color = "yellow"
                        prefix = "⚠ Warning: "
                        style = "bold yellow"
                        title = "⚠️ Warning" if CLI_CONFIG["display"]["show_emoji"] else "Warning"
                    else:
                        color = "blue"
                        prefix = "ℹ Info: "
                        style = "blue"
                        title = "ℹ️ Info" if CLI_CONFIG["display"]["show_emoji"] else "Info"
                    
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
                            title="⚠️ OpenRouter Model Limitation" if CLI_CONFIG["display"]["show_emoji"] else "OpenRouter Model Limitation",
                            border_style="yellow",
                            box=CLI_CONFIG["display"]["panel_box"],
                            width=100
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
                            width=100
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
        third_party_logger.addHandler(logging.FileHandler(os.path.join(LOGS_DIR, "glue.log"), mode='a'))
        third_party_logger.propagate = False  # Prevent double logging
    
    return glue_logger


# ==================== Project Management Functions ====================

def create_new_project(project_name: Optional[str] = None, 
                   template: Optional[str] = None, 
                   force: bool = False) -> None:
    """Create a new GLUE project.
    
    Args:
        project_name: Name of the project to create
        template: Template to use (ignored - interactive mode is used)
        force: Whether to overwrite an existing project
    """
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    
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
        padding=(1, 2)
    )
    console.print(welcome_panel)
    console.print()
    
    # If no project name provided, prompt for one
    if not project_name:
        project_name = Prompt.ask("[bold]Enter project name[/bold]", default="my_glue_app")
        
        # Sanitize project name
        project_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
    
    # Check if project directory already exists
    project_dir = os.path.join(os.getcwd(), project_name)
    glue_file = os.path.join(project_dir, f"app.glue")
    
    if os.path.exists(project_dir) and not force:
        error_panel = Panel(
            f"[bold red]Project '{project_name}' already exists.[/bold red]\n\n"
            "To create a new project with this name, use the --force flag:\n"
            f"[dim]glue new {project_name} --force[/dim]",
            title="❌ Error",
            border_style="red",
            padding=(1, 2)
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
                f.write(f"A GLUE project for multi-agent applications.\n\n")
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
                padding=(1, 2)
            )
            console.print()
            console.print(success_panel)
            
        except Exception as e:
            console.print(f"[bold red]Failed to create project configuration: {str(e)}[/bold red]")
            return
                
    except Exception as e:
        console.print(f"[bold red]Failed to create project directory: {str(e)}[/bold red]")
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
            "google": ["gemini-pro", "gemini-ultra"]
        }
        print("\nFallback model information:")
        for provider, models in FALLBACK_MODELS.items():
            print(f"\n{provider.capitalize()} Models:")
            for model in models:
                print(f"  - {model}")
            
def display_tools(args: argparse.Namespace) -> int:
    """Display available tools.

    Args:
        args: The arguments from the parser.

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
            console.print(f"[{CLI_CONFIG['theme']['error']}]Error: File not found: {config_file}[/{CLI_CONFIG['theme']['error']}]")
            sys.exit(1)
        
        # Read the file
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Display file info
        file_stats = os.stat(config_file)
        file_size = file_stats.st_size
        file_modified = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        display_section_header(console, "File Information")
        console.print(f"File: [cyan]{config_file}[/cyan]")
        console.print(f"Size: {file_size:,} bytes")
        console.print(f"Last Modified: {file_modified}")
        console.print(f"Lines: {len(content.splitlines())}")
        console.print("")
        
        # Parse the file with lexer
        lexer = GlueLexer()
        parser = GlueDSLParser()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Lexical analysis
            task1 = progress.add_task("[cyan]Performing lexical analysis...", total=1)
            try:
                tokens = lexer.tokenize(content)
                # Convert tokens to a list so they can be iterated multiple times
                tokens_list = list(tokens)
                progress.update(task1, advance=1, description="[green]Lexical analysis completed")
                syntax_errors = []
            except Exception as e:
                progress.update(task1, advance=1, description="[red]Lexical analysis failed")
                console.print(f"[{CLI_CONFIG['theme']['error']}]Lexical error: {str(e)}[/{CLI_CONFIG['theme']['error']}]")
                sys.exit(1)
            
            # Syntax analysis
            task2 = progress.add_task("[cyan]Performing syntax analysis...", total=1)
            try:
                ast = parser.parse(iter(tokens_list))
                progress.update(task2, advance=1, description="[green]Syntax analysis completed")
            except Exception as e:
                progress.update(task2, advance=1, description="[red]Syntax analysis failed")
                console.print(f"[{CLI_CONFIG['theme']['error']}]Syntax error: {str(e)}[/{CLI_CONFIG['theme']['error']}]")
                sys.exit(1)
            
            # Semantic analysis
            task3 = progress.add_task("[cyan]Performing semantic validation...", total=1)
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
                        semantic_errors.append(f"Model '{model_name}' is missing required 'provider' field")
            
            # Check for teams (magnetize section)
            if "magnetize" not in ast:
                warnings.append("No teams defined (missing 'magnetize' section)")
            else:
                teams = ast["magnetize"]
                for team_name, team_config in teams.items():
                    # Check for lead model
                    if "lead" not in team_config:
                        semantic_errors.append(f"Team '{team_name}' is missing required 'lead' field")
            
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
                            if model_config.get("provider") == "openai" and "model" not in config:
                                semantic_errors.append(f"OpenAI model '{model_name}' is missing 'model' in config")
                        else:
                            warnings.append(f"Model '{model_name}' has no configuration section")
                
                # Check tool configs
                if "tools" in ast:
                    for tool_name, tool_config in ast["tools"].items():
                        if tool_name not in AVAILABLE_TOOLS:
                            warnings.append(f"Unknown tool: '{tool_name}'")
            
            if semantic_errors:
                progress.update(task3, advance=1, description="[red]Semantic validation failed")
            else:
                progress.update(task3, advance=1, description="[green]Semantic validation completed")
        
        # Display validation results
        display_section_header(console, "Validation Results")
        
        if not syntax_errors and not semantic_errors:
            console.print(f"[{CLI_CONFIG['theme']['success']}]✓ File is valid![/{CLI_CONFIG['theme']['success']}]")
        
        if syntax_errors:
            console.print(f"[{CLI_CONFIG['theme']['error']}]Syntax Errors:[/{CLI_CONFIG['theme']['error']}]")
            for i, error in enumerate(syntax_errors, 1):
                console.print(f"  {i}. {error}")
        
        if semantic_errors:
            console.print(f"[{CLI_CONFIG['theme']['error']}]Semantic Errors:[/{CLI_CONFIG['theme']['error']}]")
            for i, error in enumerate(semantic_errors, 1):
                console.print(f"  {i}. {error}")
        
        if warnings:
            console.print(f"[{CLI_CONFIG['theme']['warning']}]Warnings:[/{CLI_CONFIG['theme']['warning']}]")
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
        console.print(f"[{CLI_CONFIG['theme']['error']}]Error validating file: {e}[/{CLI_CONFIG['theme']['error']}]")
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
        Layout(name="footer", size=3)
    )
    
    # Split main area into sidebar and content
    layout["main"].split_row(
        Layout(name="sidebar", size=30),
        Layout(name="content")
    )
    
    # Add sub-layouts in content area
    layout["content"].split(
        Layout(name="output", ratio=3),
        Layout(name="input", size=5)
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
        title_style="tool"
    )
    
    # Add emoji if enabled
    tool_emoji = CLI_CONFIG["emoji"]["tool"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    
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
        title_style="team"
    )
    
    # Add emoji if enabled
    team_emoji = CLI_CONFIG["emoji"]["team"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    model_emoji = CLI_CONFIG["emoji"]["model"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    tool_emoji = CLI_CONFIG["emoji"]["tool"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    
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
    team_emoji = CLI_CONFIG["emoji"]["team"] if CLI_CONFIG["display"]["show_emoji"] else "●"
    model_emoji = CLI_CONFIG["emoji"]["model"] if CLI_CONFIG["display"]["show_emoji"] else "○"
    
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
    model_emoji = CLI_CONFIG["emoji"]["model"] + " " if CLI_CONFIG["display"]["show_emoji"] else ""
    
    # Determine model style
    model_name = model.get("name", "Unknown")
    style = CLI_CONFIG["theme"]["model"].get(model_name, CLI_CONFIG["theme"]["model"]["default"])
    
    # Create model info
    model_info = Group(
        Text(f"Provider: {model.get('provider', 'Unknown')}", style="muted"),
        Text(f"Base Model: {model.get('base_model', 'Unknown')}", style="app.value"),
        Text(f"Temperature: {model.get('temperature', 0.7)}", style="muted"),
        Rule(style="dim"),
        Markdown(f"**Role:** {model.get('role', 'No role defined')}"),
        Text(f"Adhesives: {', '.join(model.get('adhesives', []))}", style="code")
    )
    
    panel = Panel(
        model_info,
        title=f"{model_emoji}{model_name}",
        title_align="left",
        border_style=style,
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2)
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
            Text("Use these commands to interact with the GLUE application:", style="info"),
            commands_section,
            Text("\nTip: You can type a message directly to interact with the active model or team.", style="muted")
        ),
        title="Interactive Mode Help",
        title_align="center",
        border_style="blue",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2)
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
    status_table.add_row("App Name:", Text(app.name, style="app.name" if app.name else "Undefined"))
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
    status_table.add_row("Tools:", str(len(app.tools)-1))
    
    # Add mode settings
    status_table.add_row("Verbose Mode:", "✅ Enabled" if state.get("verbose_mode") else "❌ Disabled")
    status_table.add_row("Step Mode:", "✅ Enabled" if state.get("step_mode") else "❌ Disabled")
    status_table.add_row("Color Output:", "✅ Enabled" if state.get("color_enabled") else "❌ Disabled")
    
    # Add context info
    if state.get("current_team"):
        status_table.add_row("Active Team:", Text(state["current_team"], style="team"))
    elif state.get("current_model"):
        style = CLI_CONFIG["theme"]["model"].get(
            state["current_model"], CLI_CONFIG["theme"]["model"]["default"])
        status_table.add_row("Active Model:", Text(state["current_model"], style=style))
    
    # Create panel
    panel = Panel(
        status_table,
        title="Application Status",
        title_align="center",
        border_style="blue",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2)
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
def main():
    """Main CLI entry point"""
    # Get a themed console
    console = get_console()
    
    
    try:
        # Display logo if not in a script or piped environment
        if sys.stdout.isatty():
            display_logo(console)
        
        # Set up argument parser with rich description
        parser = argparse.ArgumentParser(
            description="GLUE Framework CLI - GenAI Linking & Unification Engine",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
Examples:
  glue run examples/basic.glue --interactive      Run a GLUE app in interactive mode
  glue new my_research_app --template research    Create a new research project
  glue forge tool my_tool --description "..."     Create a custom tool
  glue list-models                                List available models
  glue --help                                     Show this help message
            """
        )
        
        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Run command
        run_parser = subparsers.add_parser("run", help="Run a GLUE application")
        run_parser.add_argument("config", help="Path to GLUE config file")
        run_parser.add_argument("--input", "-i", help="Input text for the app")
        run_parser.add_argument("--interactive", "-I", action="store_true", 
                             help="Run in interactive mode")
        run_parser.add_argument("--verbose", "-v", action="count", default=0,
                             help="Enable verbose logging (use -vv for debug level)")
        run_parser.add_argument("--env", "-e", help="Path to .env file")
        
        # New command
        new_parser = subparsers.add_parser("new", help="Create a new GLUE project")
        new_parser.add_argument("project", help="Project name", nargs="?")
        new_parser.add_argument("--template", "-t", 
                              help="Ignored - interactive mode is always used",
                              default="interactive")
        
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
        if getattr(args, 'verbose', 0) > 1:
            log_level = logging.DEBUG
        elif getattr(args, 'verbose', 0) == 1:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
            
        
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
                # If no project name provided, just pass None and let create_new_project handle it
                create_new_project(args.project)
                
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
                display_tools(args)
                
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
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

def get_template_content(template: str, project_name: str) -> str:
    """Create a GLUE file interactively with the user.
    
    This function provides a rich, wizard-style interface for creating a new GLUE
    application configuration. It guides users through a step-by-step process for defining
    models, teams, tools, and other application settings.
    
    Args:
        template: Ignored - interactive mode is always used
        project_name: Name of the project
        
    Returns:
        Generated GLUE file content as a string
    """
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    from rich.style import Style
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.live import Live
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.align import Align
    
    console = get_console()
    
    # Container for all wizard data
    wizard_data = {
        "current_step": 1,
        "total_steps": 6,
        "app_name": project_name,
        "description": f"A multi-agent GLUE application for {project_name}",
        "version": "0.1.0",
        "models": [],
        "teams": [],
        "tools": []
    }
    
    # Helper function to display wizard progress
    def show_wizard_progress(current_step, total_steps=6):
        progress_width = 40
        filled = int((current_step / total_steps) * progress_width)
        
        console.print(f"\n[bright_blue]Step {current_step}/{total_steps}:[/bright_blue] Progress")
        console.print(f"[bright_blue]{'━' * filled}[/bright_blue][dim]{'━' * (progress_width - filled)}[/dim]")
        console.print()
    
    # Helper function to display section transition animation
    def transition_to_section(title, step, total_steps=6):
        console.print(f"[dim]Moving to {title} section...[/dim]")
        time.sleep(0.2)  # Just a short pause instead of lengthy animation
    
    # Helper function to create section headers with progress
    def section_header(title, step, total_steps=6):
        console.print("\n\n")
        
        # Update and show progress
        wizard_data["current_step"] = step
        show_wizard_progress(step, total_steps)
        
        # Create a simpler header
        console.print(f"[bold white on bright_blue] STEP {step}: {title.upper()} [/bold white on bright_blue]")
        console.print()
    
    # Helper function to create subsection headers
    def subsection_header(title):
        console.print()
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(Rule(style="bright_blue", characters="·"))
    
    # Helper for creating formatted options
    def show_options(options, default=None, descriptions=None):
        table = Table(show_header=False, box=box.SIMPLE, expand=False)
        table.add_column("Number", style="bold yellow", justify="right", width=4)
        table.add_column("Option", style="cyan")
        if descriptions:
            table.add_column("Description", style="dim")
        
        for i, option in enumerate(options, 1):
            if descriptions:
                desc = descriptions[i-1] if i-1 < len(descriptions) else ""
                if default is not None and i == default:
                    table.add_row(f"{i}", f"[bold]{option}[/bold] [dim](default)[/dim]", desc)
                else:
                    table.add_row(f"{i}", option, desc)
            else:
                if default is not None and i == default:
                    table.add_row(f"{i}", f"[bold]{option}[/bold] [dim](default)[/dim]")
                else:
                    table.add_row(f"{i}", option)
        
        console.print(table)
    
    # Display tip panel
    def show_tip(message, icon="💡"):
        console.print(f"\n{icon} [bold cyan]Tip:[/bold cyan] {message}\n")
    
    # Display custom info panel
    def show_info(message, title="Information"):
        console.print(f"\nℹ️ [bold blue]{title}:[/bold blue] {message}\n")
    
    # Welcome screen
    console.clear()
    display_logo(console, show_version=True)
    console.print()
    
    # Simplified welcome panel with proper markup
    welcome_panel = Panel(
        f"[bold]Welcome to the GLUE Project Wizard![/bold]\n\n"
        f"Creating project: [bold]{project_name}[/bold]\n\n"
        "This wizard will guide you through creating your GLUE application",
        title="🚀 Create New GLUE Project",
        border_style="bright_blue",
        padding=(1, 4)
    )
    console.print(welcome_panel)
    console.print()
    
    # Skip the animation, just a small delay
    time.sleep(0.5)
    
    #----------------------------------------
    # 1. APP CONFIGURATION
    #----------------------------------------
    transition_to_section("APPLICATION DETAILS", 1)
    section_header("APPLICATION DETAILS", 1)
    
    show_tip("This section defines the basic properties of your GLUE application.")
    
    # Application name and description
    app_name = project_name
    description = Prompt.ask(
        "[bold cyan]Description[/bold cyan]", 
        default=f"A multi-agent GLUE application for {project_name}"
    )
    wizard_data["description"] = description
    
    # Version is now automatic for first-time projects
    version = "0.1.0"
    
    # Application configuration
    subsection_header("Application Configuration")
    
    # Show configuration options in columns
    config_table = Table.grid(padding=(0, 4))
    config_table.add_column()
    config_table.add_column()
    
    development_mode = Confirm.ask("[bold cyan]Enable development mode?[/bold cyan]", default=True)
    sticky_mode = Confirm.ask("[bold cyan]Enable persistence (sticky mode)?[/bold cyan]", default=True)
    portkey_enabled = Confirm.ask("[bold cyan]Enable Portkey integration?[/bold cyan]", default=False)
    
    # Show summary of the configuration
    config_summary = Table.grid(padding=1)
    config_summary.add_column(style="bright_blue")
    config_summary.add_column(style="bright_white")
    config_summary.add_row("Project Name:", f"[bold]{app_name}[/bold]")
    config_summary.add_row("Description:", description)
    config_summary.add_row("Version:", version)
    config_summary.add_row("Development Mode:", "✅ Enabled" if development_mode else "❌ Disabled")
    config_summary.add_row("Persistence:", "✅ Enabled" if sticky_mode else "❌ Disabled")
    config_summary.add_row("Portkey Integration:", "✅ Enabled" if portkey_enabled else "❌ Disabled")
    
    console.print("\n")
    console.print(Panel(
        config_summary,
        title="📋 Application Summary",
        border_style="bright_blue",
        box=box.ROUNDED
    ))
    console.print()
    
    #----------------------------------------
    # 2. MODELS CONFIGURATION
    #----------------------------------------
    transition_to_section("MODELS", 2)
    section_header("MODELS", 2)
    
    show_tip("Models are AI agents that perform specific roles in your application. Each model has a provider, capabilities, and configuration.")
    
    models = []
    model_names = []
    
    console.print("[cyan]Let's add some models to your application.[/cyan]")
    console.print("[dim]You should create at least one model to serve as a team lead.[/dim]")
    
    # Add multiple models
    add_model = True
    while add_model:
        if models:
            subsection_header(f"Model #{len(models) + 1}")
        
        # 2.1 Model name and provider
        model_name = Prompt.ask(
            "[bold cyan]Model name[/bold cyan]", 
            default=f"model_{len(models) + 1}"
        )
        
        # Ensure unique model names
        while model_name in model_names:
            console.print(f"[bold red]Model name '{model_name}' already exists. Please choose another name.[/bold red]")
            model_name = Prompt.ask("[bold cyan]Model name[/bold cyan]", default=f"model_{len(models) + 1}")
        
        model_names.append(model_name)
        
        # 2.2 Provider selection
        console.print("\n[bold cyan]Select a model provider:[/bold cyan]")
        providers = ["openai", "anthropic", "openrouter", "huggingface", "custom"]
        provider_descriptions = [
            "OpenAI models (GPT-4, GPT-3.5)",
            "Anthropic models (Claude)",
            "OpenRouter (Multiple providers)",
            "Hugging Face models (Open source)",
            "Custom provider integration"
        ]
        show_options(providers, default=3, descriptions=provider_descriptions)  # OpenRouter as default
        
        provider_idx = Prompt.ask("[bold cyan]Provider[/bold cyan]", default="3")
        try:
            provider_idx = int(provider_idx) - 1
            if 0 <= provider_idx < len(providers):
                provider = providers[provider_idx]
            else:
                provider = "openrouter"
        except ValueError:
            provider = "openrouter"
        
        # 2.3 Role description
        role = Prompt.ask(
            "[bold cyan]Role description[/bold cyan]", 
            default=f"An AI assistant specialized in {model_name.replace('_', ' ')}"
        )
        
        # 2.4 Model config
        console.print("\n[bold cyan]Specify model configuration:[/bold cyan]")
        model_type = None
        temperature = 0.7
        
        if provider == "openai":
            console.print("[bold cyan]Select OpenAI model:[/bold cyan]")
            openai_models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            openai_descriptions = [
                "Latest GPT-4 iteration (recommended)",
                "Standard GPT-4 model",
                "Faster, more economical model"
            ]
            show_options(openai_models, descriptions=openai_descriptions)
            model_choice = Prompt.ask("[bold cyan]Choose model[/bold cyan]", default="1")
            
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(openai_models):
                    model_type = openai_models[model_idx]
                else:
                    model_type = "gpt-4-turbo"
            except ValueError:
                model_type = "gpt-4-turbo"
                
            # Show temperature slider
            console.print("\n[bold cyan]Temperature setting:[/bold cyan]")
            console.print("[dim]Lower values (0.0-0.5): More deterministic, focused responses[/dim]")
            console.print("[dim]Higher values (0.7-1.0): More creative, varied responses[/dim]")
            console.print(f"[bright_blue]{'◆' * 7}[/bright_blue][dim]{'◇' * 3}[/dim] [bright_white]0.7 (Default)[/bright_white]")
            
            temp_input = Prompt.ask("[bold cyan]Temperature (0.0-1.0)[/bold cyan]", default="0.7")
            try:
                temp_value = float(temp_input)
                if 0.0 <= temp_value <= 1.0:
                    temperature = temp_value
            except ValueError:
                pass
                
        # Similar sections for other providers...
        elif provider == "anthropic":
            # Implementation for Anthropic
            console.print("[bold cyan]Select Anthropic model:[/bold cyan]")
            anthropic_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2.1"]
            anthropic_descriptions = [
                "Most capable Claude model",
                "Balanced performance and speed",
                "Fastest, most compact Claude model",
                "Previous generation model"
            ]
            show_options(anthropic_models, descriptions=anthropic_descriptions, default=2)
            model_choice = Prompt.ask("[bold cyan]Choose model[/bold cyan]", default="2")
            
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(anthropic_models):
                    model_type = anthropic_models[model_idx]
                else:
                    model_type = "claude-3-sonnet"
            except ValueError:
                model_type = "claude-3-sonnet"
                
            # Temperature setting for Anthropic
            console.print("\n[bold cyan]Temperature setting:[/bold cyan]")
            console.print("[dim]Lower values (0.0-0.5): More deterministic, focused responses[/dim]")
            console.print("[dim]Higher values (0.7-1.0): More creative, varied responses[/dim]")
            console.print(f"[bright_blue]{'◆' * 7}[/bright_blue][dim]{'◇' * 3}[/dim] [bright_white]0.7 (Default)[/bright_white]")
            
            temp_input = Prompt.ask("[bold cyan]Temperature (0.0-1.0)[/bold cyan]", default="0.7")
            try:
                temp_value = float(temp_input)
                if 0.0 <= temp_value <= 1.0:
                    temperature = temp_value
            except ValueError:
                pass
        
        # Add model to list
        model_config = {
            "name": model_name,
            "provider": provider,
            "role": role,
            "model": model_type,
            "temperature": temperature,
            "adhesives": ["glue", "velcro"]  # Default adhesives to prevent errors
        }
        models.append(model_config)
        wizard_data["models"].append(model_config)
        
        # Show model card with configuration
        model_card = Table.grid(padding=1)
        model_card.add_column(style="bright_blue")
        model_card.add_column(style="bright_white")
        model_card.add_row("Name:", f"[bold]{model_name}[/bold]")
        model_card.add_row("Provider:", provider)
        model_card.add_row("Model:", model_type or "Not specified")
        model_card.add_row("Role:", role)
        model_card.add_row("Temperature:", f"{temperature}")
        model_card.add_row("Adhesives:", ", ".join(model_config["adhesives"]))
        
        console.print("\n")
        console.print(Panel(
            model_card,
            title=f"🤖 Model: {model_name}",
            border_style="green",
            box=box.ROUNDED
        ))
        
        # Ask if user wants to add another model
        if len(models) >= 1:
            add_another = Confirm.ask("\n[bold cyan]Add another model?[/bold cyan]", default=True)
            add_model = add_another
    
    #----------------------------------------
    # 3. TOOLS CONFIGURATION
    #----------------------------------------
    transition_to_section("TOOLS", 3)
    section_header("TOOLS", 3)
    
    show_tip("Tools allow your models to interact with external systems and perform actions. Add tools that your models will need to accomplish tasks.")
    
    tools = []
    tool_names = []
    
    console.print("[cyan]Let's add some tools to your application.[/cyan]")
    
    # 3.1 Web Search Tool
    if Confirm.ask("[bold]Add web search tool?[/bold]", default=True):
        subsection_header("Web Search Tool")
        
        tool_name = "web_search"
        provider = "serp"
        max_results = Prompt.ask("[bold]Maximum search results[/bold]", default="5")
        try:
            max_results = int(max_results)
        except ValueError:
            max_results = 5
            
        tools.append({
            "name": tool_name,
            "provider": provider,
            "max_results": max_results
        })
        tool_names.append(tool_name)
        
        console.print(Panel(
            "[green]✓[/green] Web search tool added",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1)
        ))
    
    # 3.2 File Handler Tool
    if Confirm.ask("[bold]Add file handler tool?[/bold]", default=True):
        subsection_header("File Handler Tool")
        
        tool_name = "file_handler"
        description = "Read and write files"
        base_path = Prompt.ask("[bold]Base path for file operations[/bold]", default="./workspace")
        
        tools.append({
            "name": tool_name,
            "description": description,
            "base_path": base_path
        })
        tool_names.append(tool_name)
        
        console.print(Panel(
            "[green]✓[/green] File handler tool added",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1)
        ))
    
    # 3.3 Code Interpreter Tool
    if Confirm.ask("[bold]Add code interpreter tool?[/bold]", default=True):
        subsection_header("Code Interpreter Tool")
        
        tools.append({
            "name": "code_interpreter",
            "description": "Execute Python code"
        })
        tool_names.append("code_interpreter")
        
        console.print(Panel(
            "[green]✓[/green] Code interpreter tool added",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1)
        ))
    
    # 3.4 Communicate Tool (if more than one model)
    if len(models) > 1:
        subsection_header("Communication Tool")
        
        # Automatically add communicate tool without asking
        tools.append({
            "name": "communicate",
            "description": "Communicate with other models and teams"
        })
        tool_names.append("communicate")
        
        console.print(Panel(
            "[green]✓[/green] Communication tool added automatically (required for multi-model projects)",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1)
        ))
    
    # 3.5 Custom Tools
    add_custom = Confirm.ask("[bold cyan]Add a custom tool?[/bold cyan]", default=False)
    while add_custom:
        subsection_header("Custom Tool")
        
        custom_tool_name = Prompt.ask("[bold]Tool name[/bold]")
        custom_tool_description = Prompt.ask("[bold]Tool description[/bold]")
        
        tools.append({
            "name": custom_tool_name,
            "description": custom_tool_description
        })
        tool_names.append(custom_tool_name)
        
        console.print(Panel(
            f"[green]✓[/green] Custom tool '{custom_tool_name}' added",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 1)
        ))
        
        # Ask once more if they want to add another custom tool
        add_custom = Confirm.ask("[bold]Add another custom tool?[/bold]", default=False)
    
    #----------------------------------------
    # 4. TEAMS CONFIGURATION
    #----------------------------------------
    transition_to_section("TEAMS", 4)
    section_header("TEAMS", 4)
    
    show_tip("Teams organize models and assign them tools to work together. Each team needs a lead model and can have additional members.")
    
    teams = []
    team_names = []
    
    console.print("[cyan]Let's create teams for your models.[/cyan]")
    
    # Must have at least one team
    add_team = True
    available_models = [m["name"] for m in models]
    
    while add_team and available_models:
        # Initialize current_team at the start of each loop
        current_team = {}
        
        if teams:
            subsection_header(f"Team #{len(teams) + 1}")
        
        # 4.1 Team name
        team_name = Prompt.ask(
            "[bold cyan]Team name[/bold cyan]", 
            default=f"team_{len(teams) + 1}"
        )
        
        # Ensure unique team names
        while team_name in team_names:
            console.print(f"[bold red]Team name '{team_name}' already exists. Please choose another name.[/bold red]")
            team_name = Prompt.ask("[bold cyan]Team name[/bold cyan]", default=f"team_{len(teams) + 1}")
        
        team_names.append(team_name)
        current_team["name"] = team_name
        
        # 4.2 Select lead model
        console.print("\n[bold cyan]Select lead model for this team:[/bold cyan]")
        show_options(available_models)
        
        lead_idx = Prompt.ask("[bold cyan]Lead model (enter number)[/bold cyan]", default="1")
        try:
            lead_idx = int(lead_idx) - 1
            if 0 <= lead_idx < len(available_models):
                lead_model = available_models[lead_idx]
            else:
                lead_model = available_models[0]
        except ValueError:
            lead_model = available_models[0]
            
        current_team["lead"] = lead_model
        
        # Remove lead from available models for members
        remaining_models = [m for m in available_models if m != lead_model]
        
        # 4.3 Select member models
        member_models = []
        if remaining_models:
            console.print("\n[bold cyan]Select member models for this team:[/bold cyan]")
            
            for model in remaining_models:
                if Confirm.ask(f"[bold cyan]Add '{model}' as a member of team '{team_name}'?[/bold cyan]", default=False):
                    member_models.append(model)
        
        current_team["members"] = member_models
        
        # 4.4 Select tools for this team
        team_tools = []
        if tools:
            console.print("\n[bold cyan]Select tools for this team:[/bold cyan]")
            
            for tool in tools:
                tool_name = tool["name"]
                if Confirm.ask(f"[bold cyan]Add '{tool_name}' tool to team '{team_name}'?[/bold cyan]", default=True):
                    team_tools.append(tool_name)
        
        current_team["tools"] = team_tools
        
        # Add team to teams list
        teams.append(current_team)
        
        # Create a summary of the team
        member_list = ", ".join(member_models) if member_models else "None"
        tool_list = ", ".join(team_tools) if team_tools else "None"
        
        team_summary = Panel(
            f"[bold]Team:[/bold] {team_name}\n"
            f"[bold]Lead:[/bold] {lead_model}\n"
            f"[bold]Members:[/bold] {member_list}\n"
            f"[bold]Tools:[/bold] {tool_list}",
            title="✅ Team Created",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(team_summary)
        
        # Update available models for next team - make sure to exclude all assigned models
        used_models = [current_team["lead"]] + current_team["members"]
        for existing_team in teams[:-1]:  # Skip the team we just added
            used_models.append(existing_team["lead"])
            used_models.extend(existing_team["members"])
        
        available_models = [m for m in [model["name"] for model in models] if m not in used_models]
        
        if available_models:
            add_team = Confirm.ask("\n[bold cyan]Add another team?[/bold cyan]", default=True)
        else:
            console.print("[dim]All models have been assigned to teams.[/dim]")
            add_team = False
    
    #----------------------------------------
    # 5. TEAM FLOWS
    #----------------------------------------
    transition_to_section("TEAM FLOWS", 5)
    section_header("TEAM FLOWS", 5)
    
    team_flows = []
    
    if len(teams) > 1:
        show_tip("Team flows define how information moves between teams. Different flow types determine how teams can share information.")
        
        console.print("[cyan]Define how information flows between your teams.[/cyan]")
        
        # Always configure flows (no prompt asking if user wants to configure)
        flow_table = Table(box=box.SIMPLE)
        flow_table.add_column("Flow Type", style="cyan")
        flow_table.add_column("Description")
        
        # Remove the one-way flow option
        flow_table.add_row("Bidirectional", "Information flows both ways between teams")
        flow_table.add_row("Pull-based", "Target team can request information from source")
        flow_table.add_row("Push-based", "Source team actively sends information to target")
        flow_table.add_row("No flow", "No information exchange between teams")
        
        console.print(flow_table)
        console.print()
        
        for i, source_team in enumerate(teams):
            for target_team in teams[i+1:]:
                console.print(Panel(
                    f"Configuring flow between [cyan]{source_team['name']}[/cyan] and [cyan]{target_team['name']}[/cyan]",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(1, 2)
                ))
                
                console.print("[bold]Choose flow type:[/bold]")
                # Remove the one-way options
                console.print("[dim]1. Bidirectional (source <-> target)[/dim]")
                console.print("[dim]2. Pull-based (target pulls from source)[/dim]")
                console.print("[dim]3. Push-based (source pushes to target)[/dim]")
                console.print("[dim]4. No flow[/dim]")
                
                flow_type = Prompt.ask("[bold]Flow type[/bold]", default="1")
                
                flow_description = ""
                if flow_type == "1":
                    team_flows.append({
                        "source": source_team['name'],
                        "target": target_team['name'],
                        "type": "bidirectional"
                    })
                    flow_description = f"{source_team['name']} <-> {target_team['name']}"
                elif flow_type == "2":
                    team_flows.append({
                        "source": source_team['name'],
                        "target": target_team['name'],
                        "type": "pull"
                    })
                    flow_description = f"{target_team['name']} pulls from {source_team['name']}"
                elif flow_type == "3":
                    team_flows.append({
                        "source": source_team['name'],
                        "target": target_team['name'],
                        "type": "push"
                    })
                    flow_description = f"{source_team['name']} pushes to {target_team['name']}"
                elif flow_type == "4":
                    team_flows.append({
                        "source": source_team['name'],
                        "target": target_team['name'],
                        "type": "none"
                    })
                    flow_description = f"No flow between {source_team['name']} and {target_team['name']}"
                
                console.print(Panel(
                    f"[green]✓[/green] Flow configured: {flow_description}",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 1)
                ))
    else:
        console.print("[dim]You only have one team, so no team flows are needed.[/dim]")
    
    #----------------------------------------
    # 6. GENERATE GLUE FILE
    #----------------------------------------
    transition_to_section("GENERATING GLUE FILE", 6)
    section_header("GENERATING GLUE FILE", 6)
    
    # Helper function to format lists safely
    def format_list_safely(prefix, items):
        """Create a properly formatted list line without any string manipulation issues."""
        console = get_console() # Get console for debugging inside function
        console.print(f"[dim]Inside format_list_safely for prefix '{prefix}' with items: {items}[/dim]")
        if not items:
            console.print("[dim]format_list_safely: No items provided, returning empty list.[/dim]")
            return f"{prefix} = []"

        # Ensure all items are strings before attempting to build the string
        string_items = []
        for item in items:
            try:
                item_str = str(item)
                string_items.append(item_str)
            except Exception as e:
                console.print(f"[red]Error converting item '{item}' to string: {e}[/red]")
                string_items.append("") # Add empty string on error

        console.print(f"[dim]format_list_safely: Items converted to strings: {string_items}[/dim]")

        # Use manual concatenation to build the string, avoiding join()
        items_str = ""
        try:
            num_items = len(string_items)
            for i, item_str in enumerate(string_items):
                items_str += item_str
                if i < num_items - 1:
                    items_str += ", "
            console.print(f"[dim]format_list_safely: Manually joined items_str: '{items_str}'[/dim]")

            # Return complete line using BASIC CONCATENATION instead of f-string
            final_line = prefix + " = [" + items_str + "]"
            console.print(f"[dim]format_list_safely: Final constructed line (basic concat): '{final_line}'[/dim]")
            return final_line
        except Exception as e:
            console.print(f"[red]Error during manual string building or f-string creation: {str(e)}[/red]")
            return f"{prefix} = []" # Fallback on error
    
    # Plain string-based approach without complex formatting
    app_config = []
    app_config.append(f"// {project_name} GLUE Application")
    app_config.append("// Generated with GLUE Framework's Interactive Builder")
    app_config.append("")
    app_config.append("glue app {")
    app_config.append(f'    name = "{app_name}"')
    app_config.append(f'    description = "{description}"')
    app_config.append(f'    version = "{version}"')
    app_config.append("    config {")
    
    if development_mode:
        app_config.append("        development = true")
    if sticky_mode:
        app_config.append("        sticky = true  // Enable persistence")
    if portkey_enabled:
        app_config.append("        portkey = true  // Enable Portkey integration")
    
    app_config.append("    }")
    app_config.append("}")
    
    # Models section
    for model in models:
        app_config.append("")
        app_config.append(f"// Model: {model['name']}")
        app_config.append(f"model {model['name']} {{")
        app_config.append(f"    provider = {model['provider']}")
        app_config.append(f"    role = \"{model['role']}\"")
        
        # Add adhesives
        adhesives_str = ", ".join([f'"{a}"' for a in model["adhesives"]]) if model["adhesives"] else ""
        app_config.append(f"    adhesives = [{adhesives_str}]")
        
        # Config section
        app_config.append("    config {")
        
        if "model" in model and model["model"]:
            app_config.append(f"        model = \"{model['model']}\"")
        
        app_config.append(f"        temperature = {model['temperature']}")
        app_config.append("    }")
        app_config.append("}")
    
    # Tools section
    if tools:
        app_config.append("")
        app_config.append("// Tools")
        
        for tool in tools:
            app_config.append(f"tool {tool['name']} {{")
            
            if "provider" in tool:
                app_config.append(f"    provider = {tool['provider']}")
                app_config.append("    config {")
                if "max_results" in tool:
                    app_config.append(f"        max_results = {tool['max_results']}")
                app_config.append("    }")
            elif "description" in tool:
                app_config.append(f"    description = \"{tool['description']}\"")
                if "base_path" in tool:
                    app_config.append("    config {")
                    app_config.append(f"        base_path = \"{tool['base_path']}\"")
                    app_config.append("    }")
            
            app_config.append("}")
    
    # Teams and flows
    app_config.append("")
    app_config.append("// Teams and flows")
    app_config.append("magnetize {")
    
    # Teams
    for team in teams:
        # Print raw debug for diagnostics
        console.print(f"[dim]TEAM DEBUG: {team['name']}, members={team['members']}, tools={team['tools']}[/dim]")
        
        # Ensure members and tools are lists
        if not isinstance(team['members'], list):
            console.print(f"[red]WARNING: members for {team['name']} is not a list, converting: {team['members']}")
            team['members'] = [] if team['members'] is None else [str(team['members'])]
            
        if not isinstance(team['tools'], list):
            console.print(f"[red]WARNING: tools for {team['name']} is not a list, converting: {team['tools']}")
            team['tools'] = [] if team['tools'] is None else [str(team['tools'])]
        
        # Convert any None or non-string items to strings
        team['members'] = [str(m) if m is not None else "" for m in team['members']]
        team['tools'] = [str(t) if t is not None else "" for t in team['tools']]
        
        app_config.append(f"    {team['name']} {{")
        app_config.append(f"        lead = {team['lead']}")
        
        # Fix members formatting - ensure it's a properly formatted list
        if team['members']:
            # Debug raw members
            console.print(f"[dim]DEBUG: Raw members before processing: {team['members']}[/dim]")
            
            # Format member names with quotes
            quoted_members = [f'"{str(m)}"' for m in team['members']]
            
            # Use the safe formatting function
            member_line = format_list_safely("        members", quoted_members)
            console.print(f"[dim]MEMBERS LINE (safe): '{member_line}'[/dim]")
            app_config.append(member_line)
        else:
            app_config.append("        members = []")
        
        # --- START REPLACEMENT BLOCK ---
        # Fix tools formatting - REVERT TO USING HELPER FUNCTION like members
        if team['tools']:
            # Debug raw tools
            console.print(f"[dim]DEBUG: Raw tools before HELPER processing: {team['tools']}[/dim]")
            
            # Extra safety - ensure we have strings (keep this part)
            sanitized_tools = []
            for tool in team['tools']:
                tool_str = str(tool).strip()
                if tool_str:  # Only add non-empty strings
                    sanitized_tools.append(tool_str)
                    console.print(f"[dim]Sanitized tool for helper: '{tool_str}'[/dim]")
            
            # Use the safe formatting function for tools (no quotes needed)
            tool_line = format_list_safely("        tools", sanitized_tools)
            console.print(f"[dim]TOOLS LINE FROM HELPER (safe): '{tool_line}'[/dim]")
            
            # Append the result from the helper function
            app_config.append(tool_line)
        else:
            app_config.append("        tools = []")
        # --- END REPLACEMENT BLOCK ---
        
        app_config.append("    }")
        app_config.append("")
    
    # Flows
    if team_flows:
        app_config.append("    // Information flows between teams")
        for flow in team_flows:
            if flow["type"] == "bidirectional":
                app_config.append(f"    {flow['source']} >< {flow['target']}  // Bidirectional")
            elif flow["type"] == "pull":
                app_config.append(f"    {flow['target']} <- {flow['source']} pull")
            elif flow["type"] == "push":
                app_config.append(f"    {flow['source']} -> {flow['target']} push")
            elif flow["type"] == "none":
                app_config.append(f"    {flow['source']} <> {flow['target']}  // No flow")
    
    # Handle the case where we want to show a no-flow relationship
    elif len(teams) > 1:
        # Add explicit no-flow relationships between all teams
        for i, source_team in enumerate(teams):
            for target_team in teams[i+1:]:
                app_config.append(f"    {source_team['name']} <> {target_team['name']}  // No flow")
    
    # Close magnetize block
    app_config.append("}")
    app_config.append("")
    
    # Apply glue
    app_config.append("apply glue")
    
    # Join everything into a single string
    glue_content = "\n".join(app_config)
    
    # Debug: Check for members and tools lines
    console.print("[dim]FINAL DEBUG: Checking members and tools formatting...[/dim]")
    team_config_lines = []
    member_count = 0
    tool_count = 0
    
    for i, line in enumerate(glue_content.splitlines()):
        if "members =" in line:
            console.print(f"[dim]LINE DEBUG [{i+1}]: {line}[/dim]")
            team_config_lines.append((i+1, line))
            if "[" in line and "]" in line and len(line.strip()) > 13:  # More than just "members = []"
                member_count += 1
        
        if "tools =" in line:
            console.print(f"[dim]LINE DEBUG [{i+1}]: {line}[/dim]")
            team_config_lines.append((i+1, line))
            if "[" in line and "]" in line and len(line.strip()) > 10:  # More than just "tools = []"
                tool_count += 1
    
    # If any issues were detected, show more details
    if any(not ('[' in line[1] and ']' in line[1]) for line in team_config_lines):
        console.print("[bold red]WARNING: Some member or tool lists may not be properly formatted![/bold red]")
        console.print("[dim]This may cause issues with the GLUE file parsing.[/dim]")
    else:
        console.print(f"[green]INFO: Found {member_count} member lists and {tool_count} tool lists properly formatted.[/green]")
    
    console.print("\n[bold green]✅ GLUE file generated![/bold green]")
    
    # 6.1 Display generated file
    console.print(Panel(
        glue_content,
        title=f"📄 {project_name}.glue",
        border_style="green",
        box=box.ROUNDED
    ))
    
    # 6.2 Final confirmation
    if not Confirm.ask("\n[bold]Does this look good?[/bold]", default=True):
        console.print(Panel(
            "You can edit the generated file after creation to make further adjustments.",
            title="📝 Note",
            border_style="yellow", 
            box=box.ROUNDED
        ))
    
    console.print("\n[bold green]✨ GLUE file created successfully![/bold green]")
    
    return glue_content

if __name__ == "__main__":
    main()
