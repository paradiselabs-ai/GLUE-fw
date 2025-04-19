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
    "model.researcher": "green",
    "model.assistant": "cyan",
    "model.writer": "magenta",
    "model.editor": "yellow",
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
        "model": {
            "researcher": "model.researcher",
            "assistant": "model.assistant",
            "writer": "model.writer",
            "editor": "model.editor",
            "default": "white"
        }
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
                logger.info(f"--- Agent Turn {turn_count} ---")
                
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
                                    
                                    logger.info(f"Tool {tool_name} (ID: {tool_call_id}) executed. Result: {tool_result}")
                                    
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
                
                if record.levelno >= logging.WARNING:
                    # Get color based on level
                    if record.levelno >= logging.ERROR:
                        color = "red"
                        prefix = "✗ Error: "
                    elif record.levelno >= logging.WARNING:
                        color = "yellow"
                        prefix = "⚠ Warning: "
                    else:
                        color = "blue"
                        prefix = "ℹ Info: "
                    
                    # Format the message
                    message = record.getMessage()
                    
                    # Create a styled Text object like the welcome message
                    text = Text()
                    text.append(prefix, style=f"bold {color}")
                    text.append(message, style="dim")
                    
                    # We need to convert to string for the logging system
                    console = Console(file=io.StringIO())
                    console.print(text)
                    return console.file.getvalue().strip()
                
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
    table.add_column(f"{model_emoji}Lead", style="model.researcher")
    table.add_column("Members", style="model.assistant")
    table.add_column(f"{tool_emoji}Tools", style="tool")
    
    for team_name, team in teams.items():
        lead_name = team.lead.name if hasattr(team, "lead") and team.lead else "None"
        members = ", ".join([m.name for m in team.members]) if hasattr(team, "members") else "None"
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
        
        # Add lead model
        if hasattr(team, "lead") and team.lead:
            style = CLI_CONFIG["theme"]["model"].get(team.lead.name, CLI_CONFIG["theme"]["model"]["default"])
            team_node.add(f"[{style}]{model_emoji} {team.lead.name} (Lead)[/{style}]")
        
        # Add member models
        if hasattr(team, "members"):
            for member in team.members:
                style = CLI_CONFIG["theme"]["model"].get(member.name, CLI_CONFIG["theme"]["model"]["default"])
                team_node.add(f"[{style}]{model_emoji} {member.name}[/{style}]")
    
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
        # Count members if they exist
        if hasattr(team, "members") and team.members is not None:
            total_models += len(team.members)
    
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
    """Animate text as if it's being typed.
    
    Args:
        console: Rich console to use for display
        text: Text to animate
        speed: Speed of typing in seconds per character
    """
    # Always print directly without animation for better UX
    console.print(text)
    return

def display_warning(console: Console, message: str) -> None:
    """Display a warning in a styled panel.
    
    Args:
        console: Rich console to use for display
        message: Warning message to display
    """
    from rich.panel import Panel
    from rich.text import Text
    
    warning_text = Text()
    warning_text.append("⚠ ", style="bold yellow")
    warning_text.append(message, style="yellow")
    
    panel = Panel(
        warning_text,
        title="Warning",
        border_style="yellow dim",
        box=CLI_CONFIG["display"]["panel_box"],
        padding=(1, 2),
        width=100
    )
    
    # Print an empty line before the warning for better spacing
    console.print()
    console.print(panel)
    console.print()

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

if __name__ == "__main__":
    main()
