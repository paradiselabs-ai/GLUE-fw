"""
Utility functions for the GLUE CLI, particularly for the enhanced interactive mode.
"""

import re
from typing import Tuple, List, Optional, Dict, Any

# ==================== Color Constants ====================
# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

# Agent color mappings
AGENT_COLORS = {
    "researcher": COLORS["bright_green"],
    "assistant": COLORS["bright_blue"],
    "writer": COLORS["bright_cyan"],
    "coder": COLORS["bright_magenta"],
    "planner": COLORS["bright_yellow"],
    "system": COLORS["bright_yellow"],
    "default": COLORS["bright_white"],
}


# ==================== Command Parsing ====================
def parse_interactive_command(input_text: str) -> Tuple[Optional[str], List[str]]:
    """
    Parse an interactive command from user input.

    Args:
        input_text: The user input text

    Returns:
        A tuple of (command, args) where command is the command name (without the /)
        and args is a list of command arguments. If the input is not a command,
        command will be None and args will be an empty list.
    """
    if not input_text.startswith("/"):
        return None, []

    # Split the command and arguments
    parts = input_text[1:].split(maxsplit=1)
    command = parts[0].lower() if parts else ""

    # Parse arguments if present
    args = []
    if len(parts) > 1:
        # Handle quoted arguments
        arg_text = parts[1]
        quoted_args = re.findall(r'"([^"]*)"', arg_text)

        # Remove quoted sections and split remaining by whitespace
        for quoted in quoted_args:
            arg_text = arg_text.replace(f'"{quoted}"', "", 1)

        # Get non-quoted args
        non_quoted_args = [a for a in arg_text.split() if a]

        # Combine all args in order
        args = non_quoted_args + quoted_args

    return command, args


# ==================== Output Formatting ====================
def colorize_agent_output(agent: str, text: str, color_enabled: bool = True) -> str:
    """
    Colorize agent output based on agent type.

    Args:
        agent: The agent name or type
        text: The text to colorize
        color_enabled: Whether color output is enabled

    Returns:
        The colorized text if color is enabled, otherwise the original text
    """
    if not color_enabled:
        return text

    agent_lower = agent.lower()
    color = AGENT_COLORS.get(agent_lower, AGENT_COLORS["default"])
    return f"{color}{text}{COLORS['reset']}"


def format_agent_message(agent: str, message: str, color_enabled: bool = False) -> str:
    """
    Format an agent message for display.

    Args:
        agent: The agent name or type
        message: The message content
        color_enabled: Whether color output is enabled

    Returns:
        The formatted message string
    """
    # Format system messages differently
    if agent.lower() == "system":
        prefix = "\n[System]"
    else:
        # Capitalize the first letter of the agent name
        agent_display = agent[0].upper() + agent[1:] if agent else "Unknown"
        prefix = f"\n{agent_display}:"

    # Apply color if enabled
    if color_enabled:
        message = colorize_agent_output(agent, message)

    return f"{prefix} {message}"


def format_agent_interactions(
    interactions: List[Dict[str, Any]], color_enabled: bool = False
) -> List[str]:
    """
    Format a list of agent interactions for display.

    Args:
        interactions: List of agent interaction dictionaries with 'agent' and 'message' keys
        color_enabled: Whether color output is enabled

    Returns:
        List of formatted message strings
    """
    return [
        format_agent_message(
            interaction["agent"], interaction["message"], color_enabled
        )
        for interaction in interactions
    ]


# ==================== Help Text ====================
def get_interactive_help_text() -> str:
    """
    Get the help text for interactive mode.

    Returns:
        The help text as a string
    """
    return """
=== Interactive Mode Commands ===
/help       - Show this help message
/status     - Show the current app status
/tools      - List available tools
/teams      - Show team structure
/clear      - Clear conversation memory
/verbose    - Toggle verbose mode (show agent interactions)
/step       - Toggle step-by-step execution mode
/next       - Advance to the next step in step-by-step mode
/color on   - Enable colored output
/color off  - Disable colored output
/exit, /quit - Exit interactive mode
"""
