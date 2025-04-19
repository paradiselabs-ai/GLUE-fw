"""
UI utility functions that can be used throughout the GLUE framework.
"""
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

# Global reference to CLI configuration
_CLI_CONFIG = None

def set_cli_config(config):
    """Set the CLI configuration for UI functions.
    
    Args:
        config: CLI configuration dictionary
    """
    global _CLI_CONFIG
    _CLI_CONFIG = config

def display_warning(console: Console, message: str, config: dict = None) -> None:
    """Display a warning message with enhanced styling.
    
    Args:
        console: Rich console to use for display
        message: The warning message to display
        config: Optional configuration dictionary with display settings
    """
    # Use global CLI_CONFIG if available and no config provided
    if config is None and _CLI_CONFIG is not None:
        config = _CLI_CONFIG
    
    # Fallback if no config is available
    if config is None:
        config = {
            "display": {
                "show_emoji": True,
                "panel_box": None
            }
        }
    
    box_style = config.get("display", {}).get("panel_box", None)
    show_emoji = config.get("display", {}).get("show_emoji", True)
    
    # Check for specific warning patterns that need special treatment
    if "does not support tool use on OpenRouter" in message:
        # Enhanced styling for OpenRouter warnings
        warning_panel = Panel(
            f"[bold yellow]{message}[/bold yellow]",
            title="⚠️ OpenRouter Model Limitation" if show_emoji else "OpenRouter Model Limitation",
            border_style="yellow",
            box=box_style,
            width=100
        )
    else:
        # Standard warning styling
        warning_panel = Panel(
            message,
            title="⚠️ Warning" if show_emoji else "Warning",
            border_style="yellow", 
            box=box_style,
            width=100
        )
    
    # Add some spacing and center the panel
    console.print()  # Empty line before
    console.print(Align.center(warning_panel))
    console.print()  # Empty line after