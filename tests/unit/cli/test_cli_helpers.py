"""
Unit tests for the GLUE CLI helper functions.

These tests follow TDD principles to define the expected behavior of the CLI helper
functions before implementing them. The helpers provide functionality for parsing
commands, formatting output, and managing color in the CLI.
"""
import pytest
from unittest.mock import patch
import re

from glue.cliHelpers import (
    parse_interactive_command,
    colorize_agent_output,
    format_agent_message,
    format_agent_interactions,
    get_interactive_help_text,
    COLORS,
    AGENT_COLORS
)

# ==================== Command Parsing Tests ====================
class TestCommandParsing:
    """Test command parsing functionality."""
    
    def test_parse_non_command(self):
        """Test parsing a regular input (not a command)."""
        command, args = parse_interactive_command("hello world")
        assert command is None
        assert args == []
    
    def test_parse_simple_command(self):
        """Test parsing a simple command with no arguments."""
        command, args = parse_interactive_command("/help")
        assert command == "help"
        assert args == []
    
    def test_parse_command_with_args(self):
        """Test parsing a command with arguments."""
        command, args = parse_interactive_command("/color on")
        assert command == "color"
        assert args == ["on"]
    
    def test_parse_command_with_multiple_args(self):
        """Test parsing a command with multiple arguments."""
        command, args = parse_interactive_command("/search python framework")
        assert command == "search"
        assert args == ["python", "framework"]
    
    def test_parse_command_with_quoted_args(self):
        """Test parsing a command with quoted arguments."""
        command, args = parse_interactive_command('/search "python framework" advanced')
        assert command == "search"
        assert args == ["advanced", "python framework"]

# ==================== Output Formatting Tests ====================
class TestOutputFormatting:
    """Test output formatting functionality."""
    
    def test_colorize_agent_output_enabled(self):
        """Test colorizing agent output with color enabled."""
        colored = colorize_agent_output("researcher", "Hello", True)
        assert COLORS["reset"] in colored
        assert AGENT_COLORS["researcher"] in colored
        assert "Hello" in colored
    
    def test_colorize_agent_output_disabled(self):
        """Test colorizing agent output with color disabled."""
        colored = colorize_agent_output("researcher", "Hello", False)
        assert colored == "Hello"
    
    def test_colorize_unknown_agent(self):
        """Test colorizing output for an unknown agent type."""
        colored = colorize_agent_output("unknown_agent", "Hello", True)
        assert COLORS["reset"] in colored
        assert AGENT_COLORS["default"] in colored
        assert "Hello" in colored
    
    def test_format_agent_message_no_color(self):
        """Test formatting an agent message without color."""
        formatted = format_agent_message("researcher", "Hello", False)
        assert formatted == "\nResearcher: Hello"
    
    def test_format_agent_message_with_color(self):
        """Test formatting an agent message with color."""
        formatted = format_agent_message("researcher", "Hello", True)
        assert "\nResearcher:" in formatted
        assert COLORS["reset"] in formatted
        assert AGENT_COLORS["researcher"] in formatted
    
    def test_format_system_message(self):
        """Test formatting a system message."""
        formatted = format_agent_message("system", "System notification", False)
        assert formatted == "\n[System] System notification"
    
    def test_format_agent_interactions(self):
        """Test formatting a list of agent interactions."""
        interactions = [
            {"agent": "researcher", "message": "Researching..."},
            {"agent": "assistant", "message": "Processing..."}
        ]
        formatted = format_agent_interactions(interactions, False)
        assert len(formatted) == 2
        assert "\nResearcher: Researching..." in formatted
        assert "\nAssistant: Processing..." in formatted

# ==================== Help Text Tests ====================
class TestHelpText:
    """Test help text functionality."""
    
    def test_get_interactive_help_text(self):
        """Test getting the interactive help text."""
        help_text = get_interactive_help_text()
        assert "Interactive Mode Commands" in help_text
        assert "/help" in help_text
        assert "/verbose" in help_text
        assert "/step" in help_text
        assert "/color" in help_text
