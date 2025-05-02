"""
Unit tests for the interactive mode utility functions of the GLUE CLI.
"""
import pytest
from unittest.mock import patch, MagicMock
import re

# Import the CLI utility functions
# These will be implemented to support the enhanced interactive mode
from glue.cli import colorize_agent_output, format_agent_message, parse_interactive_command

# ==================== Color Utility Tests ====================
class TestColorUtilities:
    """Test color-related utility functions."""
    
    def test_colorize_agent_output(self):
        """Test that agent output is colorized correctly."""
        # Test researcher (green)
        colored_text = colorize_agent_output("researcher", "Test message")
        assert "\033[92m" in colored_text  # Green color code
        assert "Test message" in colored_text
        assert colored_text.endswith("\033[0m")  # Reset color code
        
        # Test assistant (blue)
        colored_text = colorize_agent_output("assistant", "Test message")
        assert "\033[94m" in colored_text  # Blue color code
        assert "Test message" in colored_text
        
        # Test writer (cyan)
        colored_text = colorize_agent_output("writer", "Test message")
        assert "\033[96m" in colored_text  # Cyan color code
        assert "Test message" in colored_text
        
        # Test unknown agent (default color)
        colored_text = colorize_agent_output("unknown", "Test message")
        assert "\033[97m" in colored_text  # White color code
        assert "Test message" in colored_text
    
    def test_colorize_agent_output_with_color_disabled(self):
        """Test that colorization is skipped when color is disabled."""
        # When color is disabled, the function should return the original text
        colored_text = colorize_agent_output("researcher", "Test message", color_enabled=False)
        assert "\033[" not in colored_text  # No color codes
        assert colored_text == "Test message"

# ==================== Message Formatting Tests ====================
class TestMessageFormatting:
    """Test message formatting functions."""
    
    def test_format_agent_message(self):
        """Test that agent messages are formatted correctly."""
        # Test basic formatting
        formatted = format_agent_message("researcher", "Test message")
        assert formatted == "\nResearcher: Test message"
        
        # Test with color enabled
        with patch('glue.cli.colorize_agent_output') as mock_colorize:
            mock_colorize.return_value = "\033[92mTest message\033[0m"
            formatted = format_agent_message("researcher", "Test message", color_enabled=True)
            assert formatted == "\nResearcher: \033[92mTest message\033[0m"
            mock_colorize.assert_called_with("researcher", "Test message")
    
    def test_format_agent_message_with_system_message(self):
        """Test formatting of system messages."""
        # System messages should be formatted differently
        formatted = format_agent_message("system", "Test system message")
        assert formatted == "\n[System] Test system message"
        
        # Test with color enabled
        with patch('glue.cli.colorize_agent_output') as mock_colorize:
            mock_colorize.return_value = "\033[93mTest system message\033[0m"  # Yellow
            formatted = format_agent_message("system", "Test system message", color_enabled=True)
            assert formatted == "\n[System] \033[93mTest system message\033[0m"

# ==================== Command Parsing Tests ====================
class TestCommandParsing:
    """Test command parsing functions."""
    
    def test_parse_interactive_command(self):
        """Test parsing of interactive commands."""
        # Test help command
        command, args = parse_interactive_command("/help")
        assert command == "help"
        assert args == []
        
        # Test verbose command
        command, args = parse_interactive_command("/verbose")
        assert command == "verbose"
        assert args == []
        
        # Test color command with args
        command, args = parse_interactive_command("/color on")
        assert command == "color"
        assert args == ["on"]
        
        # Test step command
        command, args = parse_interactive_command("/step")
        assert command == "step"
        assert args == []
        
        # Test next command
        command, args = parse_interactive_command("/next")
        assert command == "next"
        assert args == []
        
        # Test non-command
        command, args = parse_interactive_command("This is not a command")
        assert command is None
        assert args == []
    
    def test_parse_interactive_command_with_complex_args(self):
        """Test parsing commands with complex arguments."""
        # Test command with multiple args
        command, args = parse_interactive_command("/set temperature 0.7")
        assert command == "set"
        assert args == ["temperature", "0.7"]
        
        # Test command with quoted args
        command, args = parse_interactive_command('/set role "Research Assistant"')
        assert command == "set"
        assert args == ["role", "Research Assistant"]
