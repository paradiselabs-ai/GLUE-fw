"""
Unit tests for the enhanced interactive mode features of the GLUE CLI.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call
from io import StringIO
from pathlib import Path
import asyncio

# Import the CLI module
from glue.cli import interactive_session

# ==================== Verbose Mode Tests ====================
class TestVerboseMode:
    """Test verbose mode functionality."""
    
    @patch('builtins.input', side_effect=['/verbose', 'hello', 'exit'])
    @patch('builtins.print')
    async def test_verbose_mode_enable(self, mock_print, mock_input):
        """Test enabling verbose mode."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        # This test expects the verbose flag to be set when /verbose is entered
        await interactive_session(mock_app)
        
        # Check that verbose mode was enabled
        mock_print.assert_any_call("Verbose mode enabled. Showing agent interactions.")
    
    @patch('builtins.input', side_effect=['/verbose', '/verbose', 'hello', 'exit'])
    @patch('builtins.print')
    async def test_verbose_mode_toggle(self, mock_print, mock_input):
        """Test toggling verbose mode on and off."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        # This test expects verbose mode to be toggled off when /verbose is entered again
        await interactive_session(mock_app)
        
        # Check that verbose mode was enabled and then disabled
        mock_print.assert_any_call("Verbose mode enabled. Showing agent interactions.")
        mock_print.assert_any_call("Verbose mode disabled.")
    
    @patch('builtins.input', side_effect=['/verbose', 'hello', 'exit'])
    @patch('builtins.print')
    async def test_verbose_mode_output(self, mock_print, mock_input):
        """Test that verbose mode shows agent interactions."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        # Mock the run method to return agent interactions when verbose is enabled
        mock_app.run = MagicMock(return_value={
            "final_response": "Final response from app",
            "agent_interactions": [
                {"agent": "researcher", "message": "Researching the query..."},
                {"agent": "assistant", "message": "Processing research results..."},
                {"agent": "writer", "message": "Formatting the final response..."}
            ]
        })
        
        await interactive_session(mock_app)
        
        # Check that agent interactions were printed in verbose mode
        mock_print.assert_any_call("\nResearcher: Researching the query...")
        mock_print.assert_any_call("\nAssistant: Processing research results...")
        mock_print.assert_any_call("\nWriter: Formatting the final response...")
        mock_print.assert_any_call("\nAssistant: Final response from app")

# ==================== Step-by-Step Mode Tests ====================
class TestStepByStepMode:
    """Test step-by-step mode functionality."""
    
    @patch('builtins.input', side_effect=['/step', 'hello', '/next', '/next', '/next', 'exit'])
    @patch('builtins.print')
    async def test_step_mode_enable(self, mock_print, mock_input):
        """Test enabling step-by-step mode."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        # Mock the step-by-step execution
        mock_app.begin_step_execution = MagicMock()
        mock_app.next_step = MagicMock(side_effect=[
            {"agent": "researcher", "message": "Researching the query..."},
            {"agent": "assistant", "message": "Processing research results..."},
            {"agent": "writer", "message": "Final response: Hello!"}
        ])
        
        await interactive_session(mock_app)
        
        # Check that step mode was enabled
        mock_print.assert_any_call("Step-by-step mode enabled. Use /next to advance.")
        
        # Check that begin_step_execution was called with the input
        mock_app.begin_step_execution.assert_called_with("hello", conv_id="interactive")
        
        # Check that next_step was called for each /next command
        assert mock_app.next_step.call_count == 3
        
        # Check that each step's output was printed
        mock_print.assert_any_call("\nResearcher: Researching the query...")
        mock_print.assert_any_call("\nAssistant: Processing research results...")
        mock_print.assert_any_call("\nWriter: Final response: Hello!")
    
    @patch('builtins.input', side_effect=['/step', 'hello', '/next', '/step', 'exit'])
    @patch('builtins.print')
    async def test_step_mode_disable(self, mock_print, mock_input):
        """Test disabling step-by-step mode."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.begin_step_execution = MagicMock()
        mock_app.next_step = MagicMock(return_value={"agent": "researcher", "message": "Researching..."})
        mock_app.end_step_execution = MagicMock()
        
        await interactive_session(mock_app)
        
        # Check that step mode was enabled and then disabled
        mock_print.assert_any_call("Step-by-step mode enabled. Use /next to advance.")
        mock_print.assert_any_call("Step-by-step mode disabled.")
        
        # Check that end_step_execution was called
        mock_app.end_step_execution.assert_called_once()

# ==================== Color Output Tests ====================
class TestColorOutput:
    """Test color output functionality."""
    
    @patch('builtins.input', side_effect=['/color on', 'hello', 'exit'])
    @patch('builtins.print')
    async def test_color_mode_enable(self, mock_print, mock_input):
        """Test enabling color output."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        await interactive_session(mock_app)
        
        # Check that color mode was enabled
        mock_print.assert_any_call("Color output enabled.")
    
    @patch('builtins.input', side_effect=['/color off', 'hello', 'exit'])
    @patch('builtins.print')
    async def test_color_mode_disable(self, mock_print, mock_input):
        """Test disabling color output."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        await interactive_session(mock_app)
        
        # Check that color mode was disabled
        mock_print.assert_any_call("Color output disabled.")
    
    @patch('builtins.input', side_effect=['/color on', 'hello', 'exit'])
    @patch('builtins.print')
    @patch('glue.cli.colorize_agent_output')
    async def test_color_output_formatting(self, mock_colorize, mock_print, mock_input):
        """Test that output is colorized when color mode is enabled."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        # Mock the colorize function to return colored text
        mock_colorize.return_value = "\033[94mResponse from app\033[0m"  # Blue text
        
        await interactive_session(mock_app)
        
        # Check that colorize_agent_output was called with the response
        mock_colorize.assert_called_with("assistant", "Response from app")
        
        # Check that the colorized output was printed
        mock_print.assert_any_call("\nAssistant: \033[94mResponse from app\033[0m")

# ==================== Combined Features Tests ====================
class TestCombinedFeatures:
    """Test combinations of interactive mode features."""
    
    @patch('builtins.input', side_effect=['/verbose', '/color on', 'hello', 'exit'])
    @patch('builtins.print')
    @patch('glue.cli.colorize_agent_output')
    async def test_verbose_with_color(self, mock_colorize, mock_print, mock_input):
        """Test verbose mode with color output."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        # Mock the run method to return agent interactions
        mock_app.run = MagicMock(return_value={
            "final_response": "Final response from app",
            "agent_interactions": [
                {"agent": "researcher", "message": "Researching the query..."},
                {"agent": "assistant", "message": "Processing research results..."}
            ]
        })
        
        # Mock the colorize function to return colored text
        mock_colorize.side_effect = lambda agent, text: {
            "researcher": "\033[92mResearching the query...\033[0m",  # Green
            "assistant": "\033[94mProcessing research results...\033[0m",  # Blue
            "final_response": "\033[94mFinal response from app\033[0m"  # Blue
        }.get(agent, text)
        
        await interactive_session(mock_app)
        
        # Check that both verbose mode and color mode were enabled
        mock_print.assert_any_call("Verbose mode enabled. Showing agent interactions.")
        mock_print.assert_any_call("Color output enabled.")
        
        # Check that colorized agent interactions were printed
        mock_print.assert_any_call("\nResearcher: \033[92mResearching the query...\033[0m")
        mock_print.assert_any_call("\nAssistant: \033[94mProcessing research results...\033[0m")
        mock_print.assert_any_call("\nAssistant: \033[94mFinal response from app\033[0m")
    
    @patch('builtins.input', side_effect=['/step', '/color on', 'hello', '/next', '/next', 'exit'])
    @patch('builtins.print')
    @patch('glue.cli.colorize_agent_output')
    async def test_step_with_color(self, mock_colorize, mock_print, mock_input):
        """Test step-by-step mode with color output."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        # Mock the step-by-step execution
        mock_app.begin_step_execution = MagicMock()
        mock_app.next_step = MagicMock(side_effect=[
            {"agent": "researcher", "message": "Researching the query..."},
            {"agent": "assistant", "message": "Final response: Hello!"}
        ])
        
        # Mock the colorize function to return colored text
        mock_colorize.side_effect = lambda agent, text: {
            "researcher": "\033[92mResearching the query...\033[0m",  # Green
            "assistant": "\033[94mFinal response: Hello!\033[0m"  # Blue
        }.get(agent, text)
        
        await interactive_session(mock_app)
        
        # Check that both step mode and color mode were enabled
        mock_print.assert_any_call("Step-by-step mode enabled. Use /next to advance.")
        mock_print.assert_any_call("Color output enabled.")
        
        # Check that colorized step outputs were printed
        mock_print.assert_any_call("\nResearcher: \033[92mResearching the query...\033[0m")
        mock_print.assert_any_call("\nAssistant: \033[94mFinal response: Hello!\033[0m")
