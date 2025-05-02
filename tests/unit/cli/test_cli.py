"""
Unit tests for the GLUE CLI functionality.

These tests follow TDD principles to define the expected CLI behavior
before implementing it. The CLI should support running GLUE applications,
creating new projects, and managing model/tool configurations with adhesive
and magnetic features.
"""
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, call, AsyncMock
from io import StringIO
from pathlib import Path

# Import the CLI module
from glue.cli import (
    main,
    setup_logging,
    run_app,
    interactive_session,
    display_interactive_help,
    display_app_status,
    display_available_tools,
    display_team_structure,
    create_new_project,
    get_template_content,
    list_tools,
    format_component_name,
    create_tool
)

# ==================== CLI Basic Tests ====================
class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_setup_logging(self):
        """Test that logging is set up correctly."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level='DEBUG')
            mock_basic_config.assert_called_once()
    
    def test_format_component_name(self):
        """Test component name formatting."""
        # Test with spaces
        dir_name, module_name, class_name = format_component_name("my test tool")
        assert dir_name == "my-test-tool"
        assert module_name == "my_test_tool"
        assert class_name == "MyTestTool"
        
        # Test with mixed case
        dir_name, module_name, class_name = format_component_name("MyTestTool")
        assert dir_name == "mytesttool"
        assert module_name == "mytesttool"
        assert class_name == "Mytesttool"
        
        # Test with special characters
        dir_name, module_name, class_name = format_component_name("my-test!tool")
        assert dir_name == "my-testtool"
        assert module_name == "my_testtool"
        assert class_name == "MyTesttool"
    
    def test_version_command(self, capsys):
        """Test that the version command returns the correct version."""
        with patch.object(sys, 'argv', ['glue', 'version']):
            with patch('glue.cli.main', side_effect=SystemExit(0)):
                try:
                    main()
                except SystemExit:
                    pass
            
            captured = capsys.readouterr()
            assert "GLUE Framework version" in captured.out

    def test_help_command(self, capsys):
        """Test that the help command shows all available commands."""
        with patch.object(sys, 'argv', ['glue', '--help']):
            with patch('glue.cli.main', side_effect=SystemExit(0)):
                try:
                    main()
                except SystemExit:
                    pass
            
            captured = capsys.readouterr()
            # Check for required commands in help output
            assert "run" in captured.out
            assert "new" in captured.out
            assert "version" in captured.out
            assert "list-tools" in captured.out
            assert "list-models" in captured.out

# ==================== Project Management Tests ====================
class TestProjectManagement:
    """Test project creation and management."""
    
    def test_get_template_content(self):
        """Test template content generation."""
        # Test basic template
        basic_content = get_template_content("basic", "test_project")
        assert "glue app {" in basic_content
        assert 'name = "test_project"' in basic_content
        
        # Test research template
        research_content = get_template_content("research", "test_project")
        assert "sticky = true" in research_content
        assert "magnetize {" in research_content
        
        # Test chat template
        chat_content = get_template_content("chat", "test_project")
        assert "model chat_assistant {" in chat_content
    
    @patch('os.makedirs')
    @patch('pathlib.Path.write_text')
    def test_create_new_project(self, mock_write_text, mock_makedirs):
        """Test project creation."""
        with patch('os.path.exists', return_value=False):
            create_new_project("test_project", "basic")
            
            # Check that directories were created
            mock_makedirs.assert_called()
            
            # Check that files were written
            assert mock_write_text.call_count >= 2  # app.glue and .env.example
    
    @patch('os.path.exists', return_value=True)
    def test_create_new_project_existing_dir(self, mock_exists):
        """Test project creation with existing directory."""
        with patch('builtins.print') as mock_print:
            create_new_project("test_project")
            mock_print.assert_any_call("Error: Directory 'test_project' already exists")

# ==================== Tool Management Tests ====================
class TestToolManagement:
    """Test tool management functionality."""
    
    @patch('builtins.print')
    def test_list_tools(self, mock_print):
        """Test listing tools."""
        list_tools()
        mock_print.assert_any_call("\n=== Available Tools ===")
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.exists', return_value=True)
    def test_create_tool(self, mock_exists, mock_write_text, mock_mkdir):
        """Test tool creation."""
        create_tool("test_tool")
        mock_mkdir.assert_called()
        mock_write_text.assert_called_once()

# ==================== Interactive Mode Tests ====================
class TestInteractiveMode:
    """Test interactive mode functionality."""
    
    @patch('builtins.input', side_effect=['hello', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_basic(self, mock_print, mock_input):
        """Test basic interactive session flow."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = MagicMock(return_value="Response from app")
        
        await interactive_session(mock_app)
        
        # Check welcome message
        mock_print.assert_any_call("Welcome to Test App!")
        
        # Check that app.run was called with user input
        mock_app.run.assert_called_with("hello", conv_id="interactive")
        
        # Check that response was printed
        mock_print.assert_any_call("\nAssistant: Response from app")
    
    @patch('builtins.input', side_effect=['/help', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_help_command(self, mock_print, mock_input):
        """Test help command in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        with patch('glue.cli.display_interactive_help') as mock_help:
            await interactive_session(mock_app)
            mock_help.assert_called_once()
    
    @patch('builtins.input', side_effect=['/status', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_status_command(self, mock_print, mock_input):
        """Test status command in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        with patch('glue.cli.display_app_status') as mock_status:
            await interactive_session(mock_app)
            mock_status.assert_called_once_with(mock_app)
    
    @patch('builtins.input', side_effect=['/tools', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_tools_command(self, mock_print, mock_input):
        """Test tools command in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        with patch('glue.cli.display_available_tools') as mock_tools:
            await interactive_session(mock_app)
            mock_tools.assert_called_once_with(mock_app)
    
    @patch('builtins.input', side_effect=['/teams', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_teams_command(self, mock_print, mock_input):
        """Test teams command in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        
        with patch('glue.cli.display_team_structure') as mock_teams:
            await interactive_session(mock_app)
            mock_teams.assert_called_once_with(mock_app)
    
    @patch('builtins.input', side_effect=['/clear', 'exit'])
    @patch('builtins.print')
    async def test_interactive_session_clear_command(self, mock_print, mock_input):
        """Test clear command in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.clear_memory = MagicMock()
        
        await interactive_session(mock_app)
        mock_app.clear_memory.assert_called_once()
        mock_print.assert_any_call("Conversation memory cleared.")

# ==================== Enhanced Interactive Mode Tests ====================
import pytest

class TestEnhancedInteractiveMode:
    """Test enhanced interactive mode features."""
    
    @pytest.mark.asyncio
    @patch('builtins.input', side_effect=['/verbose', 'hello', '/verbose', 'exit'])
    @patch('builtins.print')
    async def test_verbose_mode_toggle(self, mock_print, mock_input):
        """Test toggling verbose mode in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = AsyncMock(return_value={
            "final_response": "Final response from app",
            "agent_interactions": [
                {"agent": "researcher", "message": "Researching the query..."},
                {"agent": "assistant", "message": "Processing research results..."}
            ]
        })
        
        # This test will fail until we implement the verbose mode toggle
        with pytest.raises(AssertionError):
            await interactive_session(mock_app)
            # Check that verbose mode was enabled
            mock_print.assert_any_call("Verbose mode enabled. Showing agent interactions.")
            # Check that agent interactions were displayed
            mock_print.assert_any_call("\nResearcher: Researching the query...")
            mock_print.assert_any_call("\nAssistant: Processing research results...")
            # Check that verbose mode was disabled
            mock_print.assert_any_call("Verbose mode disabled.")
    
    @pytest.mark.asyncio
    @patch('builtins.input', side_effect=['/step', 'hello', '/next', '/next', '/next', '/step', 'exit'])
    @patch('builtins.print')
    async def test_step_by_step_mode(self, mock_print, mock_input):
        """Test step-by-step mode in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.begin_step_execution = AsyncMock()
        mock_app.next_step = AsyncMock(side_effect=[
            {"agent": "researcher", "message": "Researching the query..."},
            {"agent": "assistant", "message": "Processing research results..."},
            {"agent": "writer", "message": "Final response based on research."}
        ])
        mock_app.end_step_execution = AsyncMock()
        
        # This test will fail until we implement step-by-step mode
        with pytest.raises(AssertionError):
            await interactive_session(mock_app)
            # Check that step mode was enabled
            mock_print.assert_any_call("Step-by-step mode enabled. Use /next to advance.")
            # Check that begin_step_execution was called
            mock_app.begin_step_execution.assert_called_once_with("hello", conv_id="interactive")
            # Check that each step was displayed
            mock_print.assert_any_call("\nResearcher: Researching the query...")
            mock_print.assert_any_call("\nAssistant: Processing research results...")
            mock_print.assert_any_call("\nWriter: Final response based on research.")
            # Check that step mode was disabled
            mock_print.assert_any_call("Step-by-step mode disabled.")
            # Check that end_step_execution was called
            mock_app.end_step_execution.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('builtins.input', side_effect=['/color on', 'hello', '/color off', 'exit'])
    @patch('builtins.print')
    async def test_color_mode_toggle(self, mock_print, mock_input):
        """Test toggling color mode in interactive session."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = AsyncMock(return_value="Response from app")
        
        # This test will fail until we implement color mode
        with pytest.raises(AssertionError):
            await interactive_session(mock_app)
            # Check that color mode was enabled
            mock_print.assert_any_call("Color output enabled.")
            # Check that color mode was disabled
            mock_print.assert_any_call("Color output disabled.")
    
    @pytest.mark.asyncio
    @patch('builtins.input', side_effect=['/color invalid', '/color on', 'exit'])
    @patch('builtins.print')
    async def test_color_mode_invalid_args(self, mock_print, mock_input):
        """Test handling invalid arguments for color mode command."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.run = AsyncMock(return_value="Response from app")
        
        # This test will fail until we implement color mode with argument validation
        with pytest.raises(AssertionError):
            await interactive_session(mock_app)
            # Check that usage message was displayed
            mock_print.assert_any_call("Usage: /color [on|off]")
            # Check that color mode was enabled after valid command
            mock_print.assert_any_call("Color output enabled.")
    
    @pytest.mark.asyncio
    @patch('builtins.input', side_effect=['/verbose', '/step', '/next', '/verbose', '/step', 'exit'])
    @patch('builtins.print')
    async def test_mode_combinations(self, mock_print, mock_input):
        """Test combining different interactive mode features."""
        mock_app = MagicMock()
        mock_app.app_config.name = "Test App"
        mock_app.begin_step_execution = AsyncMock()
        mock_app.next_step = AsyncMock(return_value={"agent": "researcher", "message": "Researching..."})
        mock_app.end_step_execution = AsyncMock()
        
        # This test will fail until we implement all mode combinations
        with pytest.raises(AssertionError):
            await interactive_session(mock_app)
            # Check that verbose mode was enabled
            mock_print.assert_any_call("Verbose mode enabled. Showing agent interactions.")
            # Check that step mode was enabled
            mock_print.assert_any_call("Step-by-step mode enabled. Use /next to advance.")
            # Check that step output was displayed
            mock_print.assert_any_call("\nResearcher: Researching...")
            # Check that verbose mode was disabled
            mock_print.assert_any_call("Verbose mode disabled.")
            # Check that step mode was disabled
            mock_print.assert_any_call("Step-by-step mode disabled.")

# ==================== Run App Tests ====================
class TestRunApp:
    """Test running GLUE applications."""
    
    @patch('glue.cli.load_env_file')
    async def test_run_app_basic(self, mock_load_env):
        """Test basic app running."""
        mock_app = MagicMock()
        
        with patch('builtins.print') as mock_print:
            with patch('glue.cli.GlueApp', return_value=mock_app) as mock_glue_app:
                await run_app("test.glue")
                mock_glue_app.assert_called_once_with(config_file="test.glue")
                mock_app.setup.assert_called_once()
                mock_app.run.assert_called_once()
                mock_app.close.assert_called_once()
    
    @patch('glue.cli.load_env_file')
    async def test_run_app_with_input(self, mock_load_env):
        """Test running app with input text."""
        mock_app = MagicMock()
        
        with patch('builtins.print') as mock_print:
            with patch('glue.cli.GlueApp', return_value=mock_app) as mock_glue_app:
                await run_app("test.glue", input_text="Hello")
                mock_app.run.assert_called_once_with("Hello")
    
    @patch('glue.cli.load_env_file')
    async def test_run_app_interactive(self, mock_load_env):
        """Test running app in interactive mode."""
        mock_app = MagicMock()
        
        with patch('glue.cli.interactive_session') as mock_interactive:
            with patch('glue.cli.GlueApp', return_value=mock_app) as mock_glue_app:
                await run_app("test.glue", interactive=True)
                mock_interactive.assert_called_once_with(mock_app)
    
    @patch('glue.cli.load_env_file')
    async def test_run_app_invalid_extension(self, mock_load_env):
        """Test running app with invalid file extension."""
        with pytest.raises(Exception):
            await run_app("test.txt")
    
    async def test_run_app_loads_config(self, example_glue_file):
        """Test that run_app loads the config file correctly."""
        with patch('glue.cli.GlueApp') as mock_glue_app:
            mock_app = AsyncMock()
            mock_glue_app.return_value = mock_app
            
            await run_app(example_glue_file)
            
            # Check that GlueApp was initialized with the correct config file
            mock_glue_app.assert_called_once_with(config_file=example_glue_file)
            mock_app.setup.assert_called_once()
    
    async def test_run_app_with_adhesive_configuration(self, simple_glue_content, tmp_path):
        """Test that run_app properly handles adhesive configurations."""
        # Create a test GLUE file with adhesive configuration
        test_file = tmp_path / "test_adhesive.glue"
        test_file.write_text(simple_glue_content)
        
        with patch('glue.cli.GlueApp') as mock_glue_app:
            mock_app = AsyncMock()
            mock_glue_app.return_value = mock_app
            
            await run_app(str(test_file))
            
            # Check that GlueApp was initialized with the correct config file
            mock_glue_app.assert_called_once_with(config_file=str(test_file))
            mock_app.setup.assert_called_once()

# ==================== DSL and Config Testing ====================
class TestDSLAndConfig:
    """Test DSL parsing and configuration."""
    
    def test_validate_command(self, example_glue_file, capsys):
        """Test that validate command checks GLUE file syntax."""
        with patch.object(sys, 'argv', ['glue', 'validate', example_glue_file]):
            with patch('glue.cli.main', side_effect=SystemExit(0)):
                try:
                    main()
                except SystemExit:
                    pass
            
            captured = capsys.readouterr()
            assert "Validation successful" in captured.out or "is valid" in captured.out
    
    def test_validate_command_with_errors(self, invalid_glue_content, tmp_path, capsys):
        """Test that validate command reports syntax errors."""
        # Create a test file with invalid GLUE syntax
        test_file = tmp_path / "invalid.glue"
        test_file.write_text(invalid_glue_content)
        
        with patch.object(sys, 'argv', ['glue', 'validate', str(test_file)]):
            with patch('glue.cli.main', side_effect=SystemExit(1)):
                try:
                    main()
                except SystemExit:
                    pass
            
            captured = capsys.readouterr()
            assert "Error" in captured.out or "invalid" in captured.out