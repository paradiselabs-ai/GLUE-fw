"""
Unit tests for the GLUE CLI.

These tests follow TDD principles to define the expected CLI behavior
before implementing it. The CLI should support running GLUE applications,
creating new projects, and managing model/tool configurations with adhesive
and magnetic features.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Import CLI module
from glue.cli import main, run_app, create_new_project

# ==================== CLI Basic Functions Tests ====================
class TestCLIBasics:
    """Test basic CLI functions."""

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

# ==================== New Project Creation Tests ====================
class TestProjectCreation:
    """Test project creation functionality."""

    def test_new_project_creates_directories(self, tmp_path):
        """Test that new project command creates the correct directory structure."""
        project_name = "test_project"
        project_path = tmp_path / project_name
        
        with patch('os.getcwd', return_value=str(tmp_path)):
            create_new_project(project_name)
        
        # Check that the project directory was created
        assert project_path.exists()
        
        # Check for required files
        assert (project_path / "app.glue").exists()
        assert (project_path / ".env.example").exists()

    def test_new_project_creates_valid_glue_file(self, tmp_path):
        """Test that the created glue file is valid."""
        project_name = "test_project"
        project_path = tmp_path / project_name
        
        with patch('os.getcwd', return_value=str(tmp_path)):
            create_new_project(project_name)
        
        # Read the created file
        with open(project_path / "app.glue", "r") as f:
            content = f.read()
        
        # Check for required sections
        assert f'name = "{project_name}"' in content
        assert "model " in content
        assert "tool " in content
        assert "magnetize " in content
        assert "apply glue" in content
        
        # Check for adhesive types
        assert "adhesives = [" in content
        assert "glue" in content
        assert "velcro" in content

    def test_new_project_existing_directory(self, tmp_path, capsys):
        """Test creating a project in an existing directory."""
        project_name = "existing_project"
        project_path = tmp_path / project_name
        project_path.mkdir()
        
        with patch('os.getcwd', return_value=str(tmp_path)):
            create_new_project(project_name)
        
        # Check that an error message was displayed
        captured = capsys.readouterr()
        assert f"Error: Directory '{project_name}' already exists" in captured.out

# ==================== Run App Tests ====================
class TestRunApp:
    """Test app running functionality."""

    def test_run_app_loads_config(self, example_glue_file):
        """Test that run_app loads the config file correctly."""
        with patch('glue.core.GlueApp') as MockGlueApp:
            mock_app = MockGlueApp.return_value
            mock_app.setup = MagicMock()
            mock_app.run = AsyncMock()
            mock_app.close = AsyncMock()
            
            # Run the app with a test input
            asyncio.run(run_app(str(example_glue_file), "Test input"))
            
            # Check that GlueApp was initialized with the correct config file
            MockGlueApp.assert_called_once_with(config_file=str(example_glue_file))
            mock_app.setup.assert_called_once()
            mock_app.run.assert_called_once()
            mock_app.close.assert_called_once()

    def test_run_app_interactive_mode(self, example_glue_file):
        """Test that interactive mode works correctly."""
        with patch('glue.cli.interactive_session', new_callable=AsyncMock) as mock_interactive:
            with patch('glue.core.GlueApp') as MockGlueApp:
                mock_app = MockGlueApp.return_value
                mock_app.setup = MagicMock()
                mock_app.close = AsyncMock()
                
                # Run the app in interactive mode
                asyncio.run(run_app(str(example_glue_file), interactive=True))
                
                # Check that interactive_session was called
                mock_interactive.assert_called_once_with(mock_app)
                mock_app.setup.assert_called_once()
                mock_app.close.assert_called_once()

# ==================== Tool Management Tests ====================
class TestToolManagement:
    """Test tool management functionality."""

    def test_list_tools_command(self, capsys):
        """Test that list-tools shows available tools."""
        with patch.object(sys, 'argv', ['glue', 'list-tools']):
            with patch('glue.cli.main', side_effect=SystemExit(0)):
                try:
                    main()
                except SystemExit:
                    pass
            
            captured = capsys.readouterr()
            
            # Check for built-in tools
            assert "web_search" in captured.out
            assert "file_handler" in captured.out
            assert "code_interpreter" in captured.out

# ==================== Model Management Tests ====================
class TestModelManagement:
    """Test model management functionality."""

    def test_list_models_command(self, capsys):
        """Test that list-models shows available models with provider info."""
        with patch.object(sys, 'argv', ['glue', 'list-models']):
            with patch('glue.cli.list_models') as mock_list_models:
                try:
                    main()
                except SystemExit:
                    pass
            
            # Check that list_models was called
            mock_list_models.assert_called_once()

# ==================== DSL and Config Testing ====================
class TestDSLAndConfig:
    """Test DSL parsing and configuration."""

    def test_validate_command(self, example_glue_file, capsys):
        """Test that validate command checks GLUE file syntax."""
        with patch.object(sys, 'argv', ['glue', 'validate', str(example_glue_file)]):
            with patch('glue.cli.validate_glue_file') as mock_validate:
                mock_validate.return_value = True
                try:
                    main()
                except SystemExit:
                    pass
            
            # Check that validate_glue_file was called with the correct file
            mock_validate.assert_called_once_with(str(example_glue_file))

    def test_validate_command_with_errors(self, invalid_glue_content, tmp_path, capsys):
        """Test that validate command reports syntax errors."""
        # Create a temporary file with invalid content
        invalid_file = tmp_path / "invalid.glue"
        with open(invalid_file, "w") as f:
            f.write(invalid_glue_content)
        
        with patch.object(sys, 'argv', ['glue', 'validate', str(invalid_file)]):
            with patch('glue.cli.validate_glue_file') as mock_validate:
                mock_validate.return_value = False
                mock_validate.side_effect = ValueError("Syntax error: Missing closing brace")
                try:
                    main()
                except (SystemExit, ValueError):
                    pass
            
            # Check that validate_glue_file was called with the correct file
            mock_validate.assert_called_once_with(str(invalid_file))

# ==================== Adhesive & Magnetic System Tests ====================
class TestAdhesiveAndMagneticSystem:
    """Test adhesive and magnetic system integration with CLI."""

    def test_run_app_with_adhesive_configuration(self, simple_glue_content, tmp_path):
        """Test that run_app properly handles adhesive configurations."""
        # Create a temporary file with the simple glue content
        glue_file = tmp_path / "simple.glue"
        with open(glue_file, "w") as f:
            f.write(simple_glue_content)
            
        with patch('glue.core.GlueApp') as MockGlueApp:
            mock_app = MockGlueApp.return_value
            mock_app.setup = MagicMock()
            mock_app.run = AsyncMock()
            mock_app.close = AsyncMock()
            
            # Run the app
            asyncio.run(run_app(str(glue_file), "Test input"))
            
            # Check app was initialized and setup
            MockGlueApp.assert_called_once_with(config_file=str(glue_file))
            mock_app.setup.assert_called_once()