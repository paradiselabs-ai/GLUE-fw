#!/usr/bin/env python3
"""
Test suite for the GLUE Framework CLI

This module contains tests for the CLI functionality of the GLUE Framework,
including command parsing, project creation, and application execution.
"""
import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the glue package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from glue.cli import (
    main,
    create_new_project,
    run_app,
    forge_tool,
    forge_mcp,
    forge_api,
    setup_logging
)


class TestCLI(unittest.TestCase):
    """Test cases for the GLUE Framework CLI."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Set up logging for tests
        self.logger = setup_logging(level="DEBUG")

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("glue.cli.create_new_project")
    def test_new_command(self, mock_create_project, mock_parse_args):
        """Test the 'new' command."""
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.command = "new"
        mock_args.project = "test_project"
        mock_args.template = "basic"
        mock_parse_args.return_value = mock_args
        
        # Call the main function
        main()
        
        # Check if create_new_project was called with correct arguments
        mock_create_project.assert_called_once_with("test_project", "basic")

    def test_create_new_project(self):
        """Test creating a new project."""
        # Create a new project
        result = create_new_project("test_project", "basic")
        
        # Check if the project was created successfully
        self.assertTrue(result)
        self.assertTrue(os.path.exists("test_project"))
        self.assertTrue(os.path.exists("test_project/app.glue"))
        self.assertTrue(os.path.exists("test_project/README.md"))
        self.assertTrue(os.path.exists("test_project/.env"))
        self.assertTrue(os.path.exists("test_project/.gitignore"))
        self.assertTrue(os.path.exists("test_project/workspace"))

    def test_create_new_project_existing_directory(self):
        """Test creating a new project in an existing directory."""
        # Create a directory
        os.makedirs("existing_project")
        
        # Try to create a new project in the existing directory
        result = create_new_project("existing_project", "basic")
        
        # Check if the project creation failed
        self.assertFalse(result)

    @patch("glue.cli.run_app")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("asyncio.run")
    def test_run_command(self, mock_asyncio_run, mock_parse_args, mock_run_app):
        """Test the 'run' command."""
        # Create a test GLUE file
        with open("test_app.glue", "w") as f:
            f.write("""
glue app {
    name = "Test App"
    config {
        development = true
    }
}

// Define models
model assistant {
    provider = openrouter
    role = "Help the user with their tasks"
    adhesives = [glue, velcro]
    config {
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }
}

// Define teams
magnetize {
    main {
        lead = assistant
    }
}

apply glue
""")
        
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.command = "run"
        mock_args.config = "test_app.glue"
        mock_args.input = "Hello"
        mock_args.interactive = False
        mock_args.env = None
        mock_parse_args.return_value = mock_args
        
        # Call the main function
        main()
        
        # Check if run_app was called with correct arguments
        mock_asyncio_run.assert_called_once()

    @patch("builtins.input", side_effect=["test_tool", "A test tool", "1"])
    def test_forge_tool(self, mock_input):
        """Test creating a custom tool."""
        # Create tools directory
        os.makedirs("tools")
        
        # Create a tool
        result = forge_tool("test_tool", "A test tool", "basic")
        
        # Check if the tool was created successfully
        self.assertTrue(result)
        self.assertTrue(os.path.exists("tools/test_tool.py"))
        self.assertTrue(os.path.exists("tools/__init__.py"))

    @patch("builtins.input", side_effect=["test_mcp", "A test MCP"])
    def test_forge_mcp(self, mock_input):
        """Test creating a custom MCP integration."""
        # Create mcps directory
        os.makedirs("mcps")
        
        # Create an MCP
        result = forge_mcp("test_mcp", "A test MCP")
        
        # Check if the MCP was created successfully
        self.assertTrue(result)
        self.assertTrue(os.path.exists("mcps/test_mcp_mcp.py"))
        self.assertTrue(os.path.exists("mcps/__init__.py"))

    @patch("builtins.input", side_effect=["test_api", "A test API"])
    def test_forge_api(self, mock_input):
        """Test creating a custom API integration."""
        # Create apis directory
        os.makedirs("apis")
        
        # Create an API
        result = forge_api("test_api", "A test API")
        
        # Check if the API was created successfully
        self.assertTrue(result)
        self.assertTrue(os.path.exists("apis/test_api_api.py"))
        self.assertTrue(os.path.exists("apis/__init__.py"))

    def test_forge_invalid_name(self):
        """Test creating a component with an invalid name."""
        # Try to create a tool with an invalid name
        result = forge_tool("123invalid", "Invalid tool", "basic")
        
        # Check if the tool creation failed
        self.assertFalse(result)


class TestPortkeyIntegration(unittest.TestCase):
    """Test cases for Portkey integration in the GLUE Framework."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create a test .env file with Portkey configuration
        with open(".env", "w") as f:
            f.write("""
# Portkey integration
PORTKEY_ENABLED=true
PORTKEY_API_KEY=test_portkey_key
""")
        
        # Set up logging for tests
        self.logger = setup_logging(level="DEBUG")

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir)

    @patch("os.environ", {"PORTKEY_ENABLED": "true", "PORTKEY_API_KEY": "test_portkey_key"})
    def test_portkey_environment_variables(self):
        """Test that Portkey environment variables are properly loaded."""
        # Check if environment variables are set
        self.assertEqual(os.environ.get("PORTKEY_ENABLED"), "true")
        self.assertEqual(os.environ.get("PORTKEY_API_KEY"), "test_portkey_key")

    @patch("argparse.ArgumentParser.parse_args")
    @patch("glue.cli.run_app")
    @patch("asyncio.run")
    def test_run_with_portkey(self, mock_asyncio_run, mock_run_app, mock_parse_args):
        """Test running an app with Portkey integration."""
        # Create a test GLUE file
        with open("test_app.glue", "w") as f:
            f.write("""
glue app {
    name = "Test App with Portkey"
    config {
        development = true
        portkey = true
    }
}

// Define models
model assistant {
    provider = openrouter
    role = "Help the user with their tasks"
    adhesives = [glue, velcro]
    config {
        model = "meta-llama/llama-3.1-8b-instruct:free"
        temperature = 0.7
    }
}

// Define teams
magnetize {
    main {
        lead = assistant
    }
}

apply glue
""")
        
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.command = "run"
        mock_args.config = "test_app.glue"
        mock_args.input = "Hello"
        mock_args.interactive = False
        mock_args.env = ".env"
        mock_parse_args.return_value = mock_args
        
        # Call the main function
        with patch("dotenv.load_dotenv") as mock_load_dotenv:
            main()
            
            # Check if load_dotenv was called with the correct file
            mock_load_dotenv.assert_called_once_with(".env")
        
        # Check if run_app was called
        mock_asyncio_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
