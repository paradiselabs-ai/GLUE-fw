"""
End-to-end integration tests for the GLUE framework with Agno as the engine.

These tests verify that a complete GLUE application with multiple teams,
tools, adhesives, and magnetic flows works correctly when running on the
Agno engine.
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from click.testing import CliRunner

from glue.cli import cli
from glue.core.adapters.agno import GlueAgnoAdapter
from glue.dsl.parser import GlueParser


@pytest.fixture
def temp_glue_app_dir():
    """Create a temporary directory for GLUE app tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_complete_application_structure(temp_glue_app_dir):
    """
    Test that a complete GLUE application can be parsed and set up with Agno.
    
    This test verifies that:
    1. A complex GLUE app with multiple teams can be parsed
    2. All components (teams, agents, tools) are correctly created in Agno
    3. Team connections (magnetic flows) are properly established
    4. Adhesive bindings are correctly set up
    """
    # Set up a complete GLUE application config
    config_path = os.path.join(temp_glue_app_dir, "complete_app.glue")
    with open(config_path, "w") as f:
        f.write("""
# Complete GLUE application for end-to-end testing
app_name: CompleteAgnoTestApp
engine: agno
description: "A complete test application with multiple teams, tools, and flows"

teams:
  ResearchTeam:
    description: "Conducts research and information gathering"
    lead: true
    agents:
      - name: Researcher
        provider: gemini
        model: gemini-1.5-pro
      - name: Analyst
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - SearchTool
      - FileHandlerTool
  
  CodeTeam:
    description: "Writes and analyzes code"
    agents:
      - name: Programmer
        provider: gemini
        model: gemini-1.5-pro
      - name: Reviewer
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - CodeInterpreterTool
      - FileHandlerTool
  
  QATeam:
    description: "Tests and validates work"
    agents:
      - name: Tester
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - FileHandlerTool

# Define magnetic flows between teams
magnetic_field:
  name: ProjectFlow
  flows:
    - source: ResearchTeam
      target: CodeTeam
      type: PUSH
    - source: CodeTeam
      target: QATeam
      type: PUSH
    - source: QATeam
      target: ResearchTeam
      type: PULL

# Define tool bindings with adhesives
adhesives:
  - tool: SearchTool
    type: VELCRO
  - tool: FileHandlerTool
    type: GLUE
  - tool: CodeInterpreterTool
    type: TAPE
""")

    # Test using the GlueAgnoAdapter directly
    parser = GlueParser()
    config = parser.parse_file(config_path)
    
    # Create adapter and attempt setup
    adapter = GlueAgnoAdapter()
    # This test is expected to fail initially as some Agno integration might not be complete
    try:
        # Only test the setup part without running the workflow
        adapter.setup(config)
        
        # Verify correct setup
        assert adapter.workflow is not None, "Agno workflow was not created"
        assert len(adapter.teams) == 3, f"Expected 3 teams but found {len(adapter.teams)}"
        assert len(adapter.agents) >= 5, f"Expected at least 5 agents but found {len(adapter.agents)}"
        assert len(adapter.tools) >= 3, f"Expected at least 3 tools but found {len(adapter.tools)}"
        
        # Verify magnetic flows (team connections)
        assert len(adapter.workflow.team_connections) >= 3, \
            f"Expected at least 3 team connections but found {len(adapter.workflow.team_connections)}"
        
        setup_successful = True
    except Exception as e:
        setup_successful = False
        pytest.fail(f"Failed to set up Agno adapter: {str(e)}")
    
    assert setup_successful, "Failed to set up complete Agno application"


def test_complete_application_via_cli(temp_glue_app_dir):
    """
    Test that a complete GLUE application can be run via the CLI with Agno.
    
    This test verifies the CLI integration, focusing on the application startup 
    and initialization phases.
    """
    # Set up a complete GLUE application config (simplified for CLI test)
    config_path = os.path.join(temp_glue_app_dir, "cli_app.glue")
    with open(config_path, "w") as f:
        f.write("""
# GLUE application for CLI end-to-end testing
app_name: CLIAgnoTestApp
engine: agno
description: "A test application for CLI integration"

teams:
  MainTeam:
    description: "Main team for testing"
    lead: true
    agents:
      - name: Agent1
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - SearchTool
""")

    # Run the CLI command
    runner = CliRunner()
    result = runner.invoke(cli, ['run', config_path, '--input', 'This is a test input'])
    
    # This test is expected to fail initially until the Agno CLI integration is complete
    # We're testing that the CLI attempts to run the Agno workflow
    assert "Agno" in result.output, "CLI output doesn't mention Agno engine"


def test_adhesive_persistence_between_teams(temp_glue_app_dir):
    """
    Test that adhesives correctly maintain persistence between teams in Agno.
    
    This ensures that GLUE's adhesive system works correctly with Agno's persistence
    mechanism.
    """
    # Set up a GLUE app with adhesives between teams
    config_path = os.path.join(temp_glue_app_dir, "adhesive_test.glue")
    with open(config_path, "w") as f:
        f.write("""
# GLUE application for testing adhesives with Agno
app_name: AdhesiveAgnoTestApp
engine: agno
description: "Tests adhesive persistence between teams"

teams:
  TeamA:
    description: "First team"
    lead: true
    agents:
      - name: Agent1
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - SearchTool
  
  TeamB:
    description: "Second team"
    agents:
      - name: Agent2
        provider: gemini
        model: gemini-1.5-pro
    tools:
      - SearchTool

magnetic_field:
  name: TestFlow
  flows:
    - source: TeamA
      target: TeamB
      type: BIDIRECTIONAL

adhesives:
  - tool: SearchTool
    type: GLUE
""")

    # Parse the config
    parser = GlueParser()
    config = parser.parse_file(config_path)
    
    # Create adapter and set up
    adapter = GlueAgnoAdapter()
    try:
        adapter.setup(config)
        assert adapter.adhesive_system is not None, "Adhesive system was not created"
        
        # Ideally we would test actual adhesive persistence here,
        # but for now we'll just verify the setup
        setup_successful = True
    except Exception as e:
        setup_successful = False
        pytest.fail(f"Failed to set up adhesive test: {str(e)}")
    
    assert setup_successful, "Failed to set up adhesive test"


def test_magnetic_flow_communication(temp_glue_app_dir):
    """
    Test that magnetic flows correctly enable communication between teams in Agno.
    
    This ensures that GLUE's magnetic field system properly integrates with Agno's
    team connection mechanism.
    """
    # Set up a GLUE app with various magnetic flows
    config_path = os.path.join(temp_glue_app_dir, "magnetic_test.glue")
    with open(config_path, "w") as f:
        f.write("""
# GLUE application for testing magnetic flows with Agno
app_name: MagneticAgnoTestApp
engine: agno
description: "Tests magnetic flow communication between teams"

teams:
  TeamA:
    description: "First team"
    lead: true
    agents:
      - name: Agent1
        provider: gemini
        model: gemini-1.5-pro
  
  TeamB:
    description: "Second team"
    agents:
      - name: Agent2
        provider: gemini
        model: gemini-1.5-pro
  
  TeamC:
    description: "Third team"
    agents:
      - name: Agent3
        provider: gemini
        model: gemini-1.5-pro

magnetic_field:
  name: TestFlow
  flows:
    - source: TeamA
      target: TeamB
      type: PUSH
    - source: TeamB
      target: TeamC
      type: PUSH
    - source: TeamC
      target: TeamA
      type: PULL
""")

    # Parse the config
    parser = GlueParser()
    config = parser.parse_file(config_path)
    
    # Create adapter and set up
    adapter = GlueAgnoAdapter()
    try:
        adapter.setup(config)
        
        # Verify magnetic flows were set up
        assert len(adapter.workflow.team_connections) >= 3, \
            f"Expected at least 3 team connections but found {len(adapter.workflow.team_connections)}"
        
        setup_successful = True
    except Exception as e:
        setup_successful = False
        pytest.fail(f"Failed to set up magnetic flow test: {str(e)}")
    
    assert setup_successful, "Failed to set up magnetic flow test"
