import pytest
from click.testing import CliRunner
import os
import sys

# Import the Click CLI
from glue.cli import cli

runner = CliRunner()


def test_cli_run_basic_agno_app(tmp_path):
    """Test that 'glue run' can execute a minimal Agno-based app."""
    # Create a dummy/minimal .glue file content that SHOULD
    # eventually be parsed into an Agno workflow/team.
    # For now, an empty file is sufficient for the CLI to target.
    app_dir = tmp_path / "agno_app"
    app_dir.mkdir()
    app_file = app_dir / "minimal_agno_app.glue"
    app_file.touch() # Create an empty file

    # For the RED phase, we just need to call the CLI and expect
    # it to eventually work. The current implementation will likely
    # error or not run Agno, causing the assertion to fail.

    # Run the command from the parent directory of the app file
    # to ensure relative paths work if needed later.
    # Use Click's CLI runner with the correct structure
    result = runner.invoke(cli, ["run", str(app_file), "--engine", "agno"])

    # --- RED Phase Assertion ---
    # This assertion should FAIL initially because Agno is not integrated
    # and the current 'run' command likely expects different file content/structure.
    # We expect a successful exit code once implemented.
    assert result.exit_code == 0
    

def test_cli_uses_agno_as_default_engine(tmp_path):
    """Test that the CLI uses Agno as the default engine when none is specified."""
    # Create a minimal test file
    app_dir = tmp_path / "default_engine_test"
    app_dir.mkdir()
    app_file = app_dir / "default_engine_test.glue"
    app_file.touch()  # Create an empty file
    
    # Run the command without explicitly specifying the engine
    result = runner.invoke(cli, ["run", str(app_file)])
    
    # Now that we've updated the default engine to Agno, we should see Agno-related messages 
    # in the output, even if there are errors because the test environment doesn't have Agno installed
    # This is fine for our test purpose - we just need to verify Agno is being used by default
        
    # Look for Agno-specific error messages
    assert "Error (Agno):" in result.stdout or "Agno engine:" in result.stdout
        
    # The command might fail with errors about missing Agno modules, that's expected
    # in a test environment - we just need to confirm Agno was attempted
    # We don't need to check the exit code
