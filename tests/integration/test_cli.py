import pytest
from typer.testing import CliRunner
import os

# Assuming your Typer app object is named 'app' in src/glue/cli.py
# Adjust the import path if necessary
from glue.cli import app

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
    # However, typer's CliRunner handles paths relative to the cwd
    # where pytest is run, so using the absolute path is safer.
    result = runner.invoke(app, ["run", str(app_file)])

    # --- RED Phase Assertion ---
    # This assertion should FAIL initially because Agno is not integrated
    # and the current 'run' command likely expects different file content/structure.
    # We expect a successful exit code once implemented.
    assert result.exit_code == 0
    # We might also assert specific output later, e.g.:
    # assert "Agno workflow started" in result.stdout
