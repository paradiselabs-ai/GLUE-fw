# tests/integration/test_agno_base.py
import pytest
from click.testing import CliRunner

from glue.cli import cli


@pytest.mark.integration
def test_minimal_agno_workflow_via_cli():
    """Test running a minimal Agno workflow via the 'glue run' CLI command."""
    runner = CliRunner()
    # TODO: Define how 'glue run' will invoke Agno initially.
    # For now, assume a hypothetical flag or default behavior shift.
    # We might need a minimal config file later.
    result = runner.invoke(cli, ['run', '--engine', 'agno', 'tests/integration/fixtures/minimal_agno_config.glue'])

    # Assert successful execution (exit code 0)
    # This WILL FAIL initially as cli.py doesn't support '--engine agno' yet.
    assert result.exit_code == 0, f"CLI command failed with output: {result.output}"
    # TODO: Potentially assert specific output indicating Agno ran.
    # assert "Agno workflow executed successfully" in result.output
