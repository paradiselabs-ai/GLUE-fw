"""
GLUE Framework CLI Entry Point

This module provides the main entry point for the GLUE CLI.
"""

from .cli import cli

def main():
    """Main entry point for the GLUE CLI."""
    cli(prog_name="glue")

if __name__ == "__main__":
    main()
