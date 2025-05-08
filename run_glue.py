#!/usr/bin/env python3
"""
GLUE CLI Wrapper Script

This script provides a direct way to run the GLUE CLI when the standard
installation entry point isn't working properly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the CLI function and run it
from glue.cli import cli

if __name__ == "__main__":
    # Remove this script's name from argv and replace with 'glue'
    sys.argv[0] = 'glue'
    cli()
