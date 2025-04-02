#!/usr/bin/env python3
"""
Test script to verify the GLUE package can be installed and imported correctly.
Run this after building the package to ensure everything is working.
"""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main():
    """Create a virtual environment, install the package, and test importing it."""
    # Get the directory of this script and the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Testing GLUE package installation from {project_root}")
    
    # Create a temporary directory for the virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        venv_dir = temp_path / "venv"
        
        print(f"Creating virtual environment in {venv_dir}")
        venv.create(venv_dir, with_pip=True)
        
        # Get the path to the Python executable in the virtual environment
        if sys.platform == "win32":
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
        
        # Build the package if dist directory doesn't exist or is empty
        dist_dir = project_root / "dist"
        if not dist_dir.exists() or not list(dist_dir.glob("*.whl")):
            print("Building package...")
            subprocess.run(
                [sys.executable, "-m", "build"], 
                cwd=project_root, 
                check=True
            )
        
        # Find the wheel file
        wheel_files = list(dist_dir.glob("*.whl"))
        if not wheel_files:
            print("Error: No wheel file found in dist directory")
            return 1
        
        wheel_file = wheel_files[0]
        print(f"Found wheel file: {wheel_file}")
        
        # Install the package
        print("Installing package...")
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", str(wheel_file)],
            check=True
        )
        
        # Create a test script
        test_script = temp_path / "test_import.py"
        test_script.write_text("""
import glue
from glue.core.app import GlueApp
from glue.core.model import Model
from glue.core.team import Team

print(f"Successfully imported GLUE version: {glue.__version__}")
print("Core components available:")
print(f"- GlueApp: {GlueApp}")
print(f"- Model: {Model}")
print(f"- Team: {Team}")
""")
        
        # Run the test script
        print("Testing imports...")
        result = subprocess.run(
            [str(python_exe), str(test_script)],
            capture_output=True,
            text=True
        )
        
        print("\nTest Results:")
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return 1
        
        print("Package test completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
