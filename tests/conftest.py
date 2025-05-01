import os
import sys

# Ensure the 'src' directory is on sys.path so the 'glue' package can be imported
dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if dirpath not in sys.path:
    sys.path.insert(0, dirpath)
