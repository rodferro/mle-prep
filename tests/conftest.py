"""
Tests import your code from `problems/` by default.
To run the same tests against the reference solutions:

    USE_SOLUTIONS=1 pytest
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load(module_name: str):
    package = "solutions" if os.environ.get("USE_SOLUTIONS") else "problems"
    return importlib.import_module(f"{package}.{module_name}")
