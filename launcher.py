#!/usr/bin/env python3
"""
AudioForge launcher script for PyInstaller
"""
import sys
import os

# Add paths for imports when bundled
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    # Add python directory for mic_eq package
    sys.path.insert(0, os.path.join(sys._MEIPASS, 'python'))
    # Try to find mic_eq_core in bundled modules or parent directory
    found_core = False

    # Check if mic_eq_core is in the bundled Python modules
    try:
        import mic_eq_core
        found_core = True
    except ImportError:
        pass

    # If not found in bundled modules, it should be in _internal
    if not found_core:
        core_path = os.path.join(sys._MEIPASS, '_internal')
        if os.path.exists(core_path):
            sys.path.insert(0, core_path)

from mic_eq.ui import run_app

if __name__ == '__main__':
    sys.exit(run_app())
