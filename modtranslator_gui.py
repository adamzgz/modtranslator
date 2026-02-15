"""Entry point for PyInstaller builds."""

import sys
import os

# Handle frozen (PyInstaller) paths
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle
    _MEIPASS = sys._MEIPASS  # type: ignore[attr-defined]
    # Add the bundled package to sys.path
    if _MEIPASS not in sys.path:
        sys.path.insert(0, _MEIPASS)

from modtranslator.gui.app import run_gui

if __name__ == "__main__":
    run_gui()
