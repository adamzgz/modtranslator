"""Entry point for PyInstaller builds."""

import multiprocessing
import sys
import os

# MUST be before any other imports — prevents PyInstaller frozen builds
# from opening duplicate GUI windows when libraries spawn child processes.
multiprocessing.freeze_support()

# Guard: if this exe is re-invoked as a child process, exit immediately.
if getattr(sys, "frozen", False):
    if os.environ.get("_MODTRANSLATOR_MAIN") == "1":
        # We are a child process — exit silently
        sys.exit(0)
    os.environ["_MODTRANSLATOR_MAIN"] = "1"

# Prevent tokenizers from spawning child processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Handle frozen (PyInstaller) paths
if getattr(sys, "frozen", False):
    _MEIPASS = sys._MEIPASS  # type: ignore[attr-defined]
    if _MEIPASS not in sys.path:
        sys.path.insert(0, _MEIPASS)

from modtranslator.gui.app import run_gui

if __name__ == "__main__":
    run_gui()
