"""Allow running as python -m modtranslator."""

import sys


def main() -> None:
    if "--gui" in sys.argv:
        sys.argv.remove("--gui")
        from modtranslator.gui.app import run_gui
        run_gui()
    else:
        from modtranslator.cli import app
        app()


main()
