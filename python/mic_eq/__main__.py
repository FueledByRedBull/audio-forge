"""
AudioForge source/development entry point

Run with: python -m mic_eq
"""

import sys


def main():
    """Main entry point for the AudioForge application."""
    try:
        if "--smoke-test" in sys.argv:
            sys.argv = [arg for arg in sys.argv if arg != "--smoke-test"]
            from .ui.app_bootstrap import run_smoke_test
            from .ui.main_window import MainWindow

            return run_smoke_test(MainWindow)

        from .ui.main_window import run_app
        return run_app()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Build the Rust core: maturin develop --release")
        print("2. Install PyQt6: pip install PyQt6")
        return 1


if __name__ == "__main__":
    sys.exit(main())
