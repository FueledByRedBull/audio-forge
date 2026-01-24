"""
MicEq entry point

Run with: python -m mic_eq
"""

import sys


def main():
    """Main entry point for MicEq application."""
    try:
        from .ui import run_app
        return run_app()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Build the Rust core: maturin develop --release")
        print("2. Install PyQt6: pip install PyQt6")
        return 1


if __name__ == "__main__":
    sys.exit(main())
