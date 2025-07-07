#!/usr/bin/env python3
"""Main CLI entry point for PSIREG renewable energy grid system.

This script provides the 'psi' command line interface for running
renewable energy grid simulations with scenario orchestration.

Usage:
    psi simulate --scenario storm_day
    psi list-scenarios
    psi version
"""

import sys
from pathlib import Path

from psireg.cli.main import create_cli_app

# Add src to Python path for development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main():
    """Main CLI entry point."""
    try:
        app = create_cli_app()
        app()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
