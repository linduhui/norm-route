"""Script entry point for fold-normalized batch BIR-AD construction."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.build_bir_ad import main, parse_args  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
