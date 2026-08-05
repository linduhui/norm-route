#!/usr/bin/env python3
"""Repository entry point for the label-free FBDP-AD artifact audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normroute.cli.audit_fbdp_ad import main, parse_args  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
