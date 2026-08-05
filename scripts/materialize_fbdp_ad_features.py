#!/usr/bin/env python3
"""Repository entry point for FBDP Router feature materialization."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normroute.cli.materialize_fbdp_ad_features import (  # noqa: E402,F401
    main,
    parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())
