"""Script entry point for FBDP-AD diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.diagnose_fbdp_ad import (  # noqa: E402,F401
    main,
    parse_args,
    write_fbdp_ad_diagnostics,
)


if __name__ == "__main__":
    raise SystemExit(main())
