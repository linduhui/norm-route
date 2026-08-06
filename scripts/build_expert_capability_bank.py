"""Script entry point for Stage 5 Expert Capability Profile Bank construction."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.build_expert_capability_bank import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
