"""Script entry point for the Stage 4 final artifact gate."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.validate_stage4_outputs import (  # noqa: E402,F401
    inspect_stage4_outputs,
    main,
    validate_stage4_outputs,
)


if __name__ == "__main__":
    main()
