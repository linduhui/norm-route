"""Script entry point for full evaluator-only Stage 4 summaries."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.summarize_stage4 import main


if __name__ == "__main__":
    main()
