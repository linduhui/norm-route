"""Script entry point for the Stage 4 baseline grid."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.run_stage4_grid import main


if __name__ == "__main__":
    main()
