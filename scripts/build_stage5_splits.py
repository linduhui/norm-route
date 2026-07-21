"""Script entry point for the Stage 5 category-held-out split builder."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.build_stage5_splits import (  # noqa: E402,F401
    Stage5SplitError,
    build_stage5_splits,
    load_category_cv_config,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
