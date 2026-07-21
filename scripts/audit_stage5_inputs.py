"""Script entry point for the Stage 5 inference-visible input audit."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.audit_stage5_inputs import (  # noqa: E402,F401
    Stage5InputAuditError,
    audit_inference_visible_file,
    audit_stage5_inputs,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
