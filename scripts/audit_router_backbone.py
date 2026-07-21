"""Script entry point for the Stage 5 router backbone audit."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.cli.audit_router_backbone import (  # noqa: E402,F401
    BACKBONE_AUDIT_PROTOCOL_VERSION,
    audit_router_backbone,
    main,
    parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())
