import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_normroute_imports_use_one_package_namespace() -> None:
    script = """
import sys

from normroute.agent.live_executor import LIVE_SELECTED_PREDICTION_COLUMNS
from normroute.agent.replay_executor import SELECTED_PREDICTION_COLUMNS
from normroute.cli.run_agent import parse_args
from normroute.evaluation.stage4 import EVALUATED_SAMPLE_COLUMNS

assert tuple(SELECTED_PREDICTION_COLUMNS) == tuple(
    LIVE_SELECTED_PREDICTION_COLUMNS
)
assert len(EVALUATED_SAMPLE_COLUMNS) == len(set(EVALUATED_SAMPLE_COLUMNS))
assert parse_args(["--mode", "live"]).mode == "live"
assert not any(name.startswith("src.normroute") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
