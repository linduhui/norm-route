import csv
import json
from pathlib import Path

from src.normroute.cli.validate_stage4_outputs import (
    _check_policy_feature_manifests,
    _check_pre_route_tasks,
    _check_split_isolation,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _task(seed: int = 0) -> dict[str, object]:
    return {
        "protocol_version": "stage4.pre_route.v1",
        "task_id": f"task-{seed}",
        "sample_id": "sample-0",
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 1,
        "seed": seed,
        "support_set_id": f"mvtec_bottle_k1_seed{seed}",
        "candidate_experts": ["PatchCore", "WinCLIP", "AnomalyDINO"],
        "policy_features": {"dataset": "mvtec", "category": "bottle", "k_shot": 1},
    }


def test_pre_route_gate_rejects_nested_forbidden_field(tmp_path: Path) -> None:
    path = tmp_path / "pre_route_tasks.jsonl"
    clean = _task()
    path.write_text(json.dumps(clean) + "\n", encoding="utf-8")
    assert _check_pre_route_tasks(path)["status"] == "PASS"

    clean["policy_features"]["label"] = 1  # type: ignore[index]
    path.write_text(json.dumps(clean) + "\n", encoding="utf-8")
    result = _check_pre_route_tasks(path)
    assert result["status"] == "FAIL"
    assert "forbidden" in result["errors"][0].lower()


def test_feature_manifest_checks_features_but_allows_explicit_denylist(tmp_path: Path) -> None:
    manifest = tmp_path / "policies" / "tree" / "fold0" / "feature_manifest.json"
    _write_json(
        manifest,
        {
            "raw_feature_allowlist": ["category", "k_shot", "budget", "static_cost"],
            "encoded_feature_names": ["category=bottle", "k_shot", "budget"],
            "numeric_scaling": {"k_shot": {"mean": 1.0, "scale": 1.0}},
            "feature_denylist": ["seed", "support_set_id", "label", "expert_scores"],
        },
    )
    assert _check_policy_feature_manifests(tmp_path / "policies")["status"] == "PASS"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["encoded_feature_names"].append("seed")
    _write_json(manifest, payload)
    result = _check_policy_feature_manifests(tmp_path / "policies")
    assert result["status"] == "FAIL"
    assert "seed" in result["errors"][0]


def test_split_gate_uses_seed_as_isolation_unit(tmp_path: Path) -> None:
    manifest = tmp_path / "fold_manifest.csv"
    rows = [
        {"fold": "fold0", "split": "train", "task_id": "task-0", "seed": 0},
        {"fold": "fold0", "split": "val", "task_id": "task-1", "seed": 1},
        {"fold": "fold0", "split": "test", "task_id": "task-2", "seed": 2},
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert _check_split_isolation(manifest, tmp_path / "missing_audit.json")["status"] == "PASS"

    rows[-1]["seed"] = 0
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = _check_split_isolation(manifest, tmp_path / "missing_audit.json")
    assert result["status"] == "FAIL"
    assert any("train/test seed overlap" in error for error in result["errors"])
