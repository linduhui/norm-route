from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_routing_matrix_schema_documents_outputs_and_leakage_boundary() -> None:
    schema_path = ROOT / "reports" / "stage3" / "routing_matrix_schema.md"
    text = schema_path.read_text(encoding="utf-8")

    required_phrases = [
        "routing_matrix_long.csv",
        "routing_matrix_wide.csv",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
        "evaluator-only",
        "label",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "must never be used as Agent-visible input",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]

    assert missing == []
