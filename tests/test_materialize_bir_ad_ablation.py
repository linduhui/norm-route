import json

import pytest

from src.normroute.cli.materialize_bir_ad_ablation import (
    _parse_and_validate_signatures,
    _validate_compute_compatibility,
)
from src.normroute.router.bir_ad_ablation import get_bir_ad_ablation
from src.normroute.router.bir_ad_pipeline import (
    BIR_AD_SIGNATURE_COLUMNS,
    BIR_AD_SIGNATURE_PROTOCOL_VERSION,
)


def _signature_record(task_id: str = "task-0") -> dict:
    row = {column: 0 for column in BIR_AD_SIGNATURE_COLUMNS}
    row.update(
        {
            "protocol_version": BIR_AD_SIGNATURE_PROTOCOL_VERSION,
            "bir_ad_protocol_version": "stage5.bir_ad.v4",
            "task_id": task_id,
            "dataset": "mvtec",
            "category": "bottle",
            "support_set_id": "support-0",
            "encoder_fingerprint": "encoder",
            "normalization_sha256": "a" * 64,
            "alignment_fingerprint": "alignment",
            "consistency_backend": "torch",
            "consistency_device": "cuda:0",
            "consistency_dtype": "float64",
            "query_image_sha256": "b" * 64,
            "support_image_sha256s": ["c" * 64],
            "bai_vector": [0.0] * 5,
            "clear_representation": [0.0],
            "ambiguous_representation": [0.0],
        }
    )
    return row


def test_router_only_materialization_requires_compute_equivalence() -> None:
    source = get_bir_ad_ablation("full").to_dict()

    for target in (
        "plus_cross_modal_disagreement",
        "plus_support_consistency",
        "representations_only",
    ):
        _validate_compute_compatibility(
            source, get_bir_ad_ablation(target).to_dict()
        )
    with pytest.raises(ValueError, match="compute settings differ"):
        _validate_compute_compatibility(
            source, get_bir_ad_ablation("plus_sobel").to_dict()
        )


def test_materialization_rejects_mixed_consistency_runtime() -> None:
    first = _signature_record("task-0")
    second = _signature_record("task-1")
    second["consistency_device"] = "cuda:1"
    text = "\n".join(
        json.dumps(row, sort_keys=True) for row in (first, second)
    )

    with pytest.raises(ValueError, match="mix consistency runtimes"):
        _parse_and_validate_signatures(text)


def test_materialization_accepts_complete_v2_signatures() -> None:
    row = _signature_record()
    parsed = _parse_and_validate_signatures(json.dumps(row, sort_keys=True))

    assert parsed == [row]
