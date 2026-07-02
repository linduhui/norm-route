import json
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from src.experts.fake_expert import FakeExpert
from src.experts.protocol import ExpertProtocol
from src.experts.results import (
    FORBIDDEN_EXPERT_RESULT_FIELDS,
    ExpertResult,
    ExpertResultValidationError,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)


def test_fake_expert_implements_protocol_and_writes_jsonl(tmp_path):
    support_path = tmp_path / "train" / "good" / "support.png"
    image_path = tmp_path / "test" / "query.png"
    _write_png(support_path)
    _write_png(image_path)

    expert = cast(ExpertProtocol, FakeExpert())
    expert.fit_support([str(support_path)], support_set_id="mvtec_k1_seed0", seed=7)
    result = expert.predict(str(image_path), image_id="bottle/000", output_dir=str(tmp_path))

    assert result.expert_name == "fake_expert"
    assert result.support_set_id == "mvtec_k1_seed0"
    assert result.status == "ok"
    assert 0.0 <= result.normalized_score <= 1.0
    assert Path(result.anomaly_map_path).is_file()

    jsonl_path = tmp_path / "predictions.jsonl"
    result.append_jsonl(jsonl_path)
    restored = ExpertResult.from_jsonl_line(jsonl_path.read_text(encoding="utf-8"))

    assert restored == result


def test_fake_expert_is_deterministic_for_image_id_and_seed(tmp_path):
    image_path = tmp_path / "query.png"
    _write_png(image_path)

    first = FakeExpert()
    first.fit_support([], support_set_id="support-a", seed=11)
    first_result = first.predict(str(image_path), image_id="same-image", output_dir=str(tmp_path / "a"))

    second = FakeExpert()
    second.fit_support([], support_set_id="support-a", seed=11)
    second_result = second.predict(str(image_path), image_id="same-image", output_dir=str(tmp_path / "b"))

    assert first_result.raw_score == second_result.raw_score
    assert first_result.normalized_score == second_result.normalized_score
    assert Path(first_result.anomaly_map_path).read_bytes() == Path(second_result.anomaly_map_path).read_bytes()


def test_expert_result_rejects_missing_required_fields():
    payload = {
        "schema_version": "1.0",
        "expert_name": "fake_expert",
    }

    with pytest.raises(ExpertResultValidationError, match="Missing ExpertResult fields"):
        ExpertResult.from_dict(payload)


def test_expert_result_rejects_forbidden_fields():
    payload = {
        "schema_version": "1.0",
        "expert_name": "fake_expert",
        "image_id": "image-1",
        "support_set_id": "support-1",
        "raw_score": 0.1,
        "normalized_score": 0.1,
        "anomaly_map_path": "map.png",
        "runtime_ms": 0.0,
        "peak_memory_mb": 0.0,
        "model_version": "fake-0",
        "weight_hash": "no-weights",
        "status": "ok",
        "error_message": "",
        "label": 1,
    }

    with pytest.raises(ExpertResultValidationError, match="Forbidden ExpertResult fields"):
        ExpertResult.from_dict(payload)


def test_expert_result_schema_excludes_ground_truth_fields():
    assert not FORBIDDEN_EXPERT_RESULT_FIELDS.intersection(ExpertResult.field_names())


def test_expert_result_jsonl_is_json_object_without_forbidden_fields(tmp_path):
    result = ExpertResult(
        schema_version="1.0",
        expert_name="fake_expert",
        image_id="image-1",
        support_set_id="support-1",
        raw_score=0.2,
        normalized_score=0.2,
        anomaly_map_path="map.png",
        runtime_ms=0.0,
        peak_memory_mb=0.0,
        model_version="fake-0",
        weight_hash="no-weights",
        status="ok",
        error_message="",
    )

    line = result.to_jsonl_line()
    payload = json.loads(line)

    assert isinstance(payload, dict)
    assert not FORBIDDEN_EXPERT_RESULT_FIELDS.intersection(payload)
