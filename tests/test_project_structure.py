from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required_files = [
        "AGENTS.md",
        "configs/paths.example.yaml",
        "pyproject.toml",
        "tests/test_project_structure.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/evaluation/__init__.py",
        "src/experts/__init__.py",
        "src/utils/__init__.py",
    ]

    missing = [path for path in required_files if not (ROOT / path).is_file()]

    assert missing == []


def test_paths_example_contains_required_keys():
    text = (ROOT / "configs" / "paths.example.yaml").read_text(encoding="utf-8")

    required_keys = [
        "project_root:",
        "mvtec_root:",
        "visa_root:",
        "output_root:",
        "cache_root:",
    ]

    missing = [key for key in required_keys if key not in text]

    assert missing == []


def test_agents_rules_document_required_constraints():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    required_phrases = [
        "label, mask, and defect_type must never enter model or Agent input",
        "official train/good split",
        "same support_set_id",
        "must never be used to tune thresholds or hyperparameters",
        "Do not automatically download datasets or model weights",
        "Do not silently skip failed samples",
        "save config, seed, git commit, environment, predictions, and failures",
        "run pytest",
        "must not modify the core algorithms of external baselines",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]

    assert missing == []
