import pytest
from scripts.publish_evidence import build_evidence


def test_refuses_partial_or_untraceable_reports():
    with pytest.raises(ValueError, match="every evaluated prediction"):
        build_evidence({}, {"n": 2, "predictions": [{}]})
    with pytest.raises(ValueError, match="data hash"):
        build_evidence({}, {"n": 1, "predictions": [{}]})
    with pytest.raises(ValueError, match="GPU training manifest"):
        build_evidence({}, {"n": 1, "predictions": [{}], "dataset_sha256": "a" * 64})


def test_preserves_regressions_instead_of_claiming_improvement():
    manifest = {"train_examples": 1000, "gpu": "test GPU", "config": {"num_train_epochs": 1}}
    report = {
        "n": 1,
        "predictions": [{"question": "test"}],
        "dataset_sha256": "a" * 64,
        "decoding": {"do_sample": False},
        "base_model_metrics": {"execution_accuracy": 0.8},
        "finetuned_model_metrics": {"execution_accuracy": 0.7},
    }
    evidence = build_evidence(manifest, report)
    assert evidence["metrics"]["adapter"]["execution_accuracy"] == 0.7
    assert evidence["metrics"]["base"]["execution_accuracy"] == 0.8
    assert evidence["kind"] == "recorded_predictions"


def test_published_full_data_evidence_matches_traceable_reports():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'reports/full-data/eval_results_verified.json').read_text())
    manifest = json.loads((root / 'reports/full-data/training_manifest.json').read_text())
    evidence = json.loads((root / 'app/evidence/full-data.json').read_text())
    prompts = root / 'reports/full-data/eval_samples.json'
    assert hashlib.sha256(prompts.read_bytes()).hexdigest() == report['dataset_sha256']
    assert evidence == build_evidence(manifest, report)
    assert evidence['status'] == 'full_data_complete'
    assert manifest['gpu_count'] == manifest['world_size'] == 2
    assert len(evidence['predictions']) == report['n'] == 200
    for key in ['base_model_metrics', 'finetuned_model_metrics']:
        details = report[key]['execution_details']
        assert sum(row['correct'] for row in details['per_example']) == details['num_correct']
        assert details['execution_accuracy'] == details['num_correct'] / report['n']
