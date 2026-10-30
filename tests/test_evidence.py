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
