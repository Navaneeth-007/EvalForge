"""Promote measured experiment artifacts into the public recorded-prediction viewer."""

import argparse
import json
from pathlib import Path


def build_evidence(manifest, report):
    predictions = report.get("predictions", [])
    if not predictions or len(predictions) != report.get("n"):
        raise ValueError("Report must contain every evaluated prediction")
    if len(report.get("dataset_sha256", "")) != 64:
        raise ValueError("Report must identify its evaluation data hash")
    if manifest.get("train_examples", 0) <= 0 or not manifest.get("gpu"):
        raise ValueError("A measured GPU training manifest is required")
    keys = ["exact_match", "execution_accuracy", "bleu4", "bertscore_f1"]
    metrics = {}
    for label, source in [("base", "base_model_metrics"), ("adapter", "finetuned_model_metrics")]:
        if source not in report:
            raise ValueError("Both base and adapter metrics are required")
        metrics[label] = {key: report[source].get(key) for key in keys}
        for value in metrics[label].values():
            if value is not None and not 0 <= value <= 1:
                raise ValueError("Metric values must be in [0, 1]")
    return {
        "status": "pilot_complete",
        "kind": "recorded_predictions",
        "message": "Recorded GPU pilot predictions; this service does not run live model inference.",
        "experiment": {
            key: manifest.get(key)
            for key in [
                "base_model",
                "base_revision",
                "git_commit",
                "train_examples",
                "validation_examples",
                "requested_sample_size",
                "sample_selection",
                "gpu",
                "elapsed_seconds",
                "peak_allocated_bytes",
                "archive_repo",
                "archive_revision",
                "archive_sha256",
            ]
        },
        "training_config": manifest["config"],
        "evaluation": {
            "n": report["n"],
            "dataset_sha256": report["dataset_sha256"],
            "decoding": report["decoding"],
        },
        "metrics": metrics,
        "predictions": predictions,
        "attribution": "Spider: Yu et al., EMNLP 2018. Dataset: CC BY-SA 4.0. https://yale-lily.github.io/spider",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="app/evidence/pilot.json")
    args = parser.parse_args()
    evidence = build_evidence(
        json.loads(Path(args.manifest).read_text()), json.loads(Path(args.report).read_text())
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2))
    print(f"Published {evidence['evaluation']['n']} recorded comparisons to {target}")


if __name__ == "__main__":
    main()
