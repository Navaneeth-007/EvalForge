"""Re-score saved predictions on CPU without downloading or running a model."""

import argparse
import json
from pathlib import Path
import subprocess

from eval.bleu_score import compute_bleu
from eval.exact_match import compute_exact_match_accuracy
from eval.execution_acc import compute_execution_accuracy


def rescore(report: dict, db_dir: str) -> dict:
    rows = report["predictions"]
    if not rows or len(rows) != report["n"]:
        raise ValueError("Saved predictions must match the reported sample count")
    gold = [row["sql"] for row in rows]
    db_ids = [row["db_id"] for row in rows]
    for key, field in [
        ("base_model_metrics", "base_prediction"),
        ("finetuned_model_metrics", "finetuned_prediction"),
    ]:
        predictions = [row[field] for row in rows]
        if not all(isinstance(prediction, str) for prediction in predictions):
            raise ValueError(f"Missing saved predictions for {field}")
        exact = compute_exact_match_accuracy(predictions, gold)
        bleu = compute_bleu(predictions, gold)
        execution = compute_execution_accuracy(predictions, gold, db_ids, db_dir)
        old = report.get(key, {})
        report[key] = {
            "exact_match": exact["exact_match_accuracy"],
            "exact_match_details": exact,
            "bleu4": bleu.get("bleu4"),
            "bleu_details": bleu,
            "execution_accuracy": execution.get("execution_accuracy"),
            "execution_details": execution,
            "bertscore_f1": old.get("bertscore_f1"),
            "inference_time_s": old.get("inference_time_s"),
        }
    report["scoring_protocol"] = "CPU replay; read-only SQLite; numeric-equivalent row multisets"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    report["scoring_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = rescore(json.loads(Path(args.report).read_text()), args.db_dir)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: {
                    metric: report[key][metric] for metric in ["exact_match", "execution_accuracy", "bleu4"]
                }
                for key in ["base_model_metrics", "finetuned_model_metrics"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
