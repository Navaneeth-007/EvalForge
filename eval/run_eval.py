"""
EvalForge — Before vs After Evaluation Harness.

Generates SQL predictions with both the base model and the fine-tuned model,
then computes all 4 metrics side-by-side.

Usage:
    # Full eval (requires HF Hub model)
    python eval/run_eval.py \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --finetuned-model your-username/evalforge-phi3-sql \
        --eval-samples ./data/processed/eval_samples.json \
        --n 200

    # Quick eval with local adapter
    python eval/run_eval.py \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --adapter-dir ./outputs/phi3-sql-qlora/final \
        --n 50

    # Quick smoke test (n=10, no BERTScore)
    python eval/run_eval.py --quick
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from eval.exact_match import compute_exact_match_accuracy
from eval.execution_acc import compute_execution_accuracy
from eval.bleu_score import compute_bleu
from train.dataset_prep import format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_pipeline(
    model_name: str,
    adapter_dir: Optional[str] = None,
    use_4bit: bool = False,
):
    """
    Load a text-generation pipeline.

    adapter_dir: if set, loads LoRA adapter on top of base model.
    use_4bit: enables 4-bit quantization (requires bitsandbytes + CUDA).
    """
    logger.info(f"Loading model: {model_name}" + (f" + adapter: {adapter_dir}" if adapter_dir else ""))

    model_kwargs = {"trust_remote_code": False}

    if use_4bit and torch.cuda.is_available():
        from train.qlora_config import get_bnb_config

        model_kwargs["quantization_config"] = get_bnb_config()
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        # Keep adapters attached; avoid merging into quantized weights.
        logger.info("LoRA adapter loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_ids = [tokenizer.eos_token_id]
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_id is not None and end_id != tokenizer.unk_token_id:
        stop_ids.append(end_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        eos_token_id=list(set(stop_ids)),
        do_sample=False,  # greedy decoding for eval
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────


def generate_sql(pipe, question: str, schema: str) -> str:
    """Generate SQL for a single example."""
    prompt = format_prompt(question=question, schema=schema, sql=None)
    output = pipe(prompt, return_full_text=False)
    raw = output[0]["generated_text"].strip()
    # Extract just the SQL — stop at <|end|> or newline after first statement
    sql = raw.split("<|end|>")[0].split("\n\n")[0].strip()
    return sql


def batch_generate(
    pipe,
    samples: list[dict],
    batch_size: int = 8,
) -> list[str]:
    """Generate SQL for a list of eval samples."""
    predictions = []
    total = len(samples)

    for i in range(0, total, batch_size):
        batch = samples[i : i + batch_size]
        prompts = [format_prompt(sample["question"], sample["schema"]) for sample in batch]
        outputs = pipe(prompts, return_full_text=False, batch_size=batch_size)
        predictions.extend(output[0]["generated_text"].split("<|end|>")[0].strip() for output in outputs)

        logger.info(f"  Generated {min(i + batch_size, total)}/{total}")

    return predictions


# ── Metrics ───────────────────────────────────────────────────────────────────


def run_metrics(
    predictions: list[str],
    gold_queries: list[str],
    db_ids: list[str],
    skip_bertscore: bool = False,
    skip_execution: bool = False,
) -> dict:
    """Compute all 4 metrics."""
    metrics = {}

    logger.info("Computing Exact Match...")
    em = compute_exact_match_accuracy(predictions, gold_queries)
    metrics["exact_match"] = em["exact_match_accuracy"]
    metrics["exact_match_details"] = em

    logger.info("Computing BLEU-4...")
    bleu = compute_bleu(predictions, gold_queries)
    metrics["bleu4"] = bleu.get("bleu4")
    metrics["bleu_details"] = bleu

    if not skip_execution:
        logger.info("Computing Execution Accuracy...")
        exec_acc = compute_execution_accuracy(predictions, gold_queries, db_ids)
        metrics["execution_accuracy"] = exec_acc.get("execution_accuracy")
        metrics["execution_details"] = exec_acc
    else:
        metrics["execution_accuracy"] = None

    if not skip_bertscore:
        from eval.bert_score import compute_bertscore

        logger.info("Computing BERTScore...")
        bs = compute_bertscore(predictions, gold_queries)
        metrics["bertscore_f1"] = bs.get("bertscore_f1")
        metrics["bertscore_details"] = bs
    else:
        metrics["bertscore_f1"] = None

    return metrics


# ── Main eval ─────────────────────────────────────────────────────────────────


def run_eval(
    base_model: str,
    finetuned_model: Optional[str],
    adapter_dir: Optional[str],
    eval_samples_path: str,
    n: int,
    output_path: str,
    skip_bertscore: bool = False,
    skip_execution: bool = False,
    use_4bit: bool = False,
    batch_size: int = 4,
):
    """Full before/after evaluation."""

    # Load eval samples
    logger.info(f"Loading eval samples from {eval_samples_path}")
    with open(eval_samples_path) as f:
        samples = json.load(f)
    samples = samples[:n]
    logger.info(f"Evaluating on {len(samples)} examples")

    gold_queries = [s["sql"] for s in samples]
    db_ids = [s.get("db_id", "unknown") for s in samples]

    if n <= 0 or not samples:
        raise ValueError("Evaluation requires at least one example")
    import hashlib

    results = {
        "n": len(samples),
        "base_model": base_model,
        "dataset_sha256": hashlib.sha256(Path(eval_samples_path).read_bytes()).hexdigest(),
        "decoding": {"do_sample": False, "max_new_tokens": 150, "batch_size": batch_size},
    }

    # ── BASE MODEL ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING BASE MODEL (before fine-tuning)")
    logger.info("=" * 60)

    t0 = time.time()
    base_pipe = load_pipeline(base_model, use_4bit=use_4bit)
    results["base_revision"] = getattr(base_pipe.model.config, "_commit_hash", None)
    base_preds = batch_generate(base_pipe, samples, batch_size=batch_size)
    del base_pipe  # free memory before loading fine-tuned
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    base_metrics = run_metrics(
        base_preds,
        gold_queries,
        db_ids,
        skip_bertscore=skip_bertscore,
        skip_execution=skip_execution,
    )
    base_metrics["inference_time_s"] = round(time.time() - t0, 1)
    results["base_model_metrics"] = base_metrics

    # ── FINE-TUNED MODEL ──────────────────────────────────────────────────────
    if finetuned_model or adapter_dir:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING FINE-TUNED MODEL (after fine-tuning)")
        logger.info("=" * 60)

        t0 = time.time()
        ft_pipe = load_pipeline(
            base_model,
            adapter_dir=adapter_dir or finetuned_model,
            use_4bit=use_4bit,
        )
        ft_preds = batch_generate(ft_pipe, samples, batch_size=batch_size)
        del ft_pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        ft_metrics = run_metrics(
            ft_preds,
            gold_queries,
            db_ids,
            skip_bertscore=skip_bertscore,
            skip_execution=skip_execution,
        )
        ft_metrics["inference_time_s"] = round(time.time() - t0, 1)
        results["finetuned_model"] = finetuned_model or adapter_dir
        results["finetuned_model_metrics"] = ft_metrics

    results["predictions"] = [
        {
            **sample,
            "base_prediction": base_preds[i],
            "finetuned_prediction": ft_preds[i] if finetuned_model or adapter_dir else None,
        }
        for i, sample in enumerate(samples)
    ]

    # ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────
    _print_summary(results)

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results saved to {output_path}")

    return results


def _print_summary(results: dict):
    base = results["base_model_metrics"]
    ft = results.get("finetuned_model_metrics")

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def delta(b, f):
        if b is None or f is None:
            return ""
        d = f - b
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.4f})"

    print("\n" + "=" * 65)
    print("EVALFORGE — BEFORE vs AFTER RESULTS")
    print("=" * 65)

    header = f"{'Metric':<28} {'Base Model':>12}"
    if ft:
        header += f"  {'Fine-tuned':>12}  {'Δ':>10}"
    print(header)
    print("─" * 65)

    metrics = [
        ("Exact Match Accuracy", "exact_match"),
        ("Execution Accuracy", "execution_accuracy"),
        ("BLEU-4", "bleu4"),
        ("BERTScore F1", "bertscore_f1"),
    ]

    for label, key in metrics:
        b_val = base.get(key)
        row = f"{label:<28} {fmt(b_val):>12}"
        if ft:
            f_val = ft.get(key)
            row += f"  {fmt(f_val):>12}  {delta(b_val, f_val):>10}"
        print(row)

    print("─" * 65)
    print("\n📋 Copy the above table into your README.md\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvalForge: before/after evaluation harness")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--finetuned-model", default=None, help="HF Hub model ID of the fine-tuned model")
    parser.add_argument("--adapter-dir", default=None, help="Local path to LoRA adapter directory")
    parser.add_argument("--eval-samples", default="./data/processed/eval_samples.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n", type=int, default=200, help="Number of eval examples (default: 200)")
    parser.add_argument("--output", default="./outputs/eval_results.json")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--skip-execution", action="store_true", help="Skip execution accuracy (requires Spider DB files)"
    )
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Quick smoke test: n=10, skip BERTScore + execution"
    )
    args = parser.parse_args()

    if args.quick:
        args.n = 10
        args.skip_bertscore = True
        args.skip_execution = True

    run_eval(
        base_model=args.base_model,
        finetuned_model=args.finetuned_model,
        adapter_dir=args.adapter_dir,
        eval_samples_path=args.eval_samples,
        n=args.n,
        output_path=args.output,
        skip_bertscore=args.skip_bertscore,
        skip_execution=args.skip_execution,
        use_4bit=args.use_4bit,
        batch_size=args.batch_size,
    )
