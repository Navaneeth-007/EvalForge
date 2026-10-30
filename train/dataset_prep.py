"""
Dataset preparation for Text-to-SQL fine-tuning.

Source: Spider benchmark (https://huggingface.co/datasets/spider)
Format: instruction-following prompt → SQL query

Prompt template (Phi-3 chat format):
    <|user|>
    Given the following database schema:
    {schema}

    Write a SQL query to answer:
    {question}
    <|end|>
    <|assistant|>
    {sql}
    <|end|>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural language question, "
    "write a correct SQL query. Return only the SQL query, no explanation."
)

def format_prompt(question: str, schema: str, sql: Optional[str] = None) -> str:
    """
    Build a Phi-3 chat-format prompt.
    If sql is None, returns prompt only (for inference).
    If sql is provided, returns full prompt + completion (for training).
    """
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n"
        f"Database schema:\n{schema}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
    if sql is not None:
        prompt += f"{sql}<|end|>"
    return prompt


# ── Schema extraction ─────────────────────────────────────────────────────────

def extract_schema(tables: dict) -> str:
    """
    Convert Spider table metadata into a compact CREATE TABLE schema string.

    Spider provides:
      tables[i] = {
        "table_names_original": [...],
        "column_names_original": [[-1, "*"], [table_idx, col_name], ...],
        "column_types": [...],
        "primary_keys": [...],
        "foreign_keys": [[col_idx, col_idx], ...]
      }
    """
    table_names = tables["table_names_original"]
    col_names = tables["column_names_original"]   # [table_idx, col_name]
    col_types = tables["column_types"]
    primary_keys = set(tables["primary_keys"])

    # Group columns by table
    table_cols: dict[int, list[str]] = {i: [] for i in range(len(table_names))}
    for idx, (tbl_idx, col_name) in enumerate(col_names):
        if tbl_idx == -1:
            continue
        col_type = col_types[idx].upper()
        pk_marker = " PRIMARY KEY" if idx in primary_keys else ""
        table_cols[tbl_idx].append(f"  {col_name} {col_type}{pk_marker}")

    # Build FK annotations (simple inline comment)
    fk_lines = []
    for from_col_idx, to_col_idx in tables["foreign_keys"]:
        from_tbl_idx, from_col = col_names[from_col_idx]
        to_tbl_idx, to_col = col_names[to_col_idx]
        from_tbl = table_names[from_tbl_idx]
        to_tbl = table_names[to_tbl_idx]
        fk_lines.append(f"-- FK: {from_tbl}.{from_col} → {to_tbl}.{to_col}")

    schema_parts = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = table_cols.get(tbl_idx, [])
        cols_str = ",\n".join(cols)
        schema_parts.append(f"CREATE TABLE {tbl_name} (\n{cols_str}\n);")

    if fk_lines:
        schema_parts.append("\n".join(fk_lines))

    return "\n\n".join(schema_parts)


# ── Dataset loading + formatting ──────────────────────────────────────────────

def load_spider() -> DatasetDict:
    """Download Spider from Hugging Face."""
    logger.info("Loading Spider dataset from Hugging Face...")
    from datasets import load_dataset

    dataset = load_dataset("xlangai/spider")
    logger.info(
        f"Train: {len(dataset['train'])} examples | "
        f"Validation: {len(dataset['validation'])} examples"
    )
    return dataset


def format_example(example: dict, schemas: dict | None = None) -> dict:
    """Format a single Spider example into a training prompt."""
    db_id = example["db_id"]
    if not schemas or db_id not in schemas:
        raise ValueError(f"Missing schema metadata for {db_id}; supply Spider tables.json")
    schema = extract_schema(schemas[db_id])
    question = example["question"]
    sql = example["query"]
    text = format_prompt(question=question, schema=schema, sql=sql)
    return {"text": text, "question": question, "schema": schema, "sql": sql, "db_id": db_id}


def prepare_dataset(
    tokenizer_name: str = "microsoft/Phi-3-mini-4k-instruct",
    max_seq_length: int = 2048,
    output_dir: str = "./data/processed",
    tables_path: str = "./data/spider/tables.json",
    sample_size: Optional[int] = None,   # set for quick debug runs
) -> DatasetDict:
    """
    Full pipeline: load Spider → format prompts → tokenize → save.
    Returns the tokenized DatasetDict.
    """
    from transformers import AutoTokenizer

    with open(tables_path) as source:
        schemas = {table["db_id"]: table for table in json.load(source)}
    raw = load_spider()

    # Optional sub-sample for fast iteration
    if sample_size:
        raw["train"] = raw["train"].select(range(min(sample_size, len(raw["train"]))))
        raw["validation"] = raw["validation"].select(
            range(min(max(1, sample_size // 5), len(raw["validation"])))
        )
        logger.info(f"Sub-sampled to {len(raw['train'])} train / {len(raw['validation'])} val")

    # Format prompts
    logger.info("Formatting prompts...")
    formatted = raw.map(
        lambda example: format_example(example, schemas),
        remove_columns=raw["train"].column_names,
        desc="Formatting",
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training

    # Mask the entire prompt; train only on SQL completion tokens.
    def tokenize(example):
        prompt = format_prompt(example["question"], example["schema"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(example["sql"] + "<|end|>", add_special_tokens=False)["input_ids"]
        ids = prompt_ids + completion_ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": [-100] * len(prompt_ids) + completion_ids}

    tokenized = formatted.map(tokenize, remove_columns=formatted["train"].column_names)
    # Drop overlength examples instead of cutting schema or removing all supervised tokens.
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) <= max_seq_length)
    if not len(tokenized["train"]) or not len(tokenized["validation"]):
        raise ValueError("No examples remain after length filtering; increase max_seq_length")

    # Save to disk
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")

    # Also save a few raw examples for the eval harness
    _save_eval_samples(formatted["validation"], output_dir)

    return tokenized


def _save_eval_samples(val_dataset, output_dir: str, n: int = 200):
    """Save n raw (non-tokenized) validation examples for evaluation."""
    samples = [val_dataset[i] for i in range(min(n, len(val_dataset)))]
    out_path = Path(output_dir) / "eval_samples.json"
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {len(samples)} eval samples to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Spider dataset for fine-tuning")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--tables-path", default="./data/spider/tables.json")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sub-sample for fast debug (e.g. 100)")
    args = parser.parse_args()

    prepare_dataset(output_dir=args.output_dir, sample_size=args.sample_size, tables_path=args.tables_path)
