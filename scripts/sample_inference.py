"""
Quick CLI inference test — no API server needed.
Good for checking the model works after training.

Usage:
    # Test base model
    python scripts/sample_inference.py \
        --question "Which customers spent more than $1000?" \
        --schema "CREATE TABLE customers (id INT, name TEXT); CREATE TABLE orders (id INT, customer_id INT, total REAL);"

    # Test fine-tuned (local adapter)
    python scripts/sample_inference.py \
        --adapter-dir ./outputs/phi3-sql-qlora/final \
        --question "Which customers spent more than $1000?" \
        --schema "CREATE TABLE customers (id INT, name TEXT); CREATE TABLE orders (id INT, customer_id INT, total REAL);"

    # Test fine-tuned (HF Hub)
    python scripts/sample_inference.py \
        --model your-username/evalforge-phi3-sql \
        --question "What is the average salary by department?"
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from train.dataset_prep import format_prompt
from train.qlora_config import TRAINING_CONFIG

DEFAULT_SCHEMA = (
    "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, hire_date TEXT);\n"
    "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL);"
)

DEFAULT_QUESTIONS = [
    "What is the average salary by department?",
    "Who are the top 3 highest-paid employees?",
    "How many employees were hired after 2022?",
    "Which departments have a budget over 500000?",
]


def load_model(model_name: str, adapter_dir: str | None, no_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if not no_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        print(f"✓ Adapter merged from {adapter_dir}")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe


def infer(pipe, question: str, schema: str) -> tuple[str, float]:
    prompt = format_prompt(question=question, schema=schema, sql=None)
    t0 = time.time()
    output = pipe(prompt, return_full_text=False)
    latency = time.time() - t0
    raw = output[0]["generated_text"].strip()
    sql = raw.split("<|end|>")[0].split("\n\n")[0].strip()
    return sql, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=TRAINING_CONFIG["model_name"])
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="Run all default questions and print a table")
    args = parser.parse_args()

    print(f"\nLoading: {args.model}")
    pipe = load_model(args.model, args.adapter_dir, args.no_4bit)
    print("Model ready.\n")

    if args.demo:
        print("="*60)
        print("DEMO — Default Questions")
        print("="*60)
        print(f"Schema:\n{args.schema}\n")
        for q in DEFAULT_QUESTIONS:
            sql, latency = infer(pipe, q, args.schema)
            print(f"Q: {q}")
            print(f"→ {sql}")
            print(f"   [{latency*1000:.0f}ms]\n")
    elif args.question:
        sql, latency = infer(pipe, args.question, args.schema)
        print(f"Q: {args.question}")
        print(f"→ {sql}")
        print(f"   [{latency*1000:.0f}ms]")
    else:
        # Interactive mode
        print(f"Schema:\n{args.schema}\n")
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                sql, latency = infer(pipe, q, args.schema)
                print(f"SQL: {sql}")
                print(f"[{latency*1000:.0f}ms]\n")


if __name__ == "__main__":
    main()
