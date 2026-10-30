"""
QLoRA fine-tuning for Phi-3-mini on Text-to-SQL.

Run on Kaggle (T4 GPU) — see notebooks/kaggle_training.ipynb for the full notebook.
Run locally for debugging with --sample-size 50.

Usage:
    python train/train.py                        # full run (Kaggle)
    python train/train.py --sample-size 50       # debug run (CPU/MPS)
"""

import logging
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from train.qlora_config import TRAINING_CONFIG, get_bnb_config, get_lora_config
from train.dataset_prep import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

set_seed(42)


def setup_wandb():
    """Initialise Weights & Biases run."""
    wandb.init(
        project="evalforge-phi3-sql",
        name=f"qlora-r16-ep{TRAINING_CONFIG['num_train_epochs']}",
        config=TRAINING_CONFIG,
        tags=["phi3", "qlora", "text-to-sql", "spider"],
    )
    logger.info("W&B run initialised")


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """
    Load Phi-3-mini with optional 4-bit quantization.

    use_4bit=False is used for local debug runs where bitsandbytes isn't available.
    """
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        trust_remote_code=False,
        device_map="auto",
    )

    if use_4bit:
        model_kwargs["quantization_config"] = get_bnb_config()
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_4bit:
        # Required before applying LoRA to a quantized model
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # Apply LoRA adapters
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_training_args(output_dir: str, use_fp16: bool = True) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from TRAINING_CONFIG."""
    cfg = TRAINING_CONFIG
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"] if use_fp16 else "adamw_torch",
        fp16=use_fp16 and torch.cuda.is_available(),
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        group_by_length=cfg["group_by_length"],
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=f"qlora-phi3-sql-ep{cfg['num_train_epochs']}",
        dataloader_pin_memory=False,
    )


def train(
    sample_size: int | None = None,
    use_4bit: bool = True,
    push_to_hub: bool = False,
):
    """
    Main training function.

    Args:
        sample_size: if set, sub-samples dataset for quick debug
        use_4bit: set False for CPU/MPS debug (skips bitsandbytes)
        push_to_hub: if True, pushes adapter to HF Hub after training
    """
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU; use --no-4bit only for a small debug run")
    cfg = TRAINING_CONFIG
    output_dir = cfg["output_dir"]

    # W&B
    if os.getenv("WANDB_API_KEY"):
        setup_wandb()
    else:
        logger.warning("WANDB_API_KEY not set — skipping W&B logging")
        os.environ["WANDB_DISABLED"] = "true"

    # Dataset
    processed_dir = "./data/processed"
    if Path(processed_dir).exists() and not sample_size:
        logger.info(f"Loading pre-processed dataset from {processed_dir}")
        dataset = load_from_disk(processed_dir)
    else:
        logger.info("Preparing dataset from scratch...")
        dataset = prepare_dataset(
            output_dir=processed_dir,
            sample_size=sample_size,
        )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], use_4bit=use_4bit)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    training_args = build_training_args(output_dir, use_fp16=use_4bit)
    model.config.use_cache = False
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, processing_class=tokenizer, data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    result = trainer.train()
    import json
    manifest = {"base_model": cfg["model_name"], "seed": 42, "config": cfg,
                "train_examples": len(train_dataset), "validation_examples": len(eval_dataset),
                "metrics": result.metrics, "quantized": use_4bit,
                "base_revision": getattr(model.config, "_commit_hash", None)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save best model
    logger.info(f"Saving adapter to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Push to Hub
    if push_to_hub:
        hub_id = os.getenv("HF_MODEL_REPO") or cfg.get("hub_model_id")
        if hub_id:
            logger.info(f"Pushing to HuggingFace Hub: {hub_id}")
            trainer.model.push_to_hub(hub_id)
            tokenizer.push_to_hub(hub_id)
        else:
            logger.warning("HF_MODEL_REPO not set — skipping hub push")

    logger.info("Training complete.")
    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Phi-3-mini on Text-to-SQL")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Sub-sample for debug (e.g. 50)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (for CPU/MPS debug)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push adapter to HF Hub after training")
    args = parser.parse_args()

    train(
        sample_size=args.sample_size,
        use_4bit=not args.no_4bit,
        push_to_hub=args.push_to_hub,
    )
