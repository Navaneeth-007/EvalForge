"""
Push the trained adapter + model card to Hugging Face Hub.

Usage:
    python train/push_to_hub.py --adapter-dir ./outputs/phi3-sql-qlora/final \
                                 --repo-id your-username/evalforge-phi3-sql
"""

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def push_adapter(adapter_dir: str, repo_id: str, private: bool = False):
    """Push LoRA adapter weights and tokenizer to HF Hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable not set")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    logger.info(f"Repo ready: https://huggingface.co/{repo_id}")

    # Push adapter files
    logger.info(f"Pushing adapter from {adapter_dir} → {repo_id}")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        commit_message="Add QLoRA adapter — Phi-3-mini Text-to-SQL",
    )

    # Push model card
    model_card_path = Path("model_card.md")
    if model_card_path.exists():
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card",
        )
        logger.info("Model card pushed")

    logger.info(f"Done! View at: https://huggingface.co/{repo_id}")


def merge_and_push(
    base_model: str,
    adapter_dir: str,
    repo_id: str,
    private: bool = False,
):
    """
    Merge LoRA adapter into base model weights and push the full model.
    Use this if you want users to be able to run the model without PEFT installed.
    Warning: requires ~8GB disk space for the merged weights.
    """
    import torch

    logger.info("Loading base model for merge...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    logger.info("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    merged_dir = "./outputs/merged_model"
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to {merged_dir}")

    push_adapter(merged_dir, repo_id, private=private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="./outputs/phi3-sql-qlora/final")
    parser.add_argument("--repo-id", required=True,
                        help="e.g. your-username/evalforge-phi3-sql")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base before pushing (larger upload)")
    args = parser.parse_args()

    if args.merge:
        from train.qlora_config import TRAINING_CONFIG
        merge_and_push(
            base_model=TRAINING_CONFIG["model_name"],
            adapter_dir=args.adapter_dir,
            repo_id=args.repo_id,
            private=args.private,
        )
    else:
        push_adapter(args.adapter_dir, args.repo_id, private=args.private)
