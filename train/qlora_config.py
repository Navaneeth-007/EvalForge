"""
QLoRA configuration for Phi-3-mini fine-tuning on Text-to-SQL.

Design choices:
- 4-bit NormalFloat (NF4) quantization — best quality/memory tradeoff for QLoRA
- r=16, alpha=32 — conservative rank, doubles alpha (standard starting point)
- Target modules: fused qkv_proj and gate_up_proj, plus o_proj and down_proj
- Dropout=0.05 — light regularisation for a task-specific fine-tune
"""

from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import torch


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — better than fp4 for LLMs
        bnb_4bit_compute_dtype=torch.float16, # compute in fp16 during forward pass
        bnb_4bit_use_double_quant=True,       # double quantization saves ~0.4 bits/param
    )


def get_lora_config() -> LoraConfig:
    """LoRA adapter config."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # rank — higher = more params, more expressiveness
        lora_alpha=32,           # scaling factor (alpha/r = 2 is standard)
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        inference_mode=False,
    )


# Training hyperparameters — tuned for Spider on a single T4 (Kaggle)
TRAINING_CONFIG = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "dataset_name": "spider",
    "output_dir": "./outputs/phi3-sql-qlora",
    "hub_model_id": None,           # set via env: HF_MODEL_REPO e.g. "username/evalforge-phi3-sql"

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,   # effective batch = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "max_seq_length": 2048,
    "max_grad_norm": 0.3,

    # Logging
    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",

    # Optimisation
    "optim": "paged_adamw_32bit",    # paged optimizer — reduces GPU memory spikes
    "fp16": True,
    "group_by_length": True,         # pack similar-length sequences → faster training
}
