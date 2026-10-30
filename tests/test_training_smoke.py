"""Optional real Trainer/PEFT integration; no network or pretrained weights.

Run with: EVALFORGE_ML_TESTS=1 pytest tests/test_training_smoke.py -q
"""

import json
import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(os.getenv("EVALFORGE_ML_TESTS") != "1", reason="ML dependencies optional")


def test_prepare_train_save_adapter(tmp_path, monkeypatch):
    from datasets import Dataset, DatasetDict
    from peft import get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast
    from train.dataset_prep import prepare_dataset
    from train.qlora_config import get_lora_config
    import train.train as training

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tokenizer_backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "SELECT": 3, "1": 4}, unk_token="[UNK]")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    rows = Dataset.from_list([{"db_id": "demo", "question": "Count users", "query": "SELECT 1"}] * 2)
    raw = DatasetDict(train=rows, validation=rows)
    tables = [
        {
            "db_id": "demo",
            "table_names_original": ["users"],
            "column_names_original": [[0, "id"]],
            "column_types": ["number"],
            "primary_keys": [0],
            "foreign_keys": [],
        }
    ]
    metadata = tmp_path / "tables.json"
    metadata.write_text(json.dumps(tables))
    with (
        patch("train.dataset_prep.load_spider", return_value=raw),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        prepared = prepare_dataset(tables_path=str(metadata))
    row = prepared["train"][0]
    assert -100 in row["labels"]
    assert row["labels"][-1] != -100
    assert all(label == -100 or label == token for label, token in zip(row["labels"], row["input_ids"]))
    assert json.loads((tmp_path / "data/processed/eval_samples.json").read_text())[0]["db_id"] == "demo"
    config = Phi3Config(
        vocab_size=5,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=2,
        attention_dropout=0.0,
    )
    base = Phi3ForCausalLM(config)
    base.save_pretrained(tmp_path / "base")
    tokenizer.save_pretrained(tmp_path / "base")
    model = get_peft_model(base, get_lora_config())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    monkeypatch.setitem(training.TRAINING_CONFIG, "num_train_epochs", 1)
    monkeypatch.setitem(training.TRAINING_CONFIG, "gradient_accumulation_steps", 1)
    monkeypatch.setitem(training.TRAINING_CONFIG, "load_best_model_at_end", False)
    monkeypatch.setitem(training.TRAINING_CONFIG, "output_dir", str(tmp_path / "run"))
    with patch.object(training, "load_model_and_tokenizer", return_value=(model, tokenizer)):
        training.train(use_4bit=False)
    assert (tmp_path / "run/final/adapter_model.safetensors").exists()
    manifest = json.loads((tmp_path / "run/training_manifest.json").read_text())
    assert manifest["train_examples"] == 2
    assert manifest["metrics"]["train_loss"] > 0

    from eval.run_eval import run_eval

    report = run_eval(
        base_model=str(tmp_path / "base"),
        finetuned_model=None,
        adapter_dir=str(tmp_path / "run/final"),
        eval_samples_path=str(tmp_path / "data/processed/eval_samples.json"),
        n=2,
        output_path=str(tmp_path / "evaluation.json"),
        skip_bertscore=True,
        skip_execution=True,
    )
    assert len(report["predictions"]) == 2
    assert len(report["dataset_sha256"]) == 64
    assert report["finetuned_model_metrics"]["execution_accuracy"] is None
