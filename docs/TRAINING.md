# Training runbook

1. Use a Linux CUDA GPU environment and Python 3.11. Clone this repository, install `requirements-ml.txt`, and confirm `torch.cuda.is_available()`.
2. Download the original Spider release from https://yale-lily.github.io/spider. Put its `tables.json` at `data/spider/tables.json`, and the SQLite directories at `data/spider_databases/<db_id>/<db_id>.sqlite`. These are not included in Git. Review the dataset terms before redistributing it.
3. Run `python -m train.dataset_prep --tables-path data/spider/tables.json`. The Hugging Face `xlangai/spider` questions are joined with this metadata. Every example retains `db_id`; train and validation remain separate. Overlength examples are excluded. Delete `data/processed` before changing the tokenizer, schema metadata, or length configuration.
4. Run `python -m train.train`. Optional W&B tracking uses an exported `WANDB_API_KEY`. Without it, training stays local. `--sample-size 50` makes a small dataset run; `--no-4bit` is for debugging and uses full-precision weights, not QLoRA.
5. Preserve `outputs/phi3-sql-qlora/training_manifest.json`, trainer logs, adapter configuration, tokenizer, Git commit, hardware details, package versions, and dataset hashes with the experiment. Record peak GPU memory and elapsed training time from the actual run.
6. Evaluate the adapter against the same base model and held-out samples. Keep `outputs/eval_results.json`; review incorrect predictions before drawing conclusions.
7. Update `model_card.md` with measured values and limitations before publishing. Export `HF_TOKEN` securely and run `python -m train.push_to_hub --adapter-dir outputs/phi3-sql-qlora/final --repo-id YOUR_ACCOUNT/evalforge-phi3-sql`. No upload occurs unless explicitly requested through this command or the training flag.

Configuration lives in `train/qlora_config.py`: seed 42, three epochs, LoRA rank 16 / alpha 32, dropout .05, 2048-token filtering, batch 4 with accumulation 4. These are initial settings, not empirically optimized hyperparameters. Compare at least one ablation (for example rank 8 vs 16) before describing them as tuned.

Completion-only labels mask the system instruction, question, and schema with -100. The collator pads labels with -100. Training uses Hugging Face Trainer with an already constructed PEFT model; it does not depend on changing TRL interfaces.

## Automated GPU experiment

`CUDA_VISIBLE_DEVICES=0 python -m scripts.run_gpu` downloads the Spider database mirror at a pinned revision, checks its SHA-256, prepares data, runs three epochs with batch 1 / accumulation 16 on one GPU, and evaluates 200 held-out examples. The archive mirror credits the [original Yale distribution](https://yale-lily.github.io/spider); Spider is CC BY-SA 4.0. The manifest records the archive and schema hashes, Git revision, package versions, elapsed time, GPU, and peak allocated memory. Use `--sample-size 50 --epochs 1 --eval-n 10` only for a debugging experiment and report that scope explicitly.

On notebook hosts with Python 3.13, create an isolated Python 3.11 environment (for example with uv) rather than downgrading the host's packages. Put the environment and model cache under the notebook's temporary directory so only experiment outputs are saved.
