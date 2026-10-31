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

Subsampling uses a seed-42 shuffle independently within each split. For a pilot, `--sample-size 1000 --epochs 1 --eval-n 200` selects 1,000 training and 200 validation rows before length filtering. Report the retained training count from the manifest. The selected validation artifact retains 200 raw prompts for paired generation; it is not the official test split.

## Two T4 GPUs on Kaggle

Use `notebooks/kaggle_training.ipynb`: it now defaults to the full training split, two epochs and two GPUs. Select GPU T4 x2, enable Internet, and enable file persistence before starting. Use Save Version → Save & Run All so a successful version retains `/kaggle/working` outputs. Download `evalforge-evidence.zip` from that version's Output tab and verify it before stopping/resetting or deleting the notebook. Persistence is not a substitute for downloading a backup.

Equivalent command after notebook setup:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.run_gpu --gpus 2 --epochs 2 --eval-n 200
```

The parent prepares data once, launches two torchrun workers, and evaluates once after both exit. Each worker loads the quantized model on its local GPU. Batch size 1 × accumulation 8 × 2 workers preserves an effective batch size of 16. Only rank zero writes the manifest/tokenizer. Trainer coordinates checkpoint saves. Gradient checkpointing uses non-reentrant mode with DDP unused-parameter detection disabled. The recorded peak allocation is rank zero only, not aggregate GPU memory.

This path completed a real two-T4 Kaggle run: 6,918 training examples, two configured epochs and 864 steps. See `reports/full-data/` and `notebooks/full_data_executed.ipynb`. Validate startup after changing dependencies or hardware. Do not assume an exact 2× speedup or identical numerical results to the single-GPU run. Paired generation evaluation remains a single process. For the old single-GPU behavior pass `--gpus 1` and expose only GPU 0.
