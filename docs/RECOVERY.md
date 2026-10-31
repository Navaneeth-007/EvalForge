# Recover the project after deleting local files

The source, documentation, tests, training notebooks, measured reports and public demo assets live in the [GitHub repository](https://github.com/Navaneeth-007/EvalForge). Model weights are release assets, not Git objects.

## Restore source and the trained model

1. Clone `https://github.com/Navaneeth-007/EvalForge.git`.
2. Open [release v0.2.0-full-data](https://github.com/Navaneeth-007/EvalForge/releases/tag/v0.2.0-full-data).
3. Download `evalforge-v0.2.0-full-data.zip` and `SHA256SUMS.txt` for inference. Verify the archive's SHA256 matches the checksum file. Extract it; `adapter/` contains the trained adapter and tokenizer.
4. Follow the model card for CUDA inference. Base Phi-3 weights and Python dependencies download separately; base weights are not in the archive.
5. For the lightweight recorded viewer, install `requirements.txt` and run `uvicorn app.main:app`. No adapter download or GPU is required for that viewer.

## Preserve training checkpoints

`kaggle-output-full-data.zip` is the complete original Kaggle output archive (649,341,157 bytes), including the final adapter, checkpoints 300 and 864, optimizer/scheduler/RNG state, manifest, evaluation prompts and original report. Download it only when you need the full recovery backup. It has its own entry in `SHA256SUMS.txt`; its original SHA256 is `50f2a7e5fcb7b95b0b2a9bf8ca144418b910f99c9feb615d64e0d044b68e74f9`.

The final adapter matches the best-validation checkpoint 300. The run completed 864 steps. Retained optimizer checkpoints allow inspection or a deliberately configured resume; the standard runner starts a fresh run and does not automatically resume. Only load checkpoint pickle files from a trusted source.

Training data can be reconstructed by `scripts.run_gpu` from the pinned, checksum-verified public archive. The exact 200 evaluation prompts are also tracked under `reports/full-data/`. The executed two-GPU notebook is tracked under `notebooks/full_data_executed.ipynb`. The prior single-GPU full-data notebook is archived as `notebooks/single_gpu_unretained.ipynb`; its weights were lost and are not the released adapter.

Deleting local files does not remove GitHub releases or the deployed viewer. Keep your GitHub account and repository available; remote hosting is not an immutable archival guarantee. A separate personal backup is useful if you later delete the repository or release.
