# Full-data evidence

`eval_results.json`: original paired generation report. `eval_results_verified.json`: independent CPU replay using the published SQLite evaluator. `eval_samples.json`: exact evaluation prompts, SHA256 matched to report. `training_manifest.json`: GPU run metadata. `trainer_state.json`: completed steps, loss history and selected checkpoint.

Final released weights match checkpoint 300, selected by validation loss after 864 training steps. Full original Kaggle output archive (including retained optimizer checkpoints) and smaller inference bundle are backed up in GitHub release v0.2.0-full-data. No model weights are stored in Git history.

Spider-derived content: Yu et al., EMNLP 2018, CC BY-SA 4.0, https://yale-lily.github.io/spider.
