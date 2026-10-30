# Phi-3 QLoRA pilot — measured artifacts

This directory contains the actual GPU run metadata, all 200 evaluation prompts and paired predictions, and independent CPU replay scores. `eval_results_verified.json` is the score source used by the website and README. `eval_results.json` preserves the original GPU evaluation report. The predictions are unchanged by replay.

- Training source revision is recorded in `training_manifest.json`.
- Scoring source revision is recorded in `eval_results_verified.json`.
- The model revision, dataset archive revision, dataset SHA-256, and package versions are included.
- Adapter weights are distributed separately in the GitHub pilot release; they are not checked into Git.

Spider questions, schemas, and gold queries are derived from the [Spider dataset](https://yale-lily.github.io/spider), by Yu et al., EMNLP 2018, distributed under CC BY-SA 4.0. This dataset-derived report content retains that attribution and license. Repository code licensing is separate.
