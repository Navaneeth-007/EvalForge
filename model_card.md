---
language:
- en
base_model: microsoft/Phi-3-mini-4k-instruct
datasets:
- xlangai/spider
tags:
- text-to-sql
- qlora
- peft
pipeline_tag: text-generation
---

# EvalForge Phi-3 SQL — pilot adapter

A measured QLoRA pilot adapting Phi-3-mini-4k-instruct to schema-conditioned Text-to-SQL. This is a **one-epoch pilot**, not a full-data training run or an official Spider benchmark submission.

## Measured results

Paired evaluation on the same 200 seed-42 development examples, with greedy decoding, batch size 4, and at most 150 new tokens:

| Metric | Base Phi-3 | QLoRA adapter | Change |
| --- | ---: | ---: | ---: |
| Normalized exact match | 11.0% | 26.0% | +15.0 pp |
| SQLite execution accuracy | 53.0% | 57.5% | +4.5 pp |
| BLEU-4 (0–1) | 0.3878 | 0.6547 | +0.2669 |
| BERTScore | Not measured | Not measured | Skipped |

Execution results: 85 both correct; 30 base errors fixed; 21 base-correct predictions regressed; 64 both incorrect. There were no gold-query execution errors. A 9-question net gain on 200 development questions does not establish generalization or statistical significance.

Scores evaluate raw generated SQL: Markdown fences and prose count as failures. Normalized string match is a custom metric, not the official Spider structural evaluator. Single-database result equality can reward accidentally matching or empty results. See `docs/EVALUATION.md` and the complete reports in `reports/pilot/`.

## Actual training run

| Field | Measured value / configuration |
| --- | --- |
| Base revision | `f39ac1d28e925b323eae81227eaba4464caced4e` |
| Selected / retained training rows | 1,000 / 987 |
| Filtered rows | 13 exceeding 2,048 tokens |
| Validation / evaluation rows | 200 / 200 |
| Configured epochs / reported epoch fraction | 1 / 0.988855 |
| Optimizer steps | 61 |
| GPU | One Tesla T4 |
| Training loop time | 29.49 minutes |
| Training stage wall time (including load/save) | 29.65 minutes |
| Peak CUDA allocated memory | 4.22 GiB |
| Trainable adapter parameters | 25,165,824 (approximately 0.65%) |
| Final mean training loss | 0.293330 |
| Method | NF4 4-bit, double quantization, FP16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Modules | qkv_proj, o_proj, gate_up_proj, down_proj |
| Batch / accumulation / learning rate | 1 / 16 / 2e-4 |
| Objective | Completion-only causal LM; prompt and padding labels masked |
| Seed / selection | 42 / seeded shuffle within each original split |

The Trainer reports approximately 0.989 completed epoch because of gradient accumulation at the end of the selected split. No early-stopping or hyperparameter-optimization claim is made. The manifest records the exact package environment and source commit.

## Use

Download the adapter and evidence archive from the repository's `v0.1.0-pilot` release and extract it. On a CUDA machine:

```bash
pip install -r requirements-ml.txt
EVALFORGE_MODE=model USE_4BIT=true ADAPTER_MODEL=/absolute/path/to/adapter \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The public Render website shows saved predictions and scores. It does not host live Phi-3 inference. Weights are published as a GitHub release asset; no Hugging Face adapter repository is claimed.

## Intended use and limitations

Research, education, and portfolio review of model adaptation. Queries require review before use against real data. This pilot is limited by its training subset, a single seed, 200 development examples, output-format sensitivity, and a simplified SQLite metric. Common failures include incorrect joins, missing filters, projection order, and scalar subqueries used where multiple rows are required. There is no independent test-set score or production database-access capability.

The base model and dataset retain their own terms. Spider attribution: Yu et al., EMNLP 2018, CC BY-SA 4.0, https://yale-lily.github.io/spider. Code is licensed separately by this repository.
