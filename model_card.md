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

# EvalForge Phi-3 SQL — full-data adapter

NF4 QLoRA adaptation of Phi-3-mini-4k-instruct to schema-conditioned Text-to-SQL. Trained on the full retained Spider training split using two Tesla T4 GPUs and two configured epochs. The released adapter is the best validation-loss checkpoint (step 300), reloaded after the complete 864-step run.

## Measured development results

Paired greedy generation on the same first 200 Spider validation questions; batch size 4; maximum 150 new tokens. Independently replayed on CPU.

| Metric | Base Phi-3 | QLoRA adapter | Change |
| --- | ---: | ---: | ---: |
| Normalized exact match | 14.5% | 23.0% | +8.5 pp |
| SQLite execution accuracy | 50.0% | 59.0% | +9.0 pp |
| BLEU-4 (0–1) | 0.3076 | 0.6979 | +0.3903 |
| BERTScore | Not measured | Not measured | Skipped |

88 both correct; 30 base errors fixed; 12 regressions; 70 both incorrect. No gold execution errors. These are custom development metrics, not an official Spider test score or proof of SQL equivalence. Raw outputs are scored without stripping prose or Markdown. Single-database equality can reward accidentally matching empty results; projection order matters. The 200 questions are the first 200 validation rows, include paraphrases, and are not independent observations. Validation loss also selected the checkpoint, so these are not untouched test results. Only one seed was run; no statistical-significance claim is made.

## Training configuration

| Field | Value |
| --- | --- |
| Base revision | `f39ac1d28e925b323eae81227eaba4464caced4e` |
| Selected / retained training rows | 7,000 / 6,918 |
| Validation / paired evaluation rows | 1,034 / 200 |
| Configured / reported epochs | 2 / 1.998265 |
| Completed steps / released checkpoint | 864 / 300 (best validation loss) |
| Hardware | 2 × Tesla T4; DDP |
| Training loop / stage wall time | 4.50 / 4.52 hours |
| Peak allocation | 4.35 GiB on rank zero only |
| Trainable parameters | 25,165,824 (~0.65%) |
| Mean training loss | 0.110036 |
| Quantization | NF4, double quantization, FP16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Modules | qkv_proj, o_proj, gate_up_proj, down_proj |
| Per-GPU batch / accumulation / effective batch | 1 / 8 / 16 |
| Learning rate / seed | 2e-4 / 42 |
| Objective | Completion-only causal LM; prompt and padding labels masked |

The manifest records package versions, source commit, data archive and schema hashes. See [experiment analysis](docs/EXPERIMENT.md) and `reports/full-data/`. The pilot is preserved separately; its evaluation cohort differs.

## Download and use

Get `evalforge-v0.2.0-full-data.zip` from the [GitHub release](https://github.com/Navaneeth-007/EvalForge/releases/tag/v0.2.0-full-data). Verify `SHA256SUMS.txt` and extract the ZIP. It contains `adapter/`, tokenizer files, reports and this card. Base weights download separately from Hugging Face.

```bash
pip install -r requirements-ml.txt
EVALFORGE_MODE=model USE_4BIT=true ADAPTER_MODEL=/absolute/path/to/adapter \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CUDA hardware is required for the quantized serving configuration. The public Render site shows all 200 recorded comparisons; it does not run live Phi-3 inference. No Hugging Face adapter repository is claimed.

## Intended use and limitations

Research, education and portfolio review. Review queries before using them against real data. Generated SQL is never executed by the public API. This is a reproducible portfolio deployment, not an SLA-backed database product. Failures include joins, filters, projections, subqueries and output formatting. There is no independent test-set score.

Base-model terms apply separately. Spider-derived examples and reports: Yu et al., EMNLP 2018, CC BY-SA 4.0, https://yale-lily.github.io/spider. Repository code is MIT licensed.
