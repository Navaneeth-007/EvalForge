---
language:
- en
base_model: microsoft/Phi-3-mini-4k-instruct
tags:
- text-to-sql
- qlora
- peft
pipeline_tag: text-generation
---

# EvalForge Phi-3 SQL — experiment specification

**Status: training pending. No adapter weights or verified benchmark results are published.** This card describes the planned experiment; it is not evidence of an already trained model.

Base: microsoft/Phi-3-mini-4k-instruct. Method: QLoRA with NF4, rank 16, alpha 32, dropout .05, fused Phi-3 attention and MLP projections. Objective: assistant completion-only causal language modeling on Spider Text-to-SQL examples with database schema context.

Initial configuration: seed 42, three epochs, learning rate 2e-4, effective batch 16, 2048-token maximum with overlength filtering. Training and validation counts, GPU type, measured duration, peak memory, model revision, dataset hash, and results must be added from actual artifacts after running the experiment.

Evaluation compares frozen base and adapter with identical development prompts and greedy decoding. Custom normalized string match, read-only SQLite execution, BLEU, and optional BERTScore are described in `docs/EVALUATION.md`. None of these have measured values yet. Do not cite a fine-tuning improvement until evidence is available.

Intended use: English-language SQL generation research and a portfolio demonstration of the adaptation workflow. Limitations include schema truncation/context limits, incorrect joins and aggregations, sensitivity to prompt wording, benchmark contamination, and lack of validation on private enterprise schemas. Generated queries require review before use. The public walkthrough uses authored SQL fixtures and performs no model inference.

The base model and dataset have their own terms. Review them before training, distributing adapters, or deploying model inference. Repository code licensing does not override upstream terms.
