# Evaluation protocol

Use the identical sample file for the frozen base and its trained adapter. Greedy decoding uses 150 new tokens. The report records the SHA-256 of that file and both predictions per example. `--finetuned-model` means a Hub **adapter** repository; merged models are not supported by that argument.

- Normalized exact match lowercases unquoted tokens and collapses whitespace while preserving quoted values and identifiers. This is a custom string metric, **not the official Spider structural exact-match evaluator**.
- Execution accuracy compares result rows, preserving duplicate rows and respecting row order when the gold query has ORDER BY. SQLite is opened read-only, an authorizer rejects writes and attachments, a progress handler bounds execution time, and output is capped at 10,000 rows. Query failures count as incorrect. Missing individual databases count as unavailable/incorrect; an entirely missing database directory returns null rather than claiming a score.
- BLEU-4 measures token overlap. BERTScore measures embedding similarity. Neither proves SQL correctness. BERTScore requires a separate encoder download; skip it for a lighter run.

Execution matching on a single database can reward accidentally equivalent queries, especially for empty results. Inspect errors and use the official Spider test-suite evaluator for benchmark-grade claims. The ORDER BY heuristic and string normalization are intentionally simpler than a full SQL AST evaluator.

The default artifact samples the first 200 development examples, not the full benchmark. Report the number of examples, skipped metrics, split, selection rule, decoding settings, model revision, and hardware with every score. Database-separated validation does not protect against benchmark contamination in pretrained weights.

Promote results only after a real training run: save the manifest and report, publish the matching adapter, fill in the model card, and link the immutable artifacts. Never relabel walkthrough fixtures as model output.
