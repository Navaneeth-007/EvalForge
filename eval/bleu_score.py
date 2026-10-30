"""
BLEU-4 score for Text-to-SQL evaluation.

Uses sacrebleu (corpus-level BLEU) which is the standard, reproducible
implementation. Treats SQL tokens as words.

BLEU measures n-gram overlap between prediction and gold.
For SQL, it captures surface-level generation quality across training.
"""

try:
    import sacrebleu

    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False


def tokenise_sql(sql: str) -> str:
    """
    Tokenise SQL for BLEU by adding spaces around punctuation.
    sacrebleu handles word splitting; we just ensure SQL symbols are separated.
    """
    import re

    sql = sql.strip().lower()
    # Space around SQL punctuation
    sql = re.sub(r"([(),;=<>!*])", r" \1 ", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def compute_bleu(
    predictions: list[str],
    gold_queries: list[str],
) -> dict:
    """
    Compute corpus-level BLEU-4 between predictions and gold SQL.

    Returns:
        {
            "bleu4": float,
            "bleu_score_object": sacrebleu.BLEUScore (if sacrebleu available)
        }
    """
    if not SACREBLEU_AVAILABLE:
        return {
            "bleu4": None,
            "error": "sacrebleu not installed. Run: pip install sacrebleu",
        }

    # sacrebleu expects: corpus_bleu(hypotheses, [references])
    hypotheses = [tokenise_sql(p) for p in predictions]
    references = [[tokenise_sql(g) for g in gold_queries]]

    result = sacrebleu.corpus_bleu(hypotheses, references)

    return {
        "bleu4": round(result.score / 100, 4),  # convert % → fraction
        "bleu4_percent": round(result.score, 2),
        "precisions": result.precisions,
        "bp": round(result.bp, 4),
        "ratio": round(result.ratio, 4),
    }


def compute_sentence_bleu(
    prediction: str,
    gold: str,
) -> float:
    """Sentence-level BLEU for a single example (for per-example breakdown)."""
    if not SACREBLEU_AVAILABLE:
        return None

    hyp = tokenise_sql(prediction)
    ref = [tokenise_sql(gold)]
    result = sacrebleu.sentence_bleu(hyp, ref)
    return round(result.score / 100, 4)
