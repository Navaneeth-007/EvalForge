"""
BERTScore F1 for Text-to-SQL evaluation.

BERTScore measures semantic similarity between prediction and gold using
contextual embeddings. For SQL, it captures paraphrases that exact match misses
(e.g. equivalent subqueries, different aliasing).

Uses microsoft/deberta-xlarge-mnli as the reference model (best correlation
with human judgement per the BERTScore paper).

Falls back to roberta-large for faster computation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from bert_score import score as bert_score_fn, BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


def compute_bertscore(
    predictions: list[str],
    gold_queries: list[str],
    model_type: str = "roberta-large",   # swap to deberta-xlarge-mnli for best quality
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict:
    """
    Compute BERTScore F1 between predictions and gold SQL.

    Args:
        predictions: list of predicted SQL strings
        gold_queries: list of gold SQL strings
        model_type: embedding model (roberta-large is fastest; deberta-xlarge-mnli is best)
        batch_size: reduce if OOM
        device: "cuda", "cpu", or None (auto-detect)

    Returns:
        {
            "bertscore_f1": float,        # mean F1
            "bertscore_precision": float,
            "bertscore_recall": float,
            "per_example_f1": list[float]
        }
    """
    if not BERT_SCORE_AVAILABLE:
        return {
            "bertscore_f1": None,
            "error": "bert-score not installed. Run: pip install bert-score",
        }

    logger.info(f"Computing BERTScore with model: {model_type}")

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        P, R, F1 = bert_score_fn(
            cands=predictions,
            refs=gold_queries,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=True,
            lang="en",
        )

        return {
            "bertscore_f1": round(F1.mean().item(), 4),
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "per_example_f1": [round(f.item(), 4) for f in F1],
            "model_type": model_type,
        }

    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return {"bertscore_f1": None, "error": str(e)}


class CachedBERTScorer:
    """
    Wrapper that loads the BERTScore model once and reuses it.
    More efficient when running multiple eval rounds.
    """

    def __init__(self, model_type: str = "roberta-large", device: Optional[str] = None):
        if not BERT_SCORE_AVAILABLE:
            raise ImportError("bert-score not installed. Run: pip install bert-score")

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = BERTScorer(model_type=model_type, device=self.device)
        logger.info(f"BERTScorer loaded: {model_type} on {self.device}")

    def score(self, predictions: list[str], gold_queries: list[str]) -> dict:
        P, R, F1 = self.scorer.score(predictions, gold_queries, verbose=True)
        return {
            "bertscore_f1": round(F1.mean().item(), 4),
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "per_example_f1": [round(f.item(), 4) for f in F1],
        }
