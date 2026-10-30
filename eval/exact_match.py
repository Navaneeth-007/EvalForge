"""
Exact Match Accuracy for Text-to-SQL evaluation.

Normalises both predicted and gold SQL before comparison:
- Lowercase
- Strip leading/trailing whitespace
- Collapse internal whitespace
- Normalise SQL keywords

This is normalized string match, not the official Spider structural evaluator.
"""

import re


SQL_KEYWORDS = [
    "select",
    "from",
    "where",
    "group by",
    "order by",
    "having",
    "join",
    "left join",
    "right join",
    "inner join",
    "outer join",
    "on",
    "as",
    "and",
    "or",
    "not",
    "in",
    "like",
    "between",
    "is null",
    "is not null",
    "distinct",
    "limit",
    "offset",
    "union",
    "intersect",
    "except",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "asc",
    "desc",
    "insert",
    "update",
    "delete",
    "create",
    "drop",
]


def normalise_sql(sql: str) -> str:
    """
    Normalise SQL string for comparison.

    Steps:
    1. Lowercase
    2. Strip outer whitespace
    3. Collapse internal whitespace runs to single space
    4. Remove trailing semicolons
    5. Standardise quote characters
    """
    if not sql:
        return ""

    # Preserve quoted values and identifiers; lowercasing literals changes SQL meaning.
    pieces = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql.strip().rstrip(";"))
    return "".join(
        piece if index % 2 else re.sub(r"\s+", " ", piece.lower()) for index, piece in enumerate(pieces)
    ).strip()


def exact_match(prediction: str, gold: str) -> bool:
    """Return True if normalised prediction == normalised gold."""
    return normalise_sql(prediction) == normalise_sql(gold)


def compute_exact_match_accuracy(
    predictions: list[str],
    gold_queries: list[str],
) -> dict[str, float]:
    """
    Compute exact match accuracy over a list of predictions.

    Returns:
        {
            "exact_match_accuracy": 0.xx,
            "num_correct": int,
            "num_total": int,
            "per_example": [{"prediction": ..., "gold": ..., "match": bool}]
        }
    """
    assert len(predictions) == len(gold_queries), (
        f"Length mismatch: {len(predictions)} predictions vs {len(gold_queries)} gold"
    )

    results = []
    correct = 0

    for pred, gold in zip(predictions, gold_queries):
        match = exact_match(pred, gold)
        if match:
            correct += 1
        results.append(
            {
                "prediction": pred,
                "gold": gold,
                "prediction_normalised": normalise_sql(pred),
                "gold_normalised": normalise_sql(gold),
                "match": match,
            }
        )

    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "exact_match_accuracy": round(accuracy, 4),
        "num_correct": correct,
        "num_total": total,
        "per_example": results,
    }


def print_mismatches(results: list[dict], n: int = 5):
    """Print the first n mismatches for debugging."""
    mismatches = [r for r in results if not r["match"]]
    print(f"\n{'─' * 60}")
    print(f"First {min(n, len(mismatches))} mismatches (out of {len(mismatches)}):")
    for i, r in enumerate(mismatches[:n]):
        print(f"\nExample {i + 1}:")
        print(f"  Gold:       {r['gold_normalised']}")
        print(f"  Prediction: {r['prediction_normalised']}")
