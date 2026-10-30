"""
Execution Accuracy for Text-to-SQL evaluation.

Executes both predicted and gold SQL against the Spider SQLite databases.
Compares result sets — a prediction is correct if it returns the same
rows as the gold query (order-independent).

This is more lenient than exact match: it rewards semantically equivalent
queries that look syntactically different.

Requires: Spider database files
  Download from https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9FZFZ5YtBNkPAH
  and extract to ./data/spider_databases/
"""

import time
from collections import Counter
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default path to Spider database files
SPIDER_DB_DIR = os.getenv("SPIDER_DB_DIR", "./data/spider_databases")


def execute_sql(
    sql: str,
    db_path: str,
    timeout: float = 5.0,
) -> tuple[Optional[list], Optional[str]]:
    """
    Execute SQL against a SQLite database.

    Returns:
        (rows, error): rows is list of tuples if successful, None on error.
                       error is None if successful.
    """
    conn = None
    try:
        # URI read-only mode prevents modifying the benchmark or creating missing databases.
        conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
        conn.set_authorizer(
            lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        )
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cursor = conn.execute(sql)
        rows = []
        result_bytes = 0
        for row in cursor:
            result_bytes += sum(len(cell) if isinstance(cell, (str, bytes)) else 8 for cell in row)
            if len(rows) >= 10000 or result_bytes > 10_000_000:
                return None, "Result exceeds evaluation output limit"
            rows.append(row)
        return rows, None
    except sqlite3.Error as error:
        return None, str(error)
    finally:
        if conn is not None:
            conn.close()


def result_sets_equal(rows_a: list, rows_b: list) -> bool:
    """
    Compare two result sets order-independently.
    Preserves duplicate rows and SQL NULL values; numeric equivalents compare equally.
    """
    if rows_a is None or rows_b is None:
        return False

    # Counter preserves duplicates and NULLs, while treating numeric 1 and 1.0 equally.
    return Counter(map(tuple, rows_a)) == Counter(map(tuple, rows_b))


def execution_accuracy_single(
    prediction: str,
    gold: str,
    db_id: str,
    db_dir: str = SPIDER_DB_DIR,
) -> dict:
    """
    Compute execution accuracy for a single example.

    Returns dict with:
        - correct: bool
        - prediction_rows: result of running prediction
        - gold_rows: result of running gold
        - prediction_error: error string if prediction failed
        - gold_error: error string if gold failed
    """
    db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return {
            "correct": False,
            "error": f"Database not found: {db_path}",
            "prediction_rows": None,
            "gold_rows": None,
            "prediction_error": "Database unavailable",
            "gold_error": "Database unavailable",
        }

    pred_rows, pred_err = execute_sql(prediction, str(db_path))
    gold_rows, gold_err = execute_sql(gold, str(db_path))

    correct = (
        pred_rows == gold_rows if "order by" in gold.lower() else result_sets_equal(pred_rows, gold_rows)
    )

    return {
        "correct": correct,
        "prediction_rows": pred_rows,
        "gold_rows": gold_rows,
        "prediction_error": pred_err,
        "gold_error": gold_err,
    }


def compute_execution_accuracy(
    predictions: list[str],
    gold_queries: list[str],
    db_ids: list[str],
    db_dir: str = SPIDER_DB_DIR,
) -> dict:
    """
    Compute execution accuracy over a list of examples.

    Returns:
        {
            "execution_accuracy": 0.xx,
            "num_correct": int,
            "num_executable": int,  # predictions that ran without error
            "num_total": int,
            "per_example": [...]
        }
    """
    assert len(predictions) == len(gold_queries) == len(db_ids)

    if not Path(db_dir).exists():
        logger.warning(
            f"Spider database directory not found: {db_dir}\n"
            "Download from the Spider website and set SPIDER_DB_DIR env var.\n"
            "Skipping execution accuracy."
        )
        return {
            "execution_accuracy": None,
            "error": f"Database directory not found: {db_dir}",
        }

    results = []
    correct = 0
    executable = 0

    for pred, gold, db_id in zip(predictions, gold_queries, db_ids):
        r = execution_accuracy_single(pred, gold, db_id, db_dir)
        if r["prediction_error"] is None:
            executable += 1
        if r["correct"]:
            correct += 1
        results.append(r)

    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "execution_accuracy": round(accuracy, 4),
        "num_correct": correct,
        "num_executable": executable,
        "num_total": total,
        "per_example": results,
    }
