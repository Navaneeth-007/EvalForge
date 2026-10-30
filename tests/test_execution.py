import sqlite3
import pytest
from eval.execution_acc import execute_sql, execution_accuracy_single, compute_execution_accuracy


@pytest.fixture
def database(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    path = folder / "demo.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE items (id INTEGER); INSERT INTO items VALUES (1), (2), (2);")
    return tmp_path, path


def test_deny_writes_and_attach(database, tmp_path):
    _, path = database
    for query in [
        "DELETE FROM items",
        "DROP TABLE items",
        f"ATTACH DATABASE '{tmp_path}/other.sqlite' AS other",
    ]:
        rows, error = execute_sql(query, str(path))
        assert rows is None and error
    assert execute_sql("SELECT COUNT(*) FROM items", str(path))[0] == [(3,)]


def test_timeout(database):
    _, path = database
    rows, error = execute_sql(
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n",
        str(path),
        timeout=0.01,
    )
    assert rows is None and "interrupted" in error


def test_order_and_duplicates(database):
    root, _ = database
    assert not execution_accuracy_single(
        "SELECT DISTINCT id FROM items", "SELECT id FROM items", "demo", str(root)
    )["correct"]
    assert not execution_accuracy_single(
        "SELECT id FROM items ORDER BY id DESC", "SELECT id FROM items ORDER BY id ASC", "demo", str(root)
    )["correct"]
    assert execution_accuracy_single(
        "SELECT id FROM items ORDER BY id DESC", "SELECT id FROM items", "demo", str(root)
    )["correct"]


def test_missing_database_does_not_crash(database):
    root, _ = database
    result = compute_execution_accuracy(["SELECT 1"], ["SELECT 1"], ["missing"], str(root))
    assert result["num_executable"] == 0


def test_result_types_and_blob_limit(database):
    from eval.execution_acc import result_sets_equal

    assert result_sets_equal([(1,)], [(1.0,)])
    assert not result_sets_equal([(None,)], [("None",)])
    _, path = database
    rows, error = execute_sql("SELECT zeroblob(2000000)", str(path))
    assert rows is None and error


def test_saved_predictions_can_be_rescored_without_models(database):
    from eval.rescore import rescore

    root, _ = database
    report = {
        "n": 1,
        "predictions": [
            {
                "sql": "SELECT 1.0",
                "db_id": "demo",
                "base_prediction": "SELECT 0",
                "finetuned_prediction": "SELECT 1",
            }
        ],
    }
    replay = rescore(report, str(root))
    assert replay["base_model_metrics"]["execution_accuracy"] == 0.0
    assert replay["finetuned_model_metrics"]["execution_accuracy"] == 1.0
