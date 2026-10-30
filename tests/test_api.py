import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.examples import EXAMPLES


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_health(client):
    assert client.get("/health").json()["ready"] is True


def test_walkthrough_is_explicit(client):
    response = client.post("/generate-sql", json={key: EXAMPLES[0][key] for key in ("question", "schema")})
    assert response.status_code == 200
    assert response.json()["mode"] == "walkthrough"
    assert "no model inference" in response.json()["model"]
    assert response.json()["sql"] == EXAMPLES[0]["sql"]


def test_custom_query_is_not_faked(client):
    response = client.post(
        "/generate-sql", json={"question": "Count all users", "schema": "CREATE TABLE users (id INTEGER);"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "  ", "schema": "CREATE TABLE t (id INT);"},
        {"question": "x" * 1001, "schema": "CREATE TABLE t (id INT);"},
    ],
)
def test_validation(client, payload):
    assert client.post("/generate-sql", json=payload).status_code == 422


def test_results_do_not_invent_metrics(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.EVIDENCE_PATH", tmp_path / "missing.json")
    assert client.get("/results").json()["metrics"] is None


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "recorded predictions" in response.text or "saved predictions" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_examples(client):
    assert len(client.get("/examples").json()["examples"][0]["sample_questions"]) == 3


def test_model_mode_requires_adapter(monkeypatch):
    import app.main as module

    monkeypatch.setattr(module, "MODE", "model")
    monkeypatch.setattr(module, "ADAPTER_MODEL", None)
    with pytest.raises(ValueError, match="ADAPTER_MODEL"):
        module.load_model()


def test_recorded_evidence_is_available(client, monkeypatch, tmp_path):
    import json

    artifact = tmp_path / "evidence.json"
    artifact.write_text(json.dumps({"status": "pilot_complete", "kind": "recorded_predictions"}))
    monkeypatch.setattr("app.main.EVIDENCE_PATH", artifact)
    assert client.get("/results").json()["kind"] == "recorded_predictions"


def test_model_generation_and_busy_limit(client, monkeypatch):
    from unittest.mock import MagicMock
    import app.main as module

    pipe = MagicMock(return_value=[{"generated_text": "SELECT 1;<|end|>"}])
    pipe.tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(module, "MODE", "model")
    monkeypatch.setattr(module, "ADAPTER_MODEL", "test/adapter")
    monkeypatch.setattr(module, "_pipe", pipe)
    payload = {"question": "Count all users", "schema": "CREATE TABLE users (id INTEGER);"}
    result = client.post("/generate-sql", json=payload)
    assert result.json()["sql"] == "SELECT 1;"
    assert result.json()["model"] == "test/adapter"
    module._inference_lock.acquire()
    try:
        busy = client.post("/generate-sql", json=payload)
        assert busy.status_code == 429
        assert busy.headers["retry-after"] == "5"
    finally:
        module._inference_lock.release()


def test_model_errors_do_not_expose_internal_details(client, monkeypatch):
    from unittest.mock import MagicMock
    import app.main as module

    pipe = MagicMock(side_effect=RuntimeError("private implementation details"))
    pipe.tokenizer.encode.return_value = [1]
    monkeypatch.setattr(module, "MODE", "model")
    monkeypatch.setattr(module, "_pipe", pipe)
    result = client.post(
        "/generate-sql", json={"question": "Count users", "schema": "CREATE TABLE users (id INT);"}
    )
    assert result.status_code == 503
    assert "private implementation details" not in result.text
    assert not module._inference_lock.locked()


def test_chunked_body_size_limit(client):
    chunks = (b"x" * 12001 for _ in range(2))
    response = client.post("/generate-sql", content=chunks, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
