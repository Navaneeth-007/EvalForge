import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.examples import EXAMPLES


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_health(client):
    assert client.get('/health').json()['ready'] is True


def test_walkthrough_is_explicit(client):
    response = client.post('/generate-sql', json={key: EXAMPLES[0][key] for key in ('question', 'schema')})
    assert response.status_code == 200
    assert response.json()['mode'] == 'walkthrough'
    assert 'no model inference' in response.json()['model']
    assert response.json()['sql'] == EXAMPLES[0]['sql']


def test_custom_query_is_not_faked(client):
    response = client.post('/generate-sql', json={'question': 'Count all users', 'schema': 'CREATE TABLE users (id INTEGER);'})
    assert response.status_code == 422


@pytest.mark.parametrize('payload', [{}, {'question': '  ', 'schema': 'CREATE TABLE t (id INT);'},
                                    {'question': 'x'*1001, 'schema': 'CREATE TABLE t (id INT);'}])
def test_validation(client, payload):
    assert client.post('/generate-sql', json=payload).status_code == 422


def test_results_do_not_invent_metrics(client):
    assert client.get('/results').json()['metrics'] is None


def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'authored examples' in response.text
    assert response.headers['x-content-type-options'] == 'nosniff'


def test_examples(client):
    assert len(client.get('/examples').json()['examples'][0]['sample_questions']) == 3


def test_model_mode_requires_adapter(monkeypatch):
    import app.main as module
    monkeypatch.setattr(module, 'MODE', 'model')
    monkeypatch.setattr(module, 'ADAPTER_MODEL', None)
    with pytest.raises(ValueError, match='ADAPTER_MODEL'):
        module.load_model()
