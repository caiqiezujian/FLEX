"""CLI deployment checks, without loading a model or binding a network port."""
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import server


@pytest.mark.parametrize('token_env', ['FLEX_ACCESS_TOKEN', 'FLEX_TEST_CUSTOM_TOKEN'])
@pytest.mark.parametrize('token', [None, ''])
@pytest.mark.parametrize('host', ['0.0.0.0', '10.0.0.8'])
def test_lan_bind_allows_unset_or_empty_token(monkeypatch, token_env, token, host):
    if token is None:
        monkeypatch.delenv(token_env, raising=False)
    else:
        monkeypatch.setenv(token_env, token)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', '--host', host,
                                     '--port', '18889', '--token-env', token_env, '--no-log'])
    captured = {}

    def fake_run(app, **options):
        captured.update(app=app, **options)

    monkeypatch.setattr(uvicorn, 'run', fake_run)
    server.main()
    assert captured['host'] == host and captured['port'] == 18889
    assert captured['workers'] == 1
    with TestClient(captured['app'], base_url='http://10.0.0.8:18889') as client:
        response = client.get('/api/config')
        assert response.status_code == 200
        assert response.json()['auth_required'] is False


@pytest.mark.parametrize('token_env', ['FLEX_ACCESS_TOKEN', 'FLEX_TEST_CUSTOM_TOKEN'])
def test_authenticated_lan_http_keeps_requested_port(monkeypatch, token_env):
    monkeypatch.setenv(token_env, 'synthetic-test-token-not-a-real-secret')
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', '--host', '0.0.0.0',
                                     '--port', '18889', '--token-env', token_env, '--no-log'])
    captured = {}

    def fake_run(app, **options):
        captured.update(app=app, **options)

    monkeypatch.setattr(uvicorn, 'run', fake_run)
    server.main()
    assert captured['host'] == '0.0.0.0'
    assert captured['port'] == 18889
    assert captured['workers'] == 1
    assert 'ssl_certfile' not in captured  # No implicit TLS requirement for file mode.
    with TestClient(captured['app']) as client:
        response = client.get('/api/config')
        assert response.status_code == 200
        assert response.json()['auth_required'] is True
        assert 'synthetic-test-token' not in response.text
