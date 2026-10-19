"""CLI deployment checks, without loading a model or binding a network port."""
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import server


@pytest.mark.parametrize('token_env', ['FLEX_ACCESS_TOKEN', 'FLEX_TEST_CUSTOM_TOKEN'])
def test_lan_bind_requires_the_selected_token_env(monkeypatch, capsys, token_env):
    monkeypatch.delenv(token_env, raising=False)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', '--host', '0.0.0.0',
                                     '--port', '18889', '--token-env', token_env])
    with pytest.raises(SystemExit) as failure:
        server.main()
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert f'Set {token_env}' in error
    assert 'HTTP file playback is supported' in error
    assert 'microphone capture requires a secure context' in error


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
