"""CLI deployment checks, without loading a model or binding a network port."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import server


def test_long_session_cli_settings_reach_model_and_result_store(monkeypatch, tmp_path):
    monkeypatch.delenv('FLEX_ACCESS_TOKEN', raising=False)
    monkeypatch.setattr(sys, 'argv', [
        'run.py', '--backend', 'mock', '--no-log', '--temp-dir', str(tmp_path),
        '--max-session-seconds', '36000', '--context-policy', 'rolling',
        '--history-window-seconds', '90', '--history-retain-seconds', '45',
        '--max-upload-mb', '2048', '--max-result-mb', '64', '--result-retention-hours', '12',
    ])
    captured = {}
    monkeypatch.setattr(uvicorn, 'run', lambda app, **options: captured.update(app=app, **options))
    server.main()
    with TestClient(captured['app']) as client:
        config = client.get('/api/config').json()
        assert config['max_session_seconds'] == 36000
        assert config['history_window_seconds'] == 90
        assert config['history_retain_seconds'] == 45
        assert config['context_policy'] == 'rolling'
        assert config['max_upload_bytes'] == 2048 * 1024 * 1024
        assert config['result_retention_hours'] == 12
        store = captured['app'].state.results
        assert store.max_bytes == 64 * 1024 * 1024
        assert store.ttl_seconds == 12 * 3600
        assert store.root.parent == tmp_path
    assert not list(tmp_path.iterdir())


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


def test_local_https_keeps_gpu_worker_host_port_and_optional_auth(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('FLEX_ACCESS_TOKEN', raising=False)
    cert_dir = tmp_path / 'persistent certificates'
    cert = SimpleNamespace(certfile=cert_dir / 'server.crt', keyfile=cert_dir / 'server.key',
                           ca_certfile=cert_dir / 'root-ca.crt')
    calls = []
    monkeypatch.setattr(server, 'prepare_local_tls', lambda hosts, directory:
                        calls.append((hosts, directory)) or cert)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', '--host', '0.0.0.0',
                                     '--port', '18889', '--https-host', '10.0.0.8',
                                     '--https-host', 'flex.example.internal',
                                     '--cert-dir', str(cert_dir), '--no-log'])
    captured = {}
    monkeypatch.setattr(uvicorn, 'run', lambda app, **options: captured.update(app=app, **options))
    server.main()
    assert calls == [(['10.0.0.8', 'flex.example.internal'], cert_dir)]
    assert captured['ssl_certfile'] == str(cert.certfile)
    assert captured['ssl_keyfile'] == str(cert.keyfile)
    assert (captured['host'], captured['port'], captured['workers']) == ('0.0.0.0', 18889, 1)
    with TestClient(captured['app'], base_url='https://10.0.0.8:18889') as client:
        assert client.get('/api/config').json()['auth_required'] is False
    output = capsys.readouterr().out
    assert 'https://10.0.0.8:18889' in output
    assert str(cert.ca_certfile) in output


def test_existing_https_certificate_is_preflighted(monkeypatch, tmp_path):
    cert, key = tmp_path / 'existing.crt', tmp_path / 'existing.key'
    checked, captured = [], {}
    monkeypatch.setattr(server, 'validate_tls_files', lambda *paths: checked.append(paths))
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', '--no-log',
                                     '--ssl-certfile', str(cert), '--ssl-keyfile', str(key)])
    monkeypatch.setattr(uvicorn, 'run', lambda app, **options: captured.update(options))
    server.main()
    assert checked == [(cert, key)]
    assert captured['ssl_certfile'] == str(cert)
    assert captured['ssl_keyfile'] == str(key)


@pytest.mark.parametrize('extra', [
    ['--ssl-certfile', 'cert.crt'],
    ['--ssl-keyfile', 'key.key'],
    ['--https-host', '10.0.0.8', '--ssl-certfile', 'cert.crt', '--ssl-keyfile', 'key.key'],
])
def test_incomplete_or_conflicting_tls_fails_before_app_creation(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', *extra])
    monkeypatch.setattr(server, 'create_app', lambda *_: pytest.fail('App must not be created'))
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2


@pytest.mark.parametrize('generated', [True, False])
def test_bad_tls_fails_before_app_or_model_creation(monkeypatch, generated, capsys):
    def fail(*args):
        raise server.TLSConfigError('synthetic certificate configuration failure')
    monkeypatch.setattr(server, 'prepare_local_tls', fail)
    monkeypatch.setattr(server, 'validate_tls_files', fail)
    monkeypatch.setattr(server, 'create_app', lambda *_: pytest.fail('App must not be created'))
    options = ['--https-host', '10.0.0.8'] if generated else ['--ssl-certfile', 'x.crt', '--ssl-keyfile', 'x.key']
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', *options])
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2
    assert 'synthetic certificate configuration failure' in capsys.readouterr().err


@pytest.mark.parametrize('extra', [
    ['--https-host', '10.0.0.8', '--cert-dir', str(server.STATIC)],
    ['--https-host', '10.0.0.8', '--cert-dir', str(server.STATIC / 'certs')],
    ['--ssl-certfile', str(server.STATIC / 'server.crt'), '--ssl-keyfile', 'server.key'],
    ['--ssl-certfile', 'server.crt', '--ssl-keyfile', str(server.STATIC / 'server.key')],
])
def test_tls_material_cannot_be_stored_under_public_static_directory(monkeypatch, extra, capsys):
    monkeypatch.setattr(sys, 'argv', ['run.py', '--backend', 'mock', *extra])
    monkeypatch.setattr(server, 'prepare_local_tls', lambda *_: pytest.fail('Must not create exposed keys'))
    monkeypatch.setattr(server, 'validate_tls_files', lambda *_: pytest.fail('Must reject public paths first'))
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2
    assert 'publicly served static directory' in capsys.readouterr().err
