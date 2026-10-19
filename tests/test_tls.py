"""Local CA lifecycle and real TLS trust checks, without a GPU or network listener."""
import os
from pathlib import Path
import shutil
import ssl
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import tls


@pytest.fixture
def openssl():
    executable = shutil.which('openssl')
    if not executable:
        pytest.skip('OpenSSL is required for certificate integration tests')
    return executable


@pytest.fixture
def bundle(tmp_path, openssl):
    return tls.prepare_local_tls(['10.0.0.8', 'flex.internal'], tmp_path / 'certificates with spaces')


def _handshake(bundle, hostname, *, trust=True):
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(str(bundle.certfile), str(bundle.keyfile))
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if trust:
        client_context.load_verify_locations(cafile=str(bundle.ca_certfile))
    client_input, client_output = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_input, server_output = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_context.wrap_bio(client_input, client_output, server_hostname=hostname)
    server = server_context.wrap_bio(server_input, server_output, server_side=True)
    done = set()
    for _ in range(10):
        for connection in (client, server):
            if connection not in done:
                try:
                    connection.do_handshake()
                    done.add(connection)
                except ssl.SSLWantReadError:
                    pass
        client_bytes, server_bytes = client_output.read(), server_output.read()
        if client_bytes:
            server_input.write(client_bytes)
        if server_bytes:
            client_input.write(server_bytes)
        if len(done) == 2:
            return client.getpeercert()
    pytest.fail('TLS memory handshake did not finish')


@pytest.mark.parametrize('host', [
    '', ' ', '10.0.0.8 ', '0.0.0.0', '::', 'http://10.0.0.8', 'https://flex.internal',
    '10.0.0.8:18889', 'flex.internal:443', 'flex.internal/path', 'flex.internal\\path',
    'flex.internal\nDNS.9=evil', '[::1]', 'fe80::1%eth0', '*.internal', '-flex.internal',
    'flex-.internal', 'flex..internal', 'flex.internal..', '999.123.1.8', 'host name',
])
def test_invalid_host_rejected_before_creating_files(tmp_path, host):
    directory = tmp_path / 'not-created'
    with pytest.raises(tls.TLSConfigError):
        tls.prepare_local_tls([host], directory)
    assert not directory.exists()


def test_empty_host_list_rejected(tmp_path):
    with pytest.raises(tls.TLSConfigError, match='at least one'):
        tls.prepare_local_tls([], tmp_path)


def test_normalized_hosts_deduplicate_and_include_loopback():
    assert tls._normalize_hosts(['FLEX.INTERNAL.', 'flex.internal', '2001:db8::1']) == [
        'flex.internal', '2001:db8::1', 'localhost', '127.0.0.1', '::1']


def test_missing_openssl_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(tls.shutil, 'which', lambda _: None)
    with pytest.raises(tls.TLSConfigError, match='apt-get install openssl'):
        tls.prepare_local_tls(['10.0.0.8'], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_generated_certificate_trust_and_san_are_real(bundle):
    for hostname in ('10.0.0.8', 'flex.internal', 'localhost', '127.0.0.1', '::1'):
        peer = _handshake(bundle, hostname)
        assert peer['subjectAltName']
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(bundle, 'unlisted.internal')
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(bundle, 'flex.internal', trust=False)
    assert {path.name for path in bundle.certfile.parent.iterdir()} == {
        'server.crt', 'server.key', 'root-ca.crt', 'root-ca.key'}
    if os.name != 'nt':
        for name in ('server.key', 'root-ca.key'):
            assert (bundle.certfile.parent / name).stat().st_mode & 0o777 == 0o600


def test_restart_reuses_every_file_without_reissuing(bundle):
    directory = bundle.certfile.parent
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    same = tls.prepare_local_tls(['10.0.0.8', 'flex.internal'], directory)
    assert same == bundle
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


def test_new_hostname_renews_leaf_but_preserves_trusted_ca(bundle):
    original_ca = bundle.ca_certfile.read_bytes()
    original_ca_key = (bundle.ca_certfile.parent / 'root-ca.key').read_bytes()
    original_leaf = bundle.certfile.read_bytes()
    updated = tls.prepare_local_tls(['10.0.0.8', 'new.internal'], bundle.certfile.parent)
    assert updated.ca_certfile.read_bytes() == original_ca
    assert (updated.ca_certfile.parent / 'root-ca.key').read_bytes() == original_ca_key
    assert updated.certfile.read_bytes() != original_leaf
    _handshake(updated, 'new.internal')


def test_near_expiry_leaf_renews_under_the_same_ca(bundle, openssl, tmp_path):
    directory = bundle.certfile.parent
    csr, short_cert, config = tmp_path / 'short.csr', tmp_path / 'short.crt', tmp_path / 'short.cnf'
    config.write_text('[server]\nbasicConstraints=critical,CA:FALSE\n'
                      'keyUsage=critical,digitalSignature,keyEncipherment\n'
                      'extendedKeyUsage=serverAuth\n'
                      'subjectAltName=DNS:flex.internal,IP:10.0.0.8\n', encoding='ascii')
    tls._run(openssl, 'x509', '-x509toreq', '-in', bundle.certfile, '-signkey', bundle.keyfile, '-out', csr)
    tls._run(openssl, 'x509', '-req', '-in', csr, '-CA', bundle.ca_certfile,
             '-CAkey', directory / 'root-ca.key', '-set_serial', '12345', '-days', '1',
             '-sha256', '-extfile', config, '-extensions', 'server', '-out', short_cert)
    bundle.certfile.write_bytes(short_cert.read_bytes())
    ca_before, leaf_before = bundle.ca_certfile.read_bytes(), bundle.certfile.read_bytes()
    renewed = tls.prepare_local_tls(['10.0.0.8', 'flex.internal'], directory)
    assert renewed.ca_certfile.read_bytes() == ca_before
    assert renewed.certfile.read_bytes() != leaf_before
    _handshake(renewed, 'flex.internal')


@pytest.mark.parametrize('name', ['root-ca.crt', 'root-ca.key'])
def test_partial_ca_is_not_silently_replaced(tmp_path, openssl, name):
    preserved = tmp_path / name
    preserved.write_text('keep existing trust state', encoding='ascii')
    with pytest.raises(tls.TLSConfigError, match='Incomplete local CA'):
        tls.prepare_local_tls(['10.0.0.8'], tmp_path)
    assert preserved.read_text() == 'keep existing trust state'
    assert {p.name for p in tmp_path.iterdir()} == {name}


def test_corrupt_ca_is_not_silently_replaced(tmp_path, openssl):
    for name in ('root-ca.crt', 'root-ca.key'):
        (tmp_path / name).write_text('corrupt but never overwrite', encoding='ascii')
    with pytest.raises(tls.TLSConfigError, match='has NOT been replaced'):
        tls.prepare_local_tls(['10.0.0.8'], tmp_path)
    assert all(p.read_text() == 'corrupt but never overwrite' for p in tmp_path.iterdir())


def test_unrelated_server_certificate_is_not_overwritten(bundle, tmp_path, openssl):
    other = tls.prepare_local_tls(['10.0.0.9'], tmp_path / 'other')
    bundle.certfile.write_bytes(other.certfile.read_bytes())
    bundle.keyfile.write_bytes(other.keyfile.read_bytes())
    before = bundle.certfile.read_bytes()
    with pytest.raises(tls.TLSConfigError):
        tls.prepare_local_tls(['10.0.0.8'], bundle.certfile.parent)
    assert bundle.certfile.read_bytes() == before


def test_explicit_pair_validation(bundle, tmp_path):
    tls.validate_tls_files(bundle.certfile, bundle.keyfile)
    with pytest.raises(tls.TLSConfigError, match='Cannot load TLS'):
        tls.validate_tls_files(bundle.certfile, tmp_path / 'missing.key')
    with pytest.raises(tls.TLSConfigError, match='matching PEM'):
        tls.validate_tls_files(bundle.certfile, bundle.ca_certfile.parent / 'root-ca.key')


def test_preexisting_lock_has_actionable_failure(tmp_path, openssl):
    (tmp_path / '.tls-setup.lock').mkdir()
    with pytest.raises(tls.TLSConfigError, match='Another server may be starting'):
        tls.prepare_local_tls(['10.0.0.8'], tmp_path)
    assert (tmp_path / '.tls-setup.lock').is_dir()
