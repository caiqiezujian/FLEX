"""Small, persistent local CA for an explicitly requested intranet HTTPS endpoint.

Only root-ca.crt is public. Clients must trust that certificate once; generating
it on the server does not install trust on other computers. Never distribute
root-ca.key or server.key. OpenSSL 1.1.1 or newer is required for local issuance.
"""
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shutil
import ssl
import subprocess
import tempfile
from typing import List


class TLSConfigError(ValueError):
    """Actionable TLS setup failure, suitable for argparse.error."""


@dataclass(frozen=True)
class TLSFiles:
    certfile: Path
    keyfile: Path
    ca_certfile: Path


RENEW_BEFORE_SECONDS = 30 * 24 * 60 * 60


def _normalize_hosts(hosts: List[str]) -> List[str]:
    if not hosts:
        raise TLSConfigError('Specify at least one --https-host: the server IP or DNS name used by clients.')
    result = []
    for value in [*hosts, 'localhost', '127.0.0.1', '::1']:
        if not isinstance(value, str) or not value or value != value.strip():
            raise TLSConfigError('HTTPS hosts must be nonempty IP addresses or DNS names without whitespace.')
        if any(char in value for char in '/\\\r\n\t%[]'):
            raise TLSConfigError(f'Invalid HTTPS host {value!r}: use a bare IP or DNS name, without URL, port or path.')
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            try:
                normalized = (value[:-1] if value.endswith('.') else value).encode('idna').decode('ascii').lower()
            except UnicodeError as exc:
                raise TLSConfigError(f'Invalid HTTPS host {value!r}.') from exc
            labels = normalized.split('.')
            if (len(normalized) > 253 or ':' in normalized
                    or all(label.isdigit() for label in labels)
                    or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
                           for label in labels)):
                raise TLSConfigError(f'Invalid HTTPS host {value!r}: use a bare IP or DNS name, without URL, port or path.')
        else:
            if address.is_unspecified:
                raise TLSConfigError(f'{value} is a listening address, not a client address. Use the real server IP for --https-host.')
            normalized = str(address)
        if normalized not in result:
            result.append(normalized)
    return result


def validate_tls_files(certfile: Path, keyfile: Path) -> None:
    """Check readable PEM files and a matching, unencrypted key without a model."""
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # Supplying an empty password prevents an interactive OpenSSL prompt.
        context.load_cert_chain(str(certfile), str(keyfile), password='')
    except (OSError, ValueError) as exc:
        raise TLSConfigError(f'Cannot load TLS certificate/key ({certfile}, {keyfile}): {exc}. '
                             'Provide matching PEM files with an unencrypted private key.') from exc


def _run(openssl: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run([openssl, *map(str, args)], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TLSConfigError(f'OpenSSL could not complete TLS setup: {exc}') from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise TLSConfigError(f'OpenSSL TLS setup failed: {detail}')
    return result


def _private(path: Path) -> None:
    # Windows ACLs are inherited; on Unix private material must be owner-only.
    if os.name != 'nt':
        path.chmod(0o600)


def _check_pair_presence(cert: Path, key: Path, description: str) -> bool:
    for path in (cert, key):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise TLSConfigError(f'Refusing non-regular TLS file: {path}')
    if cert.exists() != key.exists():
        raise TLSConfigError(f'Incomplete {description} in {cert.parent}: both {cert.name} and {key.name} '
                             'must exist. Restore the matching files from backup; do not replace the trusted root casually.')
    return cert.exists()


def _check_ca(openssl: str, cert: Path, key: Path) -> None:
    try:
        validate_tls_files(cert, key)
        _run(openssl, 'verify', '-CAfile', cert, '-no-CApath', '-check_ss_sig', cert)
        info = _run(openssl, 'x509', '-in', cert, '-noout', '-text').stdout
        if 'CA:TRUE' not in info or 'Certificate Sign' not in info:
            raise TLSConfigError('Certificate does not contain CA signing constraints.')
        if _run(openssl, 'x509', '-in', cert, '-noout', '-checkend',
                str(RENEW_BEFORE_SECONDS), check=False).returncode:
            raise TLSConfigError('The local root CA expires within 30 days; renew and redistribute its trust explicitly.')
    except TLSConfigError as exc:
        raise TLSConfigError(f'Existing local CA is invalid ({cert.parent}): {exc} '
                             'It has NOT been replaced. Restore its matching certificate/key from backup.') from exc
    _private(key)


def _generate_ca(openssl: str, directory: Path, cert: Path, key: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='.tls-ca-', dir=str(directory)) as temp:
        tempdir = Path(temp)
        config = tempdir / 'ca.cnf'
        config.write_text('[req]\nprompt = no\ndistinguished_name = dn\nx509_extensions = ca\n'
                          '[dn]\nCN = FLEX Local Root CA\n[ca]\n'
                          'basicConstraints = critical, CA:TRUE, pathlen:0\n'
                          'keyUsage = critical, keyCertSign, cRLSign\n'
                          'subjectKeyIdentifier = hash\n', encoding='ascii')
        new_cert, new_key = tempdir / 'ca.crt', tempdir / 'ca.key'
        _run(openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256',
             '-days', '3650', '-config', config, '-keyout', new_key, '-out', new_cert)
        _check_ca(openssl, new_cert, new_key)
        # The directory lock ensures another launcher cannot create these paths.
        if cert.exists() or key.exists():
            raise TLSConfigError('CA files appeared during setup; refusing to overwrite them.')
        os.replace(new_key, key)
        os.replace(new_cert, cert)


def _leaf_matches(openssl: str, cert: Path, key: Path, ca: Path, hosts: List[str]) -> bool:
    validate_tls_files(cert, key)
    # Corrupt/unrelated material is not overwritten. An expired *owned* leaf can renew.
    _run(openssl, 'verify', '-no_check_time', '-CAfile', ca, '-no-CApath', '-purpose', 'sslserver', cert)
    san = _run(openssl, 'x509', '-in', cert, '-noout', '-ext', 'subjectAltName').stdout
    if 'X509v3 Subject Alternative Name:' not in san:
        return False
    if _run(openssl, 'x509', '-in', cert, '-noout', '-checkend',
            str(RENEW_BEFORE_SECONDS), check=False).returncode:
        return False
    for host in hosts:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            flag = '-verify_hostname'
        else:
            flag = '-verify_ip'
        if _run(openssl, 'verify', '-CAfile', ca, '-no-CApath', '-purpose', 'sslserver',
                flag, host, cert, check=False).returncode:
            return False
    _private(key)
    return True


def _generate_leaf(openssl: str, directory: Path, cert: Path, key: Path,
                   ca_cert: Path, ca_key: Path, hosts: List[str]) -> None:
    with tempfile.TemporaryDirectory(prefix='.tls-server-', dir=str(directory)) as temp:
        tempdir = Path(temp)
        config = tempdir / 'server.cnf'
        names = []
        for index, host in enumerate(hosts, 1):
            try:
                ipaddress.ip_address(host)
            except ValueError:
                kind = 'DNS'
            else:
                kind = 'IP'
            names.append(f'{kind}.{index} = {host}')
        config.write_text('[req]\nprompt = no\ndistinguished_name = dn\n'
                          '[dn]\nCN = FLEX Intranet Server\n[server]\n'
                          'basicConstraints = critical, CA:FALSE\n'
                          'keyUsage = critical, digitalSignature, keyEncipherment\n'
                          'extendedKeyUsage = serverAuth\nsubjectKeyIdentifier = hash\n'
                          'authorityKeyIdentifier = keyid,issuer\n'
                          'subjectAltName = @names\n[names]\n' + '\n'.join(names) + '\n', encoding='ascii')
        new_cert, new_key, csr = tempdir / 'server.crt', tempdir / 'server.key', tempdir / 'server.csr'
        _run(openssl, 'req', '-new', '-newkey', 'rsa:2048', '-nodes', '-sha256',
             '-config', config, '-keyout', new_key, '-out', csr)
        _run(openssl, 'x509', '-req', '-in', csr, '-CA', ca_cert, '-CAkey', ca_key,
             '-set_serial', '0x' + secrets.token_hex(16), '-days', '365', '-sha256',
             '-extfile', config, '-extensions', 'server', '-out', new_cert)
        if not _leaf_matches(openssl, new_cert, new_key, ca_cert, hosts):
            raise TLSConfigError('Generated server certificate did not pass validation.')
        _private(new_key)
        os.replace(new_key, key)
        os.replace(new_cert, cert)


def prepare_local_tls(hosts: List[str], directory: Path) -> TLSFiles:
    """Create/reuse a private local CA and a server certificate for these hosts."""
    hosts = _normalize_hosts(hosts)
    openssl = shutil.which('openssl')
    if not openssl:
        raise TLSConfigError('OpenSSL is required to create local HTTPS certificates. '
                             'Install it (Ubuntu/Debian: apt-get install openssl; Conda: conda install openssl), '
                             'or provide --ssl-certfile and --ssl-keyfile instead.')
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        cert, key = directory / 'server.crt', directory / 'server.key'
        ca_cert, ca_key = directory / 'root-ca.crt', directory / 'root-ca.key'
        lock = directory / '.tls-setup.lock'
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise TLSConfigError(f'TLS setup lock already exists: {lock}. Another server may be starting. '
                                 'If a previous setup crashed, confirm no setup is running before removing this empty lock directory.') from exc
        try:
            has_ca = _check_pair_presence(ca_cert, ca_key, 'local CA')
            has_leaf = _check_pair_presence(cert, key, 'server certificate')
            if not has_ca and has_leaf:
                raise TLSConfigError(f'Server certificates exist without their CA in {directory}; refusing to replace them.')
            if has_ca:
                _check_ca(openssl, ca_cert, ca_key)
            else:
                _generate_ca(openssl, directory, ca_cert, ca_key)
            if not has_leaf or not _leaf_matches(openssl, cert, key, ca_cert, hosts):
                _generate_leaf(openssl, directory, cert, key, ca_cert, ca_key, hosts)
            return TLSFiles(certfile=cert, keyfile=key, ca_certfile=ca_cert)
        finally:
            lock.rmdir()
    except OSError as exc:
        raise TLSConfigError(f'Cannot prepare HTTPS certificate directory {directory}: {exc}') from exc
