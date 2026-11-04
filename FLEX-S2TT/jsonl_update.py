"""Index existing JSONL rows and replace only completed inference results."""
from __future__ import annotations

from dataclasses import dataclass
import codecs
import json
import logging
import os
from pathlib import Path
import stat
import tempfile


LOG = logging.getLogger('flex.batch')


@dataclass
class JsonlDocument:
    path: Path
    snapshot: bytes
    lines: list

    def replace_row(self, index, row):
        """Atomic per-file commit; retain untouched lines byte for byte.

        The temporary file is in the same directory and is removed on failure.
        There is no permanent backup/result copy. Do not run concurrent writers
        against this directory: the stale-content check is not a filesystem lock.
        """
        original = self.lines[index]
        ending = (b'\r\n' if original.endswith(b'\r\n') else
                  b'\n' if original.endswith(b'\n') else
                  b'\r' if original.endswith(b'\r') else b'')
        bom = codecs.BOM_UTF8 if index == 0 and original.startswith(codecs.BOM_UTF8) else b''
        encoded = bom + json.dumps(row, ensure_ascii=False, allow_nan=False).encode('utf-8') + ending
        new_lines = self.lines.copy()
        new_lines[index] = encoded
        new_bytes = b''.join(new_lines)
        self._check_unchanged()
        mode = stat.S_IMODE(self.path.stat().st_mode)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.' + self.path.name + '.', suffix='.tmp', dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(new_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            self._check_unchanged()
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.lines = new_lines
        self.snapshot = new_bytes

    def _check_unchanged(self):
        if self.path.is_symlink() or self.path.read_bytes() != self.snapshot:
            raise RuntimeError(f'JSONL changed after indexing; refusing to overwrite: {self.path}')


@dataclass
class UpdateTarget:
    audio: Path
    vid: str
    document: JsonlDocument
    line_index: int
    row: dict

    def save(self, prefix, translation):
        updated = dict(self.row)
        updated[prefix + '_trans'] = translation
        # Requested compatibility field: copy the supplied transcript, not ASR.
        updated[prefix + '_asr'] = self.row['transcript']
        self.document.replace_row(self.line_index, updated)
        self.row = updated


@dataclass
class UpdatePlan:
    targets: list
    jsonl_files: int
    jsonl_rows: int
    unmatched_rows: int
    unmatched_audios: list


def prepare_update(directory, entries, audio_suffixes):
    """Match exact vid/stem, independently of JSONL filename and original_vids."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('--jsonl-dir must be an existing directory')
    paths = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() == '.jsonl'), key=lambda p: p.name)
    if not paths:
        raise ValueError('No .jsonl files found directly in --jsonl-dir')

    def key(vid):
        # Also accept a vid that already includes a supported audio extension.
        suffix = Path(vid).suffix
        return vid[:-len(suffix)] if suffix.lower() in audio_suffixes else vid

    audio_by_key = {}
    for audio, vid in entries:
        audio_id = key(vid)
        if audio_id in audio_by_key:
            raise ValueError(f'Ambiguous audio stem {audio_id!r}: '
                             f'{audio_by_key[audio_id][0]} and {audio}')
        audio_by_key[audio_id] = (audio, vid)

    found, total_rows = {}, 0
    LOG.info('Indexing %d JSONL files against %d audio files...', len(paths), len(entries))
    for file_number, path in enumerate(paths, 1):
        if path.is_symlink():
            raise ValueError(f'Use regular JSONL files, not symlinks: {path}')
        snapshot = path.read_bytes()
        document = JsonlDocument(path, snapshot, snapshot.splitlines(keepends=True))
        matches = 0
        for index, raw in enumerate(document.lines):
            line = raw.decode('utf-8-sig' if index == 0 else 'utf-8')
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Invalid JSON at {path}:{index + 1}: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'JSONL row must be an object: {path}:{index + 1}')
            total_rows += 1
            vid = row.get('vid')
            audio_id = key(vid) if isinstance(vid, str) else None
            if audio_id not in audio_by_key:
                continue
            if audio_id in found:
                previous = found[audio_id]
                raise ValueError(f'Duplicate matching vid {vid!r}: '
                                 f'{previous.document.path}:{previous.line_index + 1} '
                                 f'and {path}:{index + 1}; nothing has been modified')
            if not isinstance(row.get('transcript'), str):
                raise ValueError(f'Matched row needs a string transcript: {path}:{index + 1}')
            # Ensure this row can be saved without allowing non-standard NaN/Infinity.
            json.dumps(row, ensure_ascii=False, allow_nan=False)
            audio, _ = audio_by_key[audio_id]
            found[audio_id] = UpdateTarget(audio, vid, document, index, row)
            matches += 1
        LOG.info('[index %d/%d] %s: matched=%d', file_number, len(paths), path.name, matches)

    targets, unmatched = [], []
    for audio, vid in entries:
        target = found.get(key(vid))
        if target is None:
            unmatched.append(audio)
        else:
            targets.append(target)
    return UpdatePlan(targets, len(paths), total_rows, total_rows - len(targets), unmatched)
