"""Long-session result files and constant-memory export snapshots."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web.results import ResultLimit, ResultStore


def turn(index, text):
    return dict(type='turn', turn_id=index, tag='DIRECT', delta=text, source_end_s=index + 1)


def test_snapshot_is_a_complete_prefix_even_while_more_turns_arrive(tmp_path):
    store = ResultStore(temp_dir=tmp_path)
    try:
        item = store.create('session')
        item.write(dict(type='ready', config={'source_lang': 'en', 'target_lang': 'zh'}, utc='test'))
        text = '译文"\\\n😀' * 13000
        item.write(turn(0, text))
        first = item.snapshot()
        item.write(turn(1, '后续'))
        payload = json.loads(b''.join(first.iter_json()))
        assert payload['translation'] == text
        assert len(payload['turns']) == 1 and payload['turn_count'] == 1
        assert payload['source_language'] == 'en' and not payload['complete']
        assert first.text.closed and first.events.closed
        item.write(dict(type='done', reason='finished', turn_count=2, translation_chars=len(text) + 2))
        item.finish()
        final = json.loads(b''.join(item.snapshot().iter_json()))
        assert final['complete'] and final['translation'] == text + '后续'
        assert len(final['events']) == 4
        assert b''.join(item.snapshot().iter_text()).decode('utf-8') == text + '后续'
    finally:
        store.close()
    assert not list(tmp_path.iterdir())


def test_storage_limit_is_explicit_and_retains_a_valid_prefix(tmp_path):
    store = ResultStore(temp_dir=tmp_path, max_bytes=256)
    try:
        item = store.create('session')
        item.write(turn(0, '保留'))
        with pytest.raises(ResultLimit):
            item.write(turn(1, 'x' * 300))
        snapshot = json.loads(b''.join(item.snapshot().iter_json()))
        assert snapshot['translation'] == '保留' and len(snapshot['turns']) == 1
    finally:
        store.close()


def test_bounded_completed_sessions_expire_without_evicting_active(tmp_path):
    store = ResultStore(temp_dir=tmp_path, max_sessions=2, ttl_seconds=30)
    try:
        active = store.create('active')
        old = store.create('old')
        old.finish()
        third = store.create('third')
        assert set(store.items) == {'active', 'third'}
        assert active.directory.exists() and not old.directory.exists()
        with pytest.raises(ResultLimit):
            store.create('fourth')
        third.finish()
        third.finished_at -= 31
        store.expire()
        assert list(store.items) == ['active']
    finally:
        store.close()


def test_hours_of_events_do_not_build_in_memory_transcript(tmp_path):
    store = ResultStore(temp_dir=tmp_path)
    try:
        item = store.create('long')
        for i in range(6 * 3600):
            item.write(turn(i, '增量 '))
        assert item.turn_count == 21600
        assert not any(isinstance(value, list) for value in vars(item).values())
        # Consume streaming output in chunks; no complete result is loaded here.
        assert sum(len(part) for part in item.snapshot().iter_text()) == len('增量 '.encode()) * 21600
    finally:
        store.close()
