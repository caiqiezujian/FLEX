"""Offline preparation regressions; duration/cutting are mocked, no API or training."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_duplex_data as prep
from duplex_audio import decode_chunk_ref


def session(audio='source.wav'):
    return {
        'id': 'source-id', 'audio': str(audio),
        'srcs': ['A', 'B'], 'chooses': [' First. ', 'Second。'],
        'tags': ['<|FLEX|>', '<|DIRECT|>'],
        'timestamps': [
            {'index': 1, 'start_ms': 0, 'end_ms': 700, 'text': 'A'},
            {'index': 2, 'start_ms': 700, 'end_ms': 1250, 'text': 'B'},
        ],
    }


def options(**overrides):
    result = dict(ffprobe='unused', ffmpeg='unused', audio_mode='virtual', audio_out=None,
                  keep_tail_silence=False, min_chunk=1, max_chunk=1, chunk_decay=0.5,
                  seed=42, target_lang='中文', typed=False, dry_run=False)
    result.update(overrides)
    return SimpleNamespace(**result)


@pytest.fixture(autouse=True)
def no_audio_execution(monkeypatch):
    monkeypatch.setattr(prep, 'probe_duration', lambda *args: 1.25)

    def forbid(*args, **kwargs):
        raise AssertionError('Tests must not invoke ffmpeg/audio cutting')

    monkeypatch.setattr(prep, 'cut_audio', forbid)


def test_protocol_defines_strategy_independently_of_sentence_completeness():
    prompt = prep.build_system_prompt('德文')
    assert '德文' in prompt
    assert '暂缓部分内容' in prompt
    assert '调整表达结构' in prompt
    assert 'DIRECT与FLEX按策略区分，不按句子是否完整区分' in prompt
    assert '语义尚不完整' not in prompt


@pytest.mark.parametrize('text', ['完整句子。', 'A whole sentence.', '  punctuation！？；… \n', ' a fragment, '])
@pytest.mark.parametrize('tag', ['<|DIRECT|>', '<|FLEX|>'])
def test_translation_content_is_preserved_exactly(text, tag):
    turns, warnings = prep.build_turns(1, [0], [text], [tag])
    assert turns == [tag + text]
    assert warnings == []


def test_merge_retains_last_policy_and_all_source_provenance():
    texts, tags = ['First。 ', '', 'second'], ['<|FLEX|>', '<|WAIT|>', '<|DIRECT|>']
    turns, warnings = prep.build_turns(2, [1, 1, 1], texts, tags)
    assert turns == ['<|WAIT|>', '<|DIRECT|>First。 second']
    assert len(warnings) == 1 and 'last_nonempty' in warnings[0]
    provenance = prep.build_turn_provenance(2, [1, 1, 1], texts, tags)
    assert provenance[0]['source_chunk_indices'] == []
    assert provenance[1]['source_chunk_indices'] == [0, 1, 2]
    assert provenance[1]['source_tags'] == tags
    assert provenance[1]['mixed_source_tags'] is True
    assert provenance[1]['mixed_emission_tags'] is True


@pytest.mark.parametrize('text,tag', [
    ('text', '<|WAIT|>'), ('', '<|FLEX|>'), (' \n', '<|DIRECT|>'),
    (None, '<|DIRECT|>'), (123, '<|FLEX|>'), ('text', '<|UNKNOWN|>'),
    ('text<|WAIT|>', '<|DIRECT|>'), ('<|im_end|>', '<|FLEX|>'),
])
def test_invalid_tag_text_combinations_are_rejected(text, tag):
    value = session()
    value['chooses'][0], value['tags'][0] = text, tag
    with pytest.raises(ValueError):
        prep.parse_session(value)
    with pytest.raises(ValueError):
        prep.build_turns(1, [0], [text], [tag])


@pytest.mark.parametrize('key,value', [
    ('start_ms', -1), ('end_ms', -1), ('start_ms', float('nan')),
    ('end_ms', float('inf')), ('end_ms', float('-inf')), ('end_ms', True),
    ('end_ms', '700'), ('start_ms', None), ('end_ms', 0),
])
def test_invalid_numeric_timestamps_are_rejected(key, value):
    raw = session()
    raw['timestamps'][0][key] = value
    with pytest.raises(ValueError):
        prep.parse_session(raw)


def test_complete_unique_indices_reorder_safely_even_when_counts_match():
    raw = session()
    raw['timestamps'].reverse()
    _, _, _, _, ends, warnings = prep.parse_session(raw)
    assert ends == [700, 1250]
    assert len(warnings) == 1 and 'reordered' in warnings[0]
    assert raw['timestamps'][0]['index'] == 2  # Raw input is not mutated.


@pytest.mark.parametrize('indices', [[1, 1], [1, 3], [1, None], [0, 2], [1.5, 2], [True, 2]])
def test_duplicate_missing_or_invalid_indices_are_rejected(indices):
    raw = session()
    for timestamp, index in zip(raw['timestamps'], indices):
        timestamp['index'] = index
    with pytest.raises(ValueError):
        prep.parse_session(raw)


def test_extra_timestamp_is_rejected_instead_of_silently_dropped():
    raw = session()
    raw['timestamps'].append({'index': 3, 'start_ms': 1250, 'end_ms': 1300, 'text': 'extra'})
    with pytest.raises(ValueError, match='count'):
        prep.parse_session(raw)


def test_mismatched_timestamp_text_is_rejected():
    raw = session()
    raw['timestamps'][0]['text'] = 'B'
    with pytest.raises(ValueError, match=r'text != srcs'):
        prep.parse_session(raw)


def test_unindexed_out_of_order_or_overlap_is_rejected():
    raw = session()
    for timestamp in raw['timestamps']:
        timestamp.pop('index')
        timestamp.pop('text')
    raw['timestamps'].reverse()
    with pytest.raises(ValueError, match='out of order'):
        prep.parse_session(raw)
    raw = session()
    raw['timestamps'][1]['start_ms'] = 699
    with pytest.raises(ValueError, match='overlapping'):
        prep.parse_session(raw)


def test_timestamp_after_real_audio_but_before_rounded_second_is_rejected():
    raw = session()
    raw['timestamps'][1]['end_ms'] = 1251
    with pytest.raises(ValueError, match='beyond actual audio'):
        prep.process_session(raw, options(), 'source', [])


def test_huge_timestamp_is_rejected_without_overflowing_the_validator():
    raw = session()
    raw['timestamps'][1]['end_ms'] = 10 ** 400
    with pytest.raises(ValueError, match='beyond actual audio'):
        prep.process_session(raw, options(), 'source', [])


@pytest.mark.parametrize('duration', [0, -1, True, float('nan'), float('inf')])
def test_invalid_audio_duration_is_rejected(monkeypatch, duration):
    monkeypatch.setattr(prep, 'probe_duration', lambda *args: duration)
    with pytest.raises(ValueError, match='finite and positive'):
        prep.process_session(session(), options(), 'source', [])


def test_assignment_rejects_uncovered_or_reversed_source_ends():
    with pytest.raises(ValueError, match='exceeds'):
        prep.assign_due_chunks([5000], [1000, 2000])
    with pytest.raises(ValueError, match='ordered'):
        prep.assign_due_chunks([1500, 1000], [1000, 2000])


@pytest.mark.parametrize('typed', [False, True])
def test_virtual_rows_reference_intervals_and_pad_only_the_final_fraction(tmp_path, typed):
    audio = tmp_path / 'original.wav'
    audio.write_bytes(b'original bytes are never touched')
    row, sizes, turns, duration = prep.process_session(session(audio), options(typed=typed), 'source', [])
    assert sizes == [1, 1] and duration == 1.25
    assert turns == ['<|FLEX|> First. ', '<|DIRECT|>Second。']
    references = ([message['content'][0]['audio'] for message in row['messages'] if message['role'] == 'user']
                  if typed else row['audios'])
    decoded = [decode_chunk_ref(value) for value in references]
    assert [(r['start'], r['end'], r['input_seconds']) for r in decoded] == [(0, 1, 1), (1, 1.25, 1)]
    assert all(r['path'] == str(audio) for r in decoded)
    assert row['audio_segments'][1] == {'path': str(audio), 'start_seconds': 1,
                                        'end_seconds': 1.25, 'input_seconds': 1}
    assert row['turn_provenance'][1]['source_chunk_indices'] == [1]
    assert audio.read_bytes() == b'original bytes are never touched'
    assert list(tmp_path.iterdir()) == [audio]


def test_tail_silence_is_wait_and_references_never_exceed_duration(monkeypatch):
    monkeypatch.setattr(prep, 'probe_duration', lambda *args: 3.25)
    row, _, turns, _ = prep.process_session(session(), options(keep_tail_silence=True), 'source', [])
    assert turns[-2:] == ['<|WAIT|>', '<|WAIT|>']
    assert row['audio_segments'][-1]['end_seconds'] == 3.25
    assert all(segment['end_seconds'] <= 3.25 for segment in row['audio_segments'])


def test_sid_allocation_prevents_natural_suffix_collisions():
    seen, used = {}, set()
    ids = [prep.allocate_session_id(audio, seen, used)
           for audio in ['/a/foo.wav', '/b/foo_2.wav', '/c/foo.wav', '/d/foo_3.wav']]
    assert ids == ['foo', 'foo_2', 'foo_3', 'foo_3_2']
    assert len(ids) == len(set(ids))


def test_files_mode_remains_available_without_colliding_paths(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(prep, 'cut_audio', lambda *args: captured.append(args))
    source = tmp_path / 'input.jsonl'
    source.write_text('\n'.join(json.dumps(session(audio)) for audio in
                                ['/a/foo.wav', '/b/foo_2.wav', '/c/foo.wav']), encoding='utf-8')
    output = tmp_path / 'output.jsonl'
    assert prep.main([str(source), '--output', str(output), '--audio-mode', 'files',
                      '--audio-out', str(tmp_path / 'chunks'), '--max-chunk', '1']) == 0
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert [row['id'] for row in rows] == ['foo', 'foo_2', 'foo_3']
    assert len({args[-1] for args in captured}) == 6
    assert all(row['audio_mode'] == 'files' for row in rows)


def test_tolerant_json_keeps_string_contents_with_comma_brackets():
    raw = '{"translation":"literal ,] and ,} with \\\"quote\\\"", "items":[1,],}'
    assert prep.tolerant_loads(raw) == {'translation': 'literal ,] and ,} with "quote"', 'items': [1]}


def test_jsonl_loader_is_lazy(tmp_path, monkeypatch):
    source = tmp_path / 'input.jsonl'
    source.write_text('\n'.join(json.dumps(session()) for _ in range(3)), encoding='utf-8')
    calls = []
    original = prep.tolerant_loads

    def observe(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(prep, 'tolerant_loads', observe)
    rows = prep.load_sessions(str(source))
    assert calls == []
    assert next(rows)[0] == 'line 1'
    assert len(calls) == 1
    rows.close()


def test_main_filters_whole_sessions_and_reports_location_id_reason(tmp_path, capsys):
    audio = tmp_path / 'source.wav'
    audio.write_bytes(b'preserved original')
    valid = session(audio)
    invalid = deepcopy(valid)
    invalid['id'] = 'bad-timing'
    invalid['timestamps'][1]['end_ms'] = 1600
    source = tmp_path / 'input.jsonl'
    source.write_text('\n'.join([json.dumps(valid), json.dumps(invalid), '{broken', json.dumps(valid)]), encoding='utf-8')
    output = tmp_path / 'output.jsonl'
    code = prep.main([str(source), '--output', str(output), '--min-chunk', '2', '--max-chunk', '2'])
    assert code == 0  # Ordinary filtering succeeds when at least one valid row remains.
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    rejected = [json.loads(line) for line in Path(str(output) + '.rejected.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2 and [r['line'] for r in rejected] == [2, 3]
    assert rejected[0]['source_id'] == 'bad-timing'
    assert rejected[0]['id'] == 'source_2'
    assert 'actual audio duration' in rejected[0]['reason']
    assert 'not valid JSON' in rejected[1]['reason']
    assert rows[1]['source_locator']['locator'] == 'line 4'
    assert audio.read_bytes() == b'preserved original'
    assert 'mixed-tag chunks: source=2, emission=2' in capsys.readouterr().out


@pytest.mark.parametrize('dry_run', [False, True])
def test_fail_on_reject_is_optional_and_preserves_valid_rows(tmp_path, dry_run):
    invalid = session()
    invalid['timestamps'][1]['end_ms'] = 1600
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(session()) + '\n' + json.dumps(invalid), encoding='utf-8')
    output = tmp_path / 'output.jsonl'
    args = [str(source), '--output', str(output)] + (['--dry-run'] if dry_run else [])
    assert prep.main(args) == 0
    assert prep.main(args + ['--fail-on-reject']) == 1
    if dry_run:
        assert not output.exists()
    else:
        rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 1
        report = Path(str(output) + '.rejected.jsonl')
        assert len(report.read_text(encoding='utf-8').splitlines()) == 1


@pytest.mark.parametrize('dry_run', [False, True])
def test_all_rejected_is_unsuccessful_even_without_strict_mode(tmp_path, dry_run):
    invalid = session()
    invalid['timestamps'][1]['end_ms'] = 1600
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(invalid), encoding='utf-8')
    output = tmp_path / 'output.jsonl'
    args = [str(source), '--output', str(output)] + (['--dry-run'] if dry_run else [])
    assert prep.main(args) == 1
    if dry_run:
        assert not output.exists()
    else:
        assert output.read_text(encoding='utf-8') == ''
        assert len(Path(str(output) + '.rejected.jsonl').read_text(encoding='utf-8').splitlines()) == 1


def test_main_writes_each_row_before_reading_the_next(monkeypatch, tmp_path):
    sinks = {}

    @contextmanager
    def capture(path):
        sink = sinks.setdefault(path, io.StringIO())
        yield sink

    output = str(tmp_path / 'stream.jsonl')

    def source(path, on_error=None):
        yield 'line 1', session()
        assert len(sinks[output].getvalue().splitlines()) == 1
        numeric_id = session()
        numeric_id['id'] = 123
        yield 'line 2', numeric_id
        assert len(sinks[output].getvalue().splitlines()) == 2

    monkeypatch.setattr(prep, 'atomic_text_output', capture)
    monkeypatch.setattr(prep, 'load_sessions', source)
    assert prep.main([str(tmp_path / 'input.jsonl'), '--output', output]) == 0
    rows = [json.loads(line) for line in sinks[output].getvalue().splitlines()]
    assert rows[1]['source_locator']['source_id'] == '123'  # Uniform Arrow column type.


def test_dry_run_creates_no_output_or_audio_files(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(session()), encoding='utf-8')
    assert prep.main([str(source), '--output', str(tmp_path / 'unused.jsonl'), '--dry-run']) == 0
    assert list(tmp_path.iterdir()) == [source]


def test_atomic_output_keeps_previous_output_if_iteration_fails(tmp_path):
    path = tmp_path / 'output.jsonl'
    path.write_text('previous', encoding='utf-8')
    with pytest.raises(RuntimeError):
        with prep.atomic_text_output(str(path)) as out:
            out.write('partial')
            raise RuntimeError('input failed')
    assert path.read_text(encoding='utf-8') == 'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_whole_second_chunk_plan_still_tiles_the_timeline():
    sizes = prep.plan_chunks(37, random.Random('reproducible'), 1, 3)
    assert sum(sizes) == 37 and all(1 <= size <= 3 for size in sizes)
    assert sizes == prep.plan_chunks(37, random.Random('reproducible'), 1, 3)
