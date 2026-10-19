"""CPU-only file replay tests: no checkpoint, GPU, microphone or real sleeps."""

import io
import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest

import infer_audio as cli
from flex_web.model import ContextLimit, Generation, ProtocolError, SAMPLE_RATE


def write_wav(path, samples, *, rate=SAMPLE_RATE, channels=1):
    samples = np.asarray(samples, dtype='<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())
    return path


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 0.25
        self.sleeps.append(seconds)
        self.now += seconds


class FakeSession:
    def __init__(self, replies, clock, service=0.0, commit_error=None):
        self.replies = iter(replies)
        self.clock = clock
        self.service = service
        self.commit_error = commit_error
        self.generated = []
        self.committed = []
        self.closed = False

    def generate(self, audio):
        self.generated.append((self.clock(), audio.copy()))
        item = next(self.replies)
        self.clock.now += self.service
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return Generation(item, [1], 42, 'stop')
        return item

    def commit(self, audio, result):
        if self.commit_error:
            raise self.commit_error
        self.committed.append((audio.copy(), result))

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, session):
        self.session = session
        self.targets = []

    def new_session(self, target):
        self.targets.append(target)
        return self.session


class MemoryOutput:
    def __init__(self):
        self.events = []
        self.text = ''

    def event(self, kind, **data):
        self.events.append({'type': kind, **data})

    def append(self, delta):
        self.text += delta


def run_fixture(tmp_path, replies, *, seconds=3.0, pace='realtime', service=0.0,
                maximum=180.0, chunk=1.0, commit_error=None):
    samples = np.arange(round(seconds * SAMPLE_RATE), dtype=np.int16)
    path = write_wav(tmp_path / 'input.wav', samples)
    args = SimpleNamespace(chunk_seconds=chunk, max_audio_seconds=maximum,
                           pace=pace, direction='zh-en')
    clock = FakeClock()
    session = FakeSession(replies, clock, service, commit_error)
    engine = FakeEngine(session)
    state = cli.RunState()
    output = MemoryOutput()
    stdout = io.StringIO()

    def run():
        with cli.AudioReader(path) as reader:
            return cli.run_stream(args, engine, reader, output, state,
                                  clock=clock, sleep=clock.sleep, stdout=stdout)

    return SimpleNamespace(run=run, clock=clock, session=session, engine=engine,
                           state=state, output=output, stdout=stdout, samples=samples)


def test_native_pcm16_exact_chunks_and_short_tail(tmp_path):
    expected = np.arange(2 * SAMPLE_RATE + 123, dtype=np.int16)
    path = write_wav(tmp_path / 'native.wav', expected)
    with cli.AudioReader(path, ffmpeg='does-not-exist') as reader:
        parts = [reader.read(SAMPLE_RATE) for _ in range(4)]
        assert [len(part) for part in parts] == [SAMPLE_RATE, SAMPLE_RATE, 123, 0]
        assert reader.samples == len(expected)
        assert parts[0].dtype == np.float32
        np.testing.assert_array_equal(np.concatenate(parts), expected.astype(np.float32) / 32768)
        assert reader.read(100).size == 0
    assert reader.wav is None


def test_empty_wav_is_not_translation(tmp_path):
    case = run_fixture(tmp_path, [], seconds=0)
    with pytest.raises(ValueError, match='no decoded samples'):
        case.run()
    assert case.session.closed
    assert not case.session.generated
    assert case.output.text == case.stdout.getvalue() == ''
    assert not case.output.events


@pytest.mark.parametrize('removed', [1, 2, 17, 200])
def test_truncated_native_wav_fails_instead_of_silent_eof(tmp_path, removed):
    path = write_wav(tmp_path / 'broken.wav', np.arange(1000, dtype=np.int16))
    path.write_bytes(path.read_bytes()[:-removed])
    with cli.AudioReader(path) as reader:
        with pytest.raises(ValueError, match='Truncated WAV|incomplete PCM16'):
            reader.read(2000)


def test_audio_reader_rejects_missing_file_and_nonpositive_count(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        with cli.AudioReader(tmp_path / 'missing.wav'):
            pass
    path = write_wav(tmp_path / 'ok.wav', [0])
    with cli.AudioReader(path) as reader:
        for count in (0, -1):
            with pytest.raises(ValueError, match='positive'):
                reader.read(count)


def test_non_native_audio_requires_ffmpeg_with_actionable_error(tmp_path, monkeypatch):
    path = write_wav(tmp_path / '48k.wav', [0] * 100, rate=48000)
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError, match='needs FFmpeg'):
        with cli.AudioReader(path):
            pass


def fake_decoder(monkeypatch, data, *, status=0):
    class ShortReadPipe(io.BytesIO):
        def read(self, count=-1):
            # Pipe reads need not fill a request or end on a PCM sample boundary.
            return super().read(min(count, 3))

    process = SimpleNamespace(stdout=ShortReadPipe(data), returncode=None, terminated=False,
                              killed=False, calls=[])

    def wait(timeout):
        process.returncode = status
        return status

    def terminate():
        process.terminated = True
        process.returncode = -15

    def kill():
        process.killed = True
        process.returncode = -9

    process.wait = wait
    process.terminate = terminate
    process.kill = kill
    process.poll = lambda: process.returncode

    def launch(command, **kwargs):
        process.calls.append((command, kwargs))
        kwargs['stderr'].write(b'decoder failure detail')
        kwargs['stderr'].flush()
        return process

    monkeypatch.setattr(cli.shutil, 'which', lambda name: '/mock/ffmpeg')
    monkeypatch.setattr(cli.subprocess, 'Popen', launch)
    return process


def test_ffmpeg_accumulates_partial_pipe_reads_and_closes_handles(tmp_path, monkeypatch):
    path = tmp_path / 'audio.flac'
    path.write_bytes(b'not a WAV')
    expected = np.array([-32768, -2, 0, 3, 32767], dtype='<i2')
    process = fake_decoder(monkeypatch, expected.tobytes())
    with cli.AudioReader(path) as reader:
        parts = [reader.read(3), reader.read(3), reader.read(3)]
        assert [len(part) for part in parts] == [3, 2, 0]
        np.testing.assert_array_equal(np.concatenate(parts), expected.astype(np.float32) / 32768)
        errors = reader.errors
    assert process.stdout.closed and errors.closed
    assert reader.process is reader.errors is None
    command, kwargs = process.calls[0]
    assert command[0] == '/mock/ffmpeg'
    assert command[command.index('-protocol_whitelist') + 1] == 'file,pipe'
    assert command[command.index('-i') + 1] == str(path.resolve())
    assert command[command.index('-ar') + 1] == str(SAMPLE_RATE)
    assert 'shell' not in kwargs


@pytest.mark.parametrize('data,status,message', [
    (b'\x00', 0, 'incomplete PCM16'),
    (b'\x00\x00', 1, 'FFmpeg failed.*decoder failure detail'),
])
def test_ffmpeg_corrupt_output_is_error_not_success(tmp_path, monkeypatch, data, status, message):
    path = tmp_path / 'audio.mp3'
    path.write_bytes(b'not a WAV')
    process = fake_decoder(monkeypatch, data, status=status)
    with cli.AudioReader(path) as reader:
        with pytest.raises(ValueError, match=message):
            reader.read(100)
    assert process.stdout.closed


def test_ffmpeg_reader_early_close_terminates_decoder(tmp_path, monkeypatch):
    path = tmp_path / 'audio.mp3'
    path.write_bytes(b'not a WAV')
    process = fake_decoder(monkeypatch, b'\x00\x00' * 100)
    with cli.AudioReader(path) as reader:
        assert len(reader.read(1)) == 1
    assert process.terminated
    assert process.stdout.closed


@pytest.mark.parametrize('service,dispatches,total_sleep,lags', [
    (0.4, [101.0, 102.0, 103.0], 2.2, [0.0, 0.0, 0.0]),
    (1.5, [101.0, 102.5, 104.0], 1.0, [0.0, 0.5, 1.0]),
])
def test_realtime_uses_absolute_deadlines_not_sleep_after_generation(
        tmp_path, service, dispatches, total_sleep, lags):
    case = run_fixture(tmp_path, ['<|DIRECT|>x'] * 3, service=service)
    case.run()
    assert [time for time, _ in case.session.generated] == pytest.approx(dispatches)
    assert sum(case.clock.sleeps) == pytest.approx(total_sleep)
    events = [event for event in case.output.events if event['type'] == 'turn']
    assert [event['dispatch_lag_s'] for event in events] == pytest.approx(lags)
    for (dispatch, _), event in zip(case.session.generated, events):
        assert dispatch >= 100 + event['source_end_s']
        assert event['service_s'] == pytest.approx(service)
        assert event['generation_finished_elapsed_s'] == pytest.approx(dispatch + service - 100)
    assert case.engine.targets == ['en']
    assert case.session.closed


def test_fast_mode_never_sleeps_and_has_no_realtime_lag(tmp_path):
    case = run_fixture(tmp_path, ['<|DIRECT|>x'] * 3, pace='fast', service=0.1)
    case.run()
    assert not case.clock.sleeps
    assert [time for time, _ in case.session.generated] == pytest.approx([100, 100.1, 100.2])
    assert all(event['dispatch_lag_s'] is None for event in case.output.events
               if event['type'] == 'turn')


def test_wait_commits_history_but_text_only_appends_exact_deltas(tmp_path):
    replies = ['<|FLEX|>He is the first', '<|WAIT|>', '<|DIRECT|> to admit it.\n']
    case = run_fixture(tmp_path, replies, pace='fast')
    case.run()
    assert [result.reply for _, result in case.session.committed] == replies
    assert case.output.text == case.stdout.getvalue() == 'He is the first to admit it.\n'
    assert case.state.deltas == ['He is the first', '', ' to admit it.\n']
    assert [event['tag'] for event in case.output.events[:-1]] == ['FLEX', 'WAIT', 'DIRECT']
    assert case.output.events[-1]['complete_input'] is True
    assert case.output.events[-1]['final_wait'] is False


def test_tail_padded_for_model_only_and_final_wait_warns_without_extra_turn(tmp_path, caplog):
    case = run_fixture(tmp_path, ['<|DIRECT|>Hello', '<|WAIT|>'], seconds=1.25)
    case.run()
    assert len(case.session.generated) == len(case.session.committed) == 2
    tail = case.session.generated[-1][1]
    assert len(tail) == SAMPLE_RATE
    np.testing.assert_array_equal(tail[:4000], case.samples[SAMPLE_RATE:].astype(np.float32) / 32768)
    assert not np.any(tail[4000:])
    event = case.output.events[-2]
    assert event['source_start_s'] == 1.0
    assert event['source_end_s'] == 1.25
    assert event['model_audio_s'] == 1.0
    assert case.state.processed_samples == 20000
    assert case.output.events[-1]['final_wait'] is True
    assert 'Final action is WAIT' in caplog.text
    assert 'No extra silence turn' in caplog.text
    assert case.output.text == 'Hello'


@pytest.mark.parametrize('bad', [
    Generation('<|DIRECT|>truncated', [1], 42, 'length'),
    '<|WAIT|>incorrect text', 'untagged text', '<|DIRECT|>',
    '<|DIRECT|>text<|FLEX|>duplicate',
])
def test_invalid_generation_is_journaled_never_committed_or_printed(tmp_path, bad):
    case = run_fixture(tmp_path, ['<|DIRECT|>safe', bad], seconds=2, pace='fast')
    with pytest.raises(ProtocolError):
        case.run()
    assert len(case.session.committed) == 1
    assert case.output.text == case.stdout.getvalue() == 'safe'
    assert case.state.turns == 1
    assert case.state.processed_samples == SAMPLE_RATE
    assert case.state.read_samples == 2 * SAMPLE_RATE
    assert case.session.closed
    assert case.output.events[-1]['type'] == 'rejected_turn'
    rejected = case.output.events[-1]
    assert rejected['response'] == (bad.reply if isinstance(bad, Generation) else bad)
    assert rejected['token_ids'] == [1]
    assert rejected['message']
    assert not any(event['type'] == 'done' for event in case.output.events)


def test_commit_protocol_failure_does_not_publish_text(tmp_path):
    case = run_fixture(tmp_path, ['<|DIRECT|>not safe'], seconds=1,
                       commit_error=ProtocolError('pending turn mismatch'))
    with pytest.raises(ProtocolError, match='mismatch'):
        case.run()
    assert case.state.turns == 0
    assert case.output.text == case.stdout.getvalue() == ''
    assert case.output.events[-1]['type'] == 'rejected_turn'
    assert case.session.closed


def test_duration_limit_keeps_valid_prefix_and_never_sends_excess_audio(tmp_path):
    case = run_fixture(tmp_path, ['<|DIRECT|>prefix'], seconds=2, maximum=1)
    with pytest.raises(ValueError, match='NOT fully translated'):
        case.run()
    assert len(case.session.generated) == len(case.session.committed) == 1
    assert case.output.text == 'prefix'
    assert case.state.processed_samples == SAMPLE_RATE
    assert case.state.read_samples == 2 * SAMPLE_RATE
    assert case.session.closed
    assert not any(event['type'] == 'done' for event in case.output.events)


def test_context_limit_keeps_valid_prefix_without_retry_reset_or_fake_wait(tmp_path):
    case = run_fixture(tmp_path, ['<|DIRECT|>prefix', ContextLimit('context budget')], seconds=2)
    with pytest.raises(ContextLimit, match='context budget'):
        case.run()
    assert len(case.session.generated) == 2
    assert len(case.session.committed) == 1
    assert case.engine.targets == ['en']
    assert case.output.text == 'prefix'
    assert case.state.last_tag == 'DIRECT'
    assert case.state.turns == 1
    assert case.session.closed


def test_output_files_preserve_whitespace_and_refuse_existing_directory(tmp_path):
    destination = tmp_path / 'new-run'
    with cli.OutputFiles(destination) as output:
        output.event('turn', delta=' 你好\n')
        output.append('Hello')
        output.append(' world.\n')
    expected = {path.name: path.read_bytes() for path in destination.iterdir()}
    assert expected['translation.txt'] == b'Hello world.\n'
    assert json.loads(expected['events.jsonl']) == {'type': 'turn', 'delta': ' 你好\n'}
    with pytest.raises(FileExistsError):
        with cli.OutputFiles(destination):
            pass
    assert expected == {path.name: path.read_bytes() for path in destination.iterdir()}


@pytest.fixture
def audio_arg(tmp_path):
    return ['--audio', str(write_wav(tmp_path / 'test.wav', [0] * 100)), '--backend', 'mock']


@pytest.mark.parametrize('extra', [
    ['--chunk-seconds', 'nan'], ['--chunk-seconds', 'inf'], ['--chunk-seconds', '0.49'],
    ['--chunk-seconds', '6.1'], ['--max-audio-seconds', '0'], ['--max-audio-seconds', 'nan'],
    ['--max-audio-seconds', 'inf'], ['--max-model-len', '160'], ['--max-audio-turns', '0'],
    ['--max-tokens', '0'], ['--gpu-memory-utilization', '1'],
    ['--gpu-memory-utilization', 'nan'], ['--mm-processor-cache-gb', '-1'],
    ['--direction', 'en-ja'], ['--protocol-form', 'invented'],
    ['--image-slots', '-1'], ['--image-slots', '2'],
])
def test_parse_args_rejects_unsafe_values(audio_arg, extra):
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(audio_arg + extra)
    assert caught.value.code == 2


def test_parse_args_requires_local_file_model_and_new_output(audio_arg, tmp_path):
    for argv in [
        ['--audio', 'https://example.com/audio.wav', '--backend', 'mock'],
        audio_arg + ['--backend', 'vllm'],
        audio_arg + ['--backend', 'vllm', '--model', str(tmp_path / 'missing')],
        audio_arg + ['--output-dir', str(tmp_path)],
    ]:
        with pytest.raises(SystemExit) as caught:
            cli.parse_args(argv)
        assert caught.value.code == 2


def test_parse_args_safe_defaults_and_explicit_real_local_checkpoint(audio_arg, tmp_path):
    first, second = cli.parse_args(audio_arg), cli.parse_args(audio_arg)
    assert first.direction == 'en-zh'
    assert first.chunk_seconds == 1.0
    assert first.pace == 'realtime'
    assert first.image_slots == cli.model_options(first).image_slots == 1
    assert first.output_dir != second.output_dir
    assert not first.output_dir.exists()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    args = cli.parse_args(audio_arg + ['--backend', 'vllm', '--model', str(checkpoint),
                                      '--protocol-form', 'text', '--no-prefix-caching'])
    options = cli.model_options(args)
    assert options.model == str(checkpoint)
    assert options.protocol_form == 'text'
    assert not options.prefix_caching


def test_image_workaround_can_be_disabled_for_fixed_vllm(audio_arg):
    args = cli.parse_args(audio_arg + ['--image-slots', '0'])
    assert cli.model_options(args).image_slots == 0


def test_main_mock_success_writes_transcript_and_explicit_mock_metadata(tmp_path):
    path = write_wav(tmp_path / 'input.wav', [0] * SAMPLE_RATE)
    destination = tmp_path / 'result'
    assert cli.main(['--audio', str(path), '--backend', 'mock', '--pace', 'fast',
                     '--no-warmup', '--output-dir', str(destination)]) == 0
    events = [json.loads(line) for line in (destination / 'events.jsonl').read_text('utf-8').splitlines()]
    assert events[0]['type'] == 'start'
    assert events[0]['mock'] is True
    assert events[0]['image_slots'] == 1
    assert events[1]['type'] == 'model_ready'
    assert events[1]['mock'] is True
    assert events[-1]['type'] == 'done'
    assert events[-1]['complete_input'] is True
    transcript = (destination / 'translation.txt').read_text('utf-8')
    assert transcript
    assert '<|' not in transcript
    assert transcript == ''.join(event['delta'] for event in events if event['type'] == 'turn')
    assert all(isinstance(event['token_ids'], list)
               for event in events if event['type'] == 'turn')


def test_main_model_initialization_error_has_nonzero_exit_and_no_fake_translation(
        tmp_path, monkeypatch):
    path = write_wav(tmp_path / 'input.wav', [0] * SAMPLE_RATE)
    destination = tmp_path / 'failure'

    def broken_engine(*args, **kwargs):
        raise RuntimeError('model could not load')

    monkeypatch.setattr(cli, 'ModelEngine', broken_engine)
    assert cli.main(['--audio', str(path), '--backend', 'mock', '--pace', 'fast',
                     '--output-dir', str(destination)]) == 1
    events = [json.loads(line) for line in (destination / 'events.jsonl').read_text('utf-8').splitlines()]
    assert [event['type'] for event in events] == ['start', 'error']
    assert events[-1]['complete_input'] is False
    assert events[-1]['code'] == 'RuntimeError'
    assert 'could not load' in events[-1]['message']
    assert (destination / 'translation.txt').read_bytes() == b''
