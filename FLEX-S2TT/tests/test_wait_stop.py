"""Standalone WAIT termination and exact EOF coverage, without GPU dependencies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model = load_module('_standalone_wait_model', ROOT / 'flex_model.py')
with mock.patch.dict(sys.modules, {'flex_model': model}):
    cli = load_module('_standalone_wait_cli', ROOT / 'infer_audio.py')


class Tokenizer:
    def __init__(self, form):
        self.specials = {name: index + 100 for index, name in enumerate(model.BASE_SPECIALS)}
        if form == 'special':
            self.specials.update(zip(model.PROTOCOL_TAGS, model.PROTOCOL_IDS))
        self.inverse = {value: key for key, value in self.specials.items()}

    def encode(self, text, **kwargs):
        pattern = '(' + '|'.join(re.escape(name) for name in self.specials) + ')'
        tokens = []
        for part in re.split(pattern, text):
            if part in self.specials:
                tokens.append(self.specials[part])
            else:
                tokens.extend(ord(char) + 1000 for char in part)
        return tokens

    def decode(self, tokens, **kwargs):
        return ''.join(self.inverse[token] if token in self.inverse else chr(token - 1000)
                       for token in tokens)

    def convert_tokens_to_ids(self, token):
        return self.specials[token]

    def apply_chat_template(self, *args, **kwargs):
        return '<|im_start|>user\n' + model.AUDIO_PLACEHOLDER + '<|im_end|>\n'


@contextlib.contextmanager
def fake_engine(directory, replies, form='special'):
    """Model the documented stop behavior, including the tag in output IDs."""
    tok = Tokenizer(form)
    observed = types.SimpleNamespace(calls=[], consumed=[], replies=iter(replies), tok=tok)
    pad = tok.convert_tokens_to_ids('<|audio_pad|>')

    def expand(request):
        # All requests here use one-second model inputs; token cost is synthetic.
        return [value for token in request['prompt_token_ids']
                for value in ([pad] * 10 if token == pad else [token])]

    class Preprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess(self, request):
            return {'prompt_token_ids': expand(request)}

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            self.llm_engine = types.SimpleNamespace(
                model_config=object(),
                engine_core=types.SimpleNamespace(shutdown=lambda: None))

        def generate(self, requests, sampling, **kwargs):
            request = requests[0]
            observed.calls.append((list(request['prompt_token_ids']),
                                   request['multi_modal_data']['audio'][-1].copy()))
            proposed = tok.encode(next(observed.replies))
            params = sampling.kwargs
            tokens, reason = [], 'length'
            for token in proposed[:params['max_tokens']]:
                tokens.append(token)
                if token in params.get('stop_token_ids', []):
                    reason = 'stop'
                    break
                text = tok.decode(tokens)
                matched = next((stop for stop in params.get('stop') or [] if stop in text), None)
                if matched is not None:
                    # vLLM may trim completion.text, but generated token IDs
                    # retain the stop string; the adapter decodes those IDs.
                    reason = 'stop'
                    break
            observed.consumed.append(len(tokens))
            completion = types.SimpleNamespace(token_ids=tokens, finish_reason=reason)
            return [types.SimpleNamespace(prompt_token_ids=expand(request), outputs=[completion])]

    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: tok)
    vllm = types.ModuleType('vllm')
    vllm.__version__, vllm.LLM, vllm.SamplingParams = '0.11.1', LLM, SamplingParams
    inputs = types.ModuleType('vllm.inputs')
    inputs.TokensPrompt = dict
    preprocess = types.ModuleType('vllm.inputs.preprocess')
    preprocess.InputPreprocessor = Preprocessor
    modules = {'transformers': transformers, 'vllm': vllm,
               'vllm.inputs': inputs, 'vllm.inputs.preprocess': preprocess}
    with mock.patch.dict(sys.modules, modules), mock.patch.dict('os.environ'):
        engine = model.ModelEngine(model.ModelOptions(model=str(directory), protocol_form=form))
        try:
            yield engine, observed
        finally:
            engine.close()


def write_wav(path, seconds):
    samples = (np.arange(round(seconds * model.SAMPLE_RATE)) % 12000 + 1).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(model.SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return samples


def stream_options():
    return types.SimpleNamespace(chunk_seconds=1.0, max_audio_seconds=180.0,
                                 direction='en-zh', pace='fast')


class WaitStopTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='flex-wait-stop-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_wait_is_retained_and_next_turn_keeps_the_action(self):
        for form in ('special', 'text'):
            with self.subTest(form=form), fake_engine(
                    self.root, ['<|WAIT|>' + '\n' * 512, '<|DIRECT|>safe<|im_end|>'], form) as (engine, seen):
                self.assertIs(engine.sampling.kwargs['include_stop_str_in_output'], True)
                session = engine.new_session('en')
                self.addCleanup(session.close)
                audio = np.ones(model.SAMPLE_RATE, np.float32)
                before = list(session.history)
                result = session.generate(audio)
                self.assertEqual(result.finish_reason, 'stop')
                self.assertEqual(result.reply, '<|WAIT|>')
                self.assertEqual(result.token_ids, seen.tok.encode('<|WAIT|>'))
                self.assertEqual(seen.consumed, [len(result.token_ids)])
                session.commit(audio, result)
                expected_history = (before + engine.audio_turn + engine.asst_head
                                    + result.token_ids + engine.tail)
                self.assertEqual(session.history, expected_history)
                following = session.generate(audio)
                self.assertEqual(seen.calls[-1][0], expected_history + engine.audio_turn + engine.asst_head)
                self.assertEqual(following.reply, '<|DIRECT|>safe')
                session.commit(audio, following)
                self.assertEqual(len(session.audios), 2)

    def test_60_179_second_file_finishes_without_dropping_tail(self):
        path = self.root / 'input.wav'
        samples = write_wav(path, 60.179)
        for form in ('special', 'text'):
            replies = ['<|DIRECT|>好。<|im_end|>'] * 60 + ['<|WAIT|>' + '\n' * 512]
            destination = self.root / form
            state = cli.RunState()
            with self.subTest(form=form), fake_engine(self.root, replies, form) as (engine, seen):
                with cli.AudioReader(path) as reader, cli.OutputFiles(destination) as output:
                    cli.run_stream(stream_options(), engine, reader, output, state, stdout=io.StringIO())
                events = [json.loads(line) for line in (destination / 'events.jsonl').read_text(
                    encoding='utf-8').splitlines()]
                turns = [event for event in events if event['type'] == 'turn']
                self.assertEqual(len(turns), 61)
                self.assertEqual(len(seen.calls), 61)
                self.assertEqual(turns[-1]['tag'], 'WAIT')
                self.assertEqual(turns[-1]['response'], '<|WAIT|>')
                self.assertEqual(turns[-1]['source_start_s'], 60.0)
                self.assertAlmostEqual(turns[-1]['source_end_s'], 60.179)
                self.assertEqual(turns[-1]['model_audio_s'], 1.0)
                for previous, current in zip(turns, turns[1:]):
                    self.assertEqual(previous['source_end_s'], current['source_start_s'])
                actual_tail = seen.calls[-1][1]
                np.testing.assert_array_equal(actual_tail[:2864], samples[-2864:] / np.float32(32768))
                self.assertFalse(np.any(actual_tail[2864:]))
                self.assertEqual(state.processed_samples, len(samples))
                self.assertEqual(state.read_samples, len(samples))
                self.assertEqual(events[-1]['type'], 'done')
                self.assertTrue(events[-1]['complete_input'])
                self.assertTrue(events[-1]['final_wait'])
                self.assertFalse(any(event['type'] == 'rejected_turn' for event in events))
                self.assertEqual((destination / 'translation.txt').read_text(encoding='utf-8'), '好。' * 60)

    def test_direct_and_flex_length_limit_still_rejects_unfinished_output(self):
        path = self.root / 'input.wav'
        write_wav(path, 1.0)
        for tag in ('DIRECT', 'FLEX'):
            destination = self.root / tag
            state = cli.RunState()
            with self.subTest(tag=tag), fake_engine(
                    self.root, [f'<|{tag}|>' + 'x' * 512]) as (engine, seen):
                with cli.AudioReader(path) as reader, cli.OutputFiles(destination) as output:
                    with self.assertRaisesRegex(model.ProtocolError, "finish_reason='length'"):
                        cli.run_stream(stream_options(), engine, reader, output, state, stdout=io.StringIO())
                events = [json.loads(line) for line in (destination / 'events.jsonl').read_text(
                    encoding='utf-8').splitlines()]
                self.assertEqual(state.turns, 0)
                self.assertEqual(state.processed_samples, 0)
                self.assertEqual(state.read_samples, model.SAMPLE_RATE)
                self.assertEqual(events[-1]['type'], 'rejected_turn')
                self.assertEqual(events[-1]['finish_reason'], 'length')
                self.assertFalse(any(event['type'] == 'done' for event in events))
                self.assertEqual((destination / 'translation.txt').read_text(encoding='utf-8'), '')


if __name__ == '__main__':
    unittest.main()
