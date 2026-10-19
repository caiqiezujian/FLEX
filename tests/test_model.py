"""CPU tests: protocol and vLLM request wiring, without loading model weights."""

from dataclasses import replace
import re
import sys
import types

import numpy as np
import pytest

from flex_web.model import (
    AUDIO_PLACEHOLDER, BASE_SPECIALS, ContextLimit, ModelEngine, ModelOptions,
    PROTOCOL_IDS, PROTOCOL_TAGS, ProtocolError, load_tokenizer_checked, parse_reply,
)


class FakeTokenizer:
    def __init__(self, form='special', wrong_ids=False, wrong_layout=False):
        self.specials = {name: i + 100 for i, name in enumerate(BASE_SPECIALS)}
        if form == 'special':
            self.specials.update(dict(zip(PROTOCOL_TAGS, PROTOCOL_IDS)))
        elif form == 'mixed':
            self.specials[PROTOCOL_TAGS[0]] = PROTOCOL_IDS[0]
        if wrong_ids:
            self.specials[PROTOCOL_TAGS[0]] += 10
        self.inverse = {value: key for key, value in self.specials.items()}
        self.wrong_layout = wrong_layout

    def encode(self, text, add_special_tokens=False):
        pattern = '(' + '|'.join(re.escape(k) for k in self.specials) + ')'
        result = []
        for piece in re.split(pattern, text):
            if piece in self.specials:
                result.append(self.specials[piece])
            else:
                result.extend(ord(c) + 1000 for c in piece)
        return result

    def decode(self, ids, **kwargs):
        return ''.join(self.inverse[i] if i in self.inverse else chr(i - 1000) for i in ids)

    def convert_tokens_to_ids(self, text):
        return self.specials[text]

    def apply_chat_template(self, *args, **kwargs):
        body = 'wrong' if self.wrong_layout else AUDIO_PLACEHOLDER
        return f'<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n'


@pytest.fixture
def fake_transformers(monkeypatch):
    tokenizers = []
    module = types.ModuleType('transformers')

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            assert kwargs == {'local_files_only': True}
            return tokenizers[-1]

    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, 'transformers', module)
    return tokenizers


@pytest.fixture
def fake_vllm(monkeypatch, tmp_path, fake_transformers):
    tok = FakeTokenizer()
    fake_transformers.append(tok)
    observed = types.SimpleNamespace(llms=[], counters=[], requests=[], count_requests=[],
                                     shutdowns=0, mismatch=0, finish_reason='stop')
    pad = tok.convert_tokens_to_ids('<|audio_pad|>')

    def expand(request):
        arrays = iter(request['multi_modal_data']['audio'])
        result = []
        for token in request['prompt_token_ids']:
            if token == pad:
                result.extend([pad] * max(1, len(next(arrays)) // 1600))
            else:
                result.append(token)
        return result

    def shutdown():
        observed.shutdowns += 1

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            observed.llms.append(self)
            self.llm_engine = types.SimpleNamespace(
                model_config=object(), engine_core=types.SimpleNamespace(shutdown=shutdown))

        def generate(self, requests, sampling, use_tqdm):
            assert use_tqdm is False
            assert sampling.kwargs['skip_special_tokens'] is False
            observed.requests.extend(requests)
            ids = expand(requests[0]) + [999] * observed.mismatch
            completion = types.SimpleNamespace(
                token_ids=tok.encode('<|DIRECT|> leading space<|im_end|>'),
                finish_reason=observed.finish_reason)
            return [types.SimpleNamespace(prompt_token_ids=ids, outputs=[completion])]

    class FakePreprocessor:
        def __init__(self, config, tokenizer, **kwargs):
            assert kwargs == {'mm_processor_cache': None}
            self.config = config
            observed.counters.append(self)

        def preprocess(self, probe):
            observed.count_requests.append(probe)
            return {'prompt_token_ids': expand(probe)}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType('vllm')
    module.__version__ = '0.11.1'
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    inputs = types.ModuleType('vllm.inputs')
    inputs.TokensPrompt = dict
    preprocess = types.ModuleType('vllm.inputs.preprocess')
    preprocess.InputPreprocessor = FakePreprocessor
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setitem(sys.modules, 'vllm.inputs', inputs)
    monkeypatch.setitem(sys.modules, 'vllm.inputs.preprocess', preprocess)
    options = ModelOptions(model=str(tmp_path))
    return options, observed, tok


@pytest.mark.parametrize('reply,expected', [
    ('<|WAIT|>', ('<|WAIT|>', '')),
    ('  <|WAIT|> \n', ('<|WAIT|>', '')),
    ('<|DIRECT|> leading ', ('<|DIRECT|>', ' leading ')),
    ('\n<|FLEX|>部分', ('<|FLEX|>', '部分')),
])
def test_parse_preserves_translation_whitespace(reply, expected):
    assert parse_reply(reply) == expected


@pytest.mark.parametrize('reply', [
    None, '', 'explanation <|DIRECT|>hello', '<|WAIT|>text', '<|DIRECT|>  ',
    '<|FLEX|>', '<|DIRECT|>a<|WAIT|>', '<|FLEX|>x<|im_end|>',
])
def test_parse_rejects_non_protocol(reply):
    with pytest.raises(ProtocolError):
        parse_reply(reply)


@pytest.mark.parametrize('form', ['special', 'text'])
def test_original_tokenizer_supported_forms(fake_transformers, form):
    tok = FakeTokenizer(form)
    fake_transformers.append(tok)
    assert load_tokenizer_checked('local', 'auto') is tok
    assert load_tokenizer_checked('local', form) is tok


@pytest.mark.parametrize('tokenizer,declared', [
    (FakeTokenizer('mixed'), 'auto'),
    (FakeTokenizer('text'), 'special'),
    (FakeTokenizer('special'), 'text'),
    (FakeTokenizer(wrong_ids=True), 'auto'),
    (FakeTokenizer(wrong_layout=True), 'auto'),
])
def test_tokenizer_mismatch_not_patched(fake_transformers, tokenizer, declared):
    original = dict(tokenizer.specials)
    fake_transformers.append(tokenizer)
    with pytest.raises(RuntimeError):
        load_tokenizer_checked('local', declared)
    assert tokenizer.specials == original


def test_one_engine_two_directions_and_shutdown(fake_vllm, monkeypatch):
    options, observed, _ = fake_vllm
    monkeypatch.delenv('VLLM_WORKER_MULTIPROC_METHOD', raising=False)
    engine = ModelEngine(options)
    en, zh = engine.new_session('en'), engine.new_session('zh')
    assert len(observed.llms) == 1
    assert '翻译为英文' in en.system_prompt
    assert '翻译为中文' in zh.system_prompt
    assert en.history is not zh.history
    opts = observed.llms[0].kwargs
    assert opts['tensor_parallel_size'] == 1
    assert opts['max_num_seqs'] == 1
    assert opts['limit_mm_per_prompt']['image'] == 0
    assert opts['limit_mm_per_prompt']['video'] == 0
    engine.close()
    engine.close()
    assert observed.shutdowns == 1
    with pytest.raises(RuntimeError, match='closed'):
        engine.new_session('zh')


@pytest.mark.parametrize('image_slots', [0, 1])
def test_image_initialization_option_does_not_add_image_input(fake_vllm, image_slots):
    options, observed, _ = fake_vllm
    engine = ModelEngine(replace(options, image_slots=image_slots))
    try:
        assert observed.llms[0].kwargs['limit_mm_per_prompt']['image'] == image_slots
        assert engine.metadata['image_slots'] == image_slots
        session = engine.new_session('zh')
        session.generate(np.zeros(16000, np.float32))
        assert set(observed.requests[-1]['multi_modal_data']) == {'audio'}
        session.close()
    finally:
        engine.close()


def test_exact_expansion_cached_only_on_commit(fake_vllm):
    options, observed, tok = fake_vllm
    engine = ModelEngine(options)
    session = engine.new_session('en')
    audio = np.zeros(16000, np.float32)
    result = session.generate(audio)
    assert result.reply == '<|DIRECT|> leading space'
    assert result.token_ids == tok.encode(result.reply)
    assert result.audio_extra_tokens == 9
    assert session.committed_audio_extra_tokens == 0
    assert len(observed.count_requests[0]['multi_modal_data']['audio']) == 1
    session.commit(audio, result)
    assert session.committed_audio_extra_tokens == 9
    again = session.generate(audio)
    assert len(observed.count_requests[1]['multi_modal_data']['audio']) == 1
    assert len(observed.requests[1]['multi_modal_data']['audio']) == 2
    expected = len(observed.requests[1]['prompt_token_ids']) + 18
    assert again.prompt_tokens == expected
    session.commit(audio, again)
    assert len(session.audios) == 2


def test_context_preflight_does_not_submit_over_budget(fake_vllm):
    options, observed, _ = fake_vllm
    engine = ModelEngine(options)
    session = engine.new_session('zh')
    text_size = len(session.history + engine.audio_turn + engine.asst_head)
    # Text alone fits, but exact expanded audio + generation reserve does not.
    engine.options = replace(options, max_model_len=text_size + options.max_tokens + 8)
    with pytest.raises(ContextLimit, match='expanded prompt'):
        session.generate(np.zeros(16000, np.float32))
    assert not observed.requests
    assert not session.audios


def test_expansion_disagreement_rejects_reply(fake_vllm):
    options, observed, _ = fake_vllm
    engine = ModelEngine(options)
    observed.mismatch = 1
    session = engine.new_session('zh')
    with pytest.raises(RuntimeError, match='expansion mismatch'):
        session.generate(np.zeros(16000, np.float32))
    assert not session.audios


def test_truncated_output_not_committed(fake_vllm):
    options, observed, _ = fake_vllm
    engine = ModelEngine(options)
    observed.finish_reason = 'length'
    session = engine.new_session('zh')
    audio = np.zeros(16000, np.float32)
    result = session.generate(audio)
    with pytest.raises(ProtocolError, match='finish_reason'):
        session.commit(audio, result)
    assert not session.audios


def test_mock_history_includes_wait_and_preserves_leading_spaces():
    engine = ModelEngine(backend='mock')
    session = engine.new_session('en')
    audio = np.zeros(16000, np.float32)
    tags, deltas = [], []
    for _ in range(3):
        result = session.generate(audio)
        tag, delta = parse_reply(result.reply)
        tags.append(tag)
        deltas.append(delta)
        session.commit(audio, result)
    assert tags == ['<|FLEX|>', '<|WAIT|>', '<|DIRECT|>']
    assert len(session.audios) == 3
    assert deltas[1] == ''
    assert deltas[2].startswith(' ')
    assert 'not a translation' in deltas[2]


def test_commit_is_single_transaction_and_audio_is_copied():
    engine = ModelEngine(backend='mock')
    session = engine.new_session('zh')
    audio = np.zeros(16000, np.float32)
    result = session.generate(audio)
    with pytest.raises(RuntimeError, match='previous generation'):
        session.generate(audio)
    with pytest.raises(RuntimeError, match='does not belong'):
        session.commit(audio, replace(result))
    audio[0] = 1
    with pytest.raises(ValueError, match='differs'):
        session.commit(audio, result)
    audio[0] = 0
    session.commit(audio, result)
    audio[0] = 1
    assert session.audios[0][0] == 0
    with pytest.raises(RuntimeError, match='does not belong'):
        session.commit(audio, result)


def test_turn_limit_and_isolated_session_reset():
    engine = ModelEngine(ModelOptions(max_audio_turns=1), 'mock')
    session = engine.new_session('zh')
    audio = np.zeros(16000, np.float32)
    result = session.generate(audio)
    session.commit(audio, result)
    with pytest.raises(ContextLimit):
        session.generate(audio)
    other = engine.new_session('en')
    assert len(session.audios) == 1
    assert not other.audios
    session.close()
    assert not session.audios and not session.history
    with pytest.raises(RuntimeError, match='closed'):
        session.generate(audio)
    assert parse_reply(other.generate(audio).reply)[0] == '<|FLEX|>'


def test_warmup_does_not_seed_new_session():
    engine = ModelEngine(backend='mock')
    engine.warmup()
    session = engine.new_session('zh')
    assert not session.audios
    assert '【演示】' in session.generate(np.zeros(16000, np.float32)).reply


@pytest.mark.parametrize('audio', [np.array([], np.float32), np.zeros((2, 4)),
                                   np.array([np.nan]), np.zeros(48001)])
def test_invalid_audio_rejected(audio):
    with pytest.raises(ValueError):
        ModelEngine(backend='mock').new_session('zh').generate(audio)


@pytest.mark.parametrize('changes', [
    {'max_audio_turns': 0}, {'max_tokens': True}, {'max_model_len': 100},
    {'gpu_memory_utilization': 1}, {'max_chunk_seconds': float('nan')},
    {'max_chunk_seconds': 9}, {'mm_processor_cache_gb': -1},
    {'image_slots': -1}, {'image_slots': 2}, {'image_slots': True}, {'image_slots': 1.0},
])
def test_invalid_options_rejected(changes):
    with pytest.raises(ValueError):
        ModelOptions(**changes)
