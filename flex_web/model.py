"""Local, text-only Qwen3-Omni inference shared by file and Web entry points.

The server owns one ModelEngine on one serial worker thread. Creating a session
changes only its private conversation, never loads another model. Audio is mono
float32 at 16 kHz. WAIT consumes an audio turn just like FLEX/DIRECT; all three
responses remain in the recent context window. Rolling windows remove complete
old turns together; they do not alter committed translations or infer semantic
coverage from an action tag. The optional stop policy preserves the old baseline.

The vLLM adapter follows the existing standalone inference script and targets the
native vLLM 0.11.1 API. GPU dependencies are imported only in the real backend.
The mock backend tests transport and UI; it NEVER claims to translate speech.
"""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import re
import time
from types import MappingProxyType

import numpy as np


LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16000
PROTOCOL_TAGS = ('<|WAIT|>', '<|DIRECT|>', '<|FLEX|>')
PROTOCOL_IDS = (151676, 151677, 151678)
AUDIO_PLACEHOLDER = '<|audio_start|><|audio_pad|><|audio_end|>'
STOP_TOKENS = ('<|im_end|>', '<|endoftext|>')
BASE_SPECIALS = ('<|im_start|>', '<|im_end|>', '<|endoftext|>',
                 '<|audio_start|>', '<|audio_pad|>', '<|audio_end|>')

SUPPORTED_DIRECTIONS = ('en-zh', 'zh-en')

# These strings match the accepted bilingual SFT prompts exactly. In particular,
# the English-target checkpoint was trained with "English", not the UI label
# "英语" or "英文". Keep display labels independent from model instructions.
_STREAMING_RESPONSE_RULES = (
    '用户将按时间顺序逐块发送音频。每收到一块音频后，立即输出响应：'
    '若该块暂无可翻译内容，输出<|WAIT|>；若仅能确认部分译文、语义尚不完整，输出<|FLEX|>并紧跟'
    '确信的译文（只输出有把握的部分，宁少勿错）；若译文语义已完整，输出<|DIRECT|>并紧跟新识别'
    '的译文。所有译文均为增量输出：只输出新增内容，不重复、不修改此前已输出的任何译文。'
)
SYSTEM_PROMPTS = MappingProxyType({
    'zh': ('你是一个同声传译专家，负责将用户音频中的语音同声翻译为中文。'
           + _STREAMING_RESPONSE_RULES),
    'en': ('你是一个同声传译专家，负责将用户音频中的语音同声翻译为English。'
           + _STREAMING_RESPONSE_RULES),
})


class ProtocolError(RuntimeError):
    """Model output cannot safely be committed to append-only translation."""


class ContextLimit(RuntimeError):
    """Session reached its explicit context/audio-turn budget."""


class ModelBackendError(RuntimeError):
    """A shared inference call failed; admission may need to be stopped.

    Unlike per-session preparation, decoding, or protocol errors, this marks
    an exception from the shared llm.generate call or an invalid batch result
    count. Its __cause__ preserves the original backend exception for logging.
    """


def parse_reply(reply):
    """Validate one action while preserving every translation-delta whitespace."""
    if not isinstance(reply, str):
        raise ProtocolError('model reply must be a string')
    content = reply.lstrip()
    tag = next((tag for tag in PROTOCOL_TAGS if content.startswith(tag)), None)
    if tag is None:
        raise ProtocolError('reply must start with WAIT, FLEX or DIRECT')
    delta = content[len(tag):]
    if re.search(r'<\|[^\r\n]*?\|>', delta):
        raise ProtocolError('nested protocol/layout special token in reply')
    if tag == '<|WAIT|>':
        if delta.strip():
            raise ProtocolError('WAIT must not contain translation text')
        return tag, ''
    if not delta.strip():
        raise ProtocolError('FLEX/DIRECT must contain a non-empty translation')
    return tag, delta


@dataclass(frozen=True)
class ModelOptions:
    model: str = ''
    dtype: str = 'bfloat16'
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_audio_turns: int = 256
    max_tokens: int = 160
    protocol_form: str = 'auto'
    max_chunk_seconds: float = 3.0
    mm_processor_cache_gb: float = 2.0
    enforce_eager: bool = True
    prefix_caching: bool = True
    seed: int = 42
    image_slots: int = 0
    context_policy: str = 'rolling'
    history_window_seconds: float = 120.0
    history_retain_seconds: float = 60.0
    max_batch_size: int = 1

    def __post_init__(self):
        if self.dtype not in ('auto', 'bfloat16', 'float16'):
            raise ValueError('dtype must be auto, bfloat16 or float16')
        if self.protocol_form not in ('auto', 'text', 'special'):
            raise ValueError('protocol_form must be auto, text or special')
        if self.context_policy not in ('rolling', 'stop'):
            raise ValueError('context_policy must be rolling or stop')
        if type(self.image_slots) is not int or self.image_slots not in (0, 1):
            raise ValueError('image_slots must be 0 or 1')
        for name in ('gpu_memory_utilization', 'max_chunk_seconds', 'mm_processor_cache_gb',
                     'history_window_seconds', 'history_retain_seconds'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        for name in ('max_model_len', 'max_audio_turns', 'max_tokens', 'max_batch_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError('gpu_memory_utilization must be between zero and one')
        if not 0.1 <= self.max_chunk_seconds <= 8:
            raise ValueError('max_chunk_seconds must be between 0.1 and 8')
        if self.mm_processor_cache_gb < 0:
            raise ValueError('mm_processor_cache_gb must be non-negative')
        if self.max_model_len <= self.max_tokens:
            raise ValueError('max_model_len must exceed max_tokens')
        if not 0 < self.history_retain_seconds < self.history_window_seconds:
            raise ValueError('history durations require 0 < retain < window seconds')


@dataclass
class Generation:
    reply: str
    token_ids: list
    prompt_tokens: int
    finish_reason: str
    audio_extra_tokens: int = 0


@dataclass(frozen=True)
class CommittedTurn:
    """One indivisible audio/assistant pair; tokens are never re-encoded."""

    audio: np.ndarray
    token_ids: tuple
    audio_extra_tokens: int


@dataclass(frozen=True)
class _WindowPlan:
    """Prospective context, applied only when its generation commits."""

    remove_count: int
    history: list
    audios: list
    audio_extra_tokens: int
    reasons: tuple = ()


def load_tokenizer_checked(model, form):
    """Read the ORIGINAL training tokenizer; never add/resize special tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tag_ids = [tok.encode(tag, add_special_tokens=False) for tag in PROTOCOL_TAGS]
    single = [len(ids) == 1 for ids in tag_ids]
    if any(single) and not all(single):
        raise RuntimeError(f'mixed protocol tokenizer forms: {tag_ids}')
    detected = 'special' if all(single) else 'text'
    if form != 'auto' and form != detected:
        raise RuntimeError(f'weights declared {form!r}, tokenizer is {detected!r}; '
                           'use the tokenizer actually used in SFT')
    for token in BASE_SPECIALS:
        ids = tok.encode(token, add_special_tokens=False)
        if len(ids) != 1 or tok.decode(ids) != token:
            raise RuntimeError(f'invalid layout token: {token} -> {ids}')
    if detected == 'special' and tuple(ids[0] for ids in tag_ids) != PROTOCOL_IDS:
        raise RuntimeError(f'protocol ids do not match training {PROTOCOL_IDS}: {tag_ids}')
    LOG.info('tokenizer protocol=%s: %s', detected, dict(zip(PROTOCOL_TAGS, tag_ids)))
    if form == 'auto':
        LOG.warning('Protocol auto checks tokenizer only, not weight training history. '
                    'Set protocol_form=text/special when known.')
    try:
        rendered = tok.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'probe.wav'}]}],
            add_generation_prompt=True, tokenize=False)
    except Exception as exc:
        LOG.warning('Chat-template probe unavailable (%s); using existing SFT layout', exc)
    else:
        match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', rendered, re.S)
        if match and match.group(1) != AUDIO_PLACEHOLDER:
            raise RuntimeError('audio chat template differs from the SFT training layout')
    return tok


class ModelEngine:
    """One local checkpoint, two supported directions, fresh private histories.

    Construct and call serially: the file CLI uses its main thread, while the
    Web server uses one worker, never FastAPI's event-loop thread. vLLM is not
    imported when backend='mock', allowing CPU-only development and tests.
    """

    def __init__(self, options=None, backend='vllm'):
        self.options = options or ModelOptions()
        if backend not in ('vllm', 'mock'):
            raise ValueError('backend must be vllm or mock')
        self.backend = backend
        self._closed = False
        self.metadata = {
            'backend': backend,
            'model_name': Path(self.options.model).name if self.options.model else 'DEMO — no model',
            'directions': list(SUPPORTED_DIRECTIONS),
            'sample_rate': SAMPLE_RATE,
            'max_model_len': self.options.max_model_len,
            'max_audio_turns': self.options.max_audio_turns,
            'max_chunk_seconds': self.options.max_chunk_seconds,
            'image_slots': self.options.image_slots,
            'context_policy': self.options.context_policy,
            'history_window_seconds': self.options.history_window_seconds,
            'history_retain_seconds': self.options.history_retain_seconds,
            'max_batch_size': self.options.max_batch_size,
            'mock': backend == 'mock',
        }
        if backend == 'mock':
            self.metadata['model_name'] = 'DEMO — transport test, not translation'
            return
        if not self.options.model or not Path(self.options.model).is_dir():
            raise ValueError('model must be an existing LOCAL merged checkpoint directory')

        # CUDA engine initialization runs on our serial worker thread; do not
        # fork a possibly CUDA-initialized, multithreaded web-server process.
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.inputs.preprocess import InputPreprocessor

        opts = self.options
        self.TokensPrompt = TokensPrompt
        self.tok = load_tokenizer_checked(opts.model, opts.protocol_form)
        self.enc = lambda text: self.tok.encode(text, add_special_tokens=False)
        self.user_head = self.enc('<|im_start|>user\n')
        self.asst_head = self.enc('<|im_start|>assistant\n')
        self.tail = self.enc('<|im_end|>\n')
        self.audio_turn = self.user_head + self.enc(AUDIO_PLACEHOLDER) + self.tail
        self.stop_ids = [self.tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        LOG.info('Loading local vLLM %s, TP=1, max_model_len=%s, eager=%s',
                 vllm.__version__, opts.max_model_len, opts.enforce_eager)
        if vllm.__version__ != '0.11.1':
            LOG.warning('This adapter targets vLLM 0.11.1; actual version is %s. '
                        'Verify native Qwen3-Omni/preprocessor compatibility.', vllm.__version__)
        if opts.image_slots:
            LOG.info('Keeping one image slot enabled for Qwen3-Omni initialization compatibility; '
                     'requests remain audio-only. This can increase GPU memory usage.')
        self.llm = LLM(
            model=opts.model, dtype=opts.dtype, tensor_parallel_size=1,
            gpu_memory_utilization=opts.gpu_memory_utilization,
            max_model_len=opts.max_model_len, max_num_seqs=opts.max_batch_size,
            enforce_eager=opts.enforce_eager, enable_prefix_caching=opts.prefix_caching,
            limit_mm_per_prompt={
                'audio': {'count': opts.max_audio_turns,
                          'length': round(opts.max_chunk_seconds * SAMPLE_RATE)},
                'image': opts.image_slots, 'video': 0},
            mm_processor_cache_gb=opts.mm_processor_cache_gb, seed=opts.seed)
        # Never preprocess separately through the live engine's mirrored MM
        # sender cache. This independent, uncached counter touches only new audio.
        self.counter = InputPreprocessor(self.llm.llm_engine.model_config, self.tok,
                                         mm_processor_cache=None)
        self.sampling = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
            max_tokens=opts.max_tokens, seed=opts.seed,
            stop_token_ids=self.stop_ids, skip_special_tokens=False)
        self.metadata['vllm_version'] = vllm.__version__

    def new_session(self, target_lang):
        if self._closed:
            raise RuntimeError('model engine is closed')
        if target_lang not in ('zh', 'en'):
            raise ValueError('target_lang must be zh or en')
        return ModelSession(self, target_lang)

    def generate_batch(self, items):
        """Generate at most one pending turn per private session in one call.

        Call only on the engine's serial worker. vLLM's synchronous generate
        returns results in input order; this is bounded microbatching, not an
        AsyncLLM or a promise of real-time throughput. Preparation/decoding
        failures affect only their corresponding item. A backend batch failure
        returns ModelBackendError for every submitted item without committing
        or retrying any of them; its cause retains the original exception.
        The caller still validates/commits each returned Generation, just as for
        ModelSession.generate. Pending replies are never published here.
        """
        items = list(items)
        if not items:
            return []
        if len(items) > self.options.max_batch_size:
            return [ValueError('batch exceeds configured max_batch_size') for _ in items]
        if self._closed:
            return [RuntimeError('model engine is closed') for _ in items]
        results = [None] * len(items)
        sessions = [item[0] if isinstance(item, (tuple, list)) and len(item) == 2
                    else None for item in items]
        # Reject ALL occurrences of a duplicate before preparing either turn.
        # Otherwise two replies could race to occupy one session's pending slot.
        counts = {}
        for session in sessions:
            counts[id(session)] = counts.get(id(session), 0) + 1
        prepared = []
        for index, (item, session) in enumerate(zip(items, sessions)):
            try:
                if not isinstance(session, ModelSession) or session.engine is not self:
                    raise ValueError('batch item must use a session owned by this engine')
                if counts[id(session)] != 1:
                    raise ValueError('duplicate session in batch')
                value = session._prepare_generation(item[1])
                prepared.append((index, session, value))
            except Exception as exc:
                results[index] = exc
        if not prepared:
            return results
        if self.backend == 'mock':
            for index, session, value in prepared:
                try:
                    results[index] = session._finish_generation(value)
                except Exception as exc:
                    results[index] = exc
            return results
        try:
            outputs = self.llm.generate([value[3] for _, _, value in prepared],
                                        self.sampling, use_tqdm=False)
            if len(outputs) != len(prepared):
                raise RuntimeError('backend batch output count does not match request count')
        except Exception as exc:
            error = ModelBackendError(f'shared inference backend failed: {exc}')
            error.__cause__ = exc
            for index, _, _ in prepared:
                results[index] = error
            return results
        for (index, session, value), output in zip(prepared, outputs):
            try:
                results[index] = session._finish_generation(value, output)
            except Exception as exc:
                results[index] = exc
        return results

    def close(self):
        """Release vLLM worker resources after every session/inference has ended.

        Call on the same serial executor as generate, not concurrently with it.
        engine_core.shutdown is the vLLM 0.11.1 EngineCoreClient cleanup API.
        """
        if self._closed:
            return
        self._closed = True
        if self.backend == 'vllm' and hasattr(self, 'llm'):
            core = getattr(self.llm.llm_engine, 'engine_core', None)
            shutdown = getattr(core, 'shutdown', None)
            if callable(shutdown):
                shutdown()
            else:
                LOG.warning('vLLM engine_core.shutdown unavailable in this build; '
                            'worker cleanup will depend on process exit')

    def warmup(self, seconds=1.0):
        """Discard one generation BEFORE capture; never contaminate real history."""
        if not math.isfinite(seconds) or not 0 < seconds <= self.options.max_chunk_seconds:
            raise ValueError('warmup duration exceeds allowed audio chunk duration')
        session = self.new_session('zh')
        try:
            session.generate(np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32))
        finally:
            session.close()

    def audio_expansion(self, audio):
        probe = self.TokensPrompt(prompt_token_ids=self.audio_turn,
                                 multi_modal_data={'audio': [audio]})
        expanded = self.counter.preprocess(probe)['prompt_token_ids']
        pad_id = self.tok.convert_tokens_to_ids('<|audio_pad|>')
        if self.audio_turn.count(pad_id) != 1 or expanded.count(pad_id) < 1:
            raise RuntimeError('unexpected audio placeholder expansion in this vLLM build')
        if [i for i in expanded if i != pad_id] != [i for i in self.audio_turn if i != pad_id]:
            raise RuntimeError('audio preprocessing changed non-audio tokens; context count is unsafe')
        return len(expanded) - len(self.audio_turn)


class ModelSession:
    """Bounded recent context, independent of the full append-only transcript.

    The window is an application-level prompt reconstruction, not native KV
    eviction. Dropping a turn says nothing about semantic translation coverage:
    WAIT, FLEX and DIRECT are treated identically. Long-audio quality must be
    evaluated on the actual checkpoint, particularly near window boundaries.
    """

    def __init__(self, engine, target_lang):
        self.engine = engine
        self.target_lang = target_lang
        self.system_prompt = SYSTEM_PROMPTS[target_lang]
        self.audios = []
        self._turns = []
        self.committed_audio_extra_tokens = 0
        self.total_committed_turns = 0
        self._total_audio_samples = 0
        self._retained_audio_samples = 0
        self._dropped_audio_samples = 0
        self._dropped_turns = 0
        self._rollovers = 0
        self._last_rollover = None
        self._closed = False
        self._pending = None
        if engine.backend == 'vllm':
            self.history = (engine.enc('<|im_start|>system\n')
                            + engine.enc(self.system_prompt) + engine.tail)
        else:
            self.history = []
        self._system_tokens = tuple(self.history)

    @property
    def context_metadata(self):
        """A bounded-size audit snapshot; counters remain monotonic on rollover.

        retained_context_tokens counts the committed expanded context only, not
        the next input, generation head or output reserve. Mock counts are
        synthetic and cannot be used for GPU sizing.
        """
        tokens = len(self.history) + self.committed_audio_extra_tokens
        if self.engine.backend == 'mock' and not self._closed:
            tokens += 200
        return {
            'policy': self.engine.options.context_policy,
            'rollovers': self._rollovers,
            'dropped_turns': self._dropped_turns,
            'dropped_audio_seconds': self._dropped_audio_samples / SAMPLE_RATE,
            'retained_turns': len(self._turns),
            'retained_audio_seconds': self._retained_audio_samples / SAMPLE_RATE,
            'retained_context_tokens': tokens,
            'total_committed_turns': self.total_committed_turns,
            'total_committed_audio_seconds': self._total_audio_samples / SAMPLE_RATE,
            'last_rollover': (None if self._last_rollover is None else
                              {**self._last_rollover,
                               'reasons': list(self._last_rollover['reasons'])}),
        }

    def _turn_tokens(self, turn):
        if self.engine.backend == 'vllm':
            return (self.engine.audio_turn + self.engine.asst_head
                    + list(turn.token_ids) + self.engine.tail)
        return list(turn.token_ids)

    def _prepare_window(self, audio, extra):
        """Plan a whole-turn trim without mutating committed state.

        We reserve the incoming audio and maximum reply before choosing the
        window. A single impossible request fails without discarding history.
        On rollover, time, token and turn low-water marks avoid trimming one
        old turn on every subsequent request. System tokens are never removed.
        """
        opts, engine = self.engine.options, self.engine
        if engine.backend == 'vllm':
            base = len(self._system_tokens) + len(engine.audio_turn) + len(engine.asst_head)
            closing_tokens = len(engine.tail)
        else:
            base = 200
            closing_tokens = 0
        # The committed assistant terminator also lives in the next request's
        # history. Reserving it keeps even a maximum-length valid reply inside
        # the context bound, rather than briefly overshooting after commit.
        bare_required = base + extra + opts.max_tokens + closing_tokens
        if bare_required > opts.max_model_len:
            raise ContextLimit(
                f'expanded prompt={base + extra} + max_tokens={opts.max_tokens} '
                f'+ closing_tokens={closing_tokens} '
                f'> max_model_len={opts.max_model_len} even without audio history; '
                'incoming audio cannot fit; no history was removed')
        window_samples = round(opts.history_window_seconds * SAMPLE_RATE)
        if opts.context_policy == 'rolling' and len(audio) > window_samples:
            raise ContextLimit('incoming audio exceeds history_window_seconds; '
                               'no history was removed')
        historical_cost = (len(self.history) - len(self._system_tokens)
                           + self.committed_audio_extra_tokens)
        reasons = []
        if len(self._turns) + 1 > opts.max_audio_turns:
            reasons.append('turn_limit')
        if bare_required + historical_cost > opts.max_model_len:
            reasons.append('token_limit')
        if (opts.context_policy == 'rolling'
                and self._retained_audio_samples + len(audio) > window_samples):
            reasons.append('audio_window')
        if not reasons:
            return _WindowPlan(0, self.history, self.audios,
                               self.committed_audio_extra_tokens)
        if opts.context_policy == 'stop':
            raise ContextLimit(
                'context stop policy reached ' + ', '.join(reasons)
                + '; no history was removed; start a new session')

        ratio = opts.history_retain_seconds / opts.history_window_seconds
        # Targets include the new input. The indivisible incoming turn always
        # fits even if it alone exceeds the preferred (not hard) retain target.
        sample_target = max(len(audio), round(opts.history_retain_seconds * SAMPLE_RATE))
        turn_target = max(1, math.floor(opts.max_audio_turns * ratio))
        old_token_target = math.floor((opts.max_model_len - bare_required) * ratio)
        retained_samples = self._retained_audio_samples
        retained_cost = historical_cost
        remove_count = 0
        while remove_count < len(self._turns) and (
                retained_samples + len(audio) > sample_target
                or len(self._turns) - remove_count + 1 > turn_target
                or retained_cost > old_token_target):
            turn = self._turns[remove_count]
            retained_samples -= len(turn.audio)
            retained_cost -= len(self._turn_tokens(turn)) + turn.audio_extra_tokens
            remove_count += 1

        kept = self._turns[remove_count:]
        history = list(self._system_tokens)
        for turn in kept:
            history.extend(self._turn_tokens(turn))
        return _WindowPlan(remove_count, history, [turn.audio for turn in kept],
                           sum(turn.audio_extra_tokens for turn in kept), tuple(reasons))

    def _check_audio(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('audio must be non-empty finite 16-kHz mono PCM')
        if len(audio) > round(self.engine.options.max_chunk_seconds * SAMPLE_RATE):
            raise ValueError('audio exceeds configured max_chunk_seconds')
        # Keep a private immutable snapshot; producer buffers may be reused.
        snapshot = audio.copy()
        snapshot.setflags(write=False)
        return snapshot

    def generate(self, audio):
        result = self.engine.generate_batch([(self, audio)])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _prepare_generation(self, audio):
        """Build a private prospective request; do not mutate session state."""
        if self._closed or self.engine._closed:
            raise RuntimeError('model session is closed')
        if self._pending is not None:
            raise RuntimeError('previous generation must be committed or session closed')
        opts, engine = self.engine.options, self.engine
        audio = self._check_audio(audio)
        extra = (max(0, math.ceil(len(audio) / SAMPLE_RATE * 13) - 1)
                 if engine.backend == 'mock' else engine.audio_expansion(audio))
        plan = self._prepare_window(audio, extra)
        if engine.backend == 'mock':
            return audio, extra, plan, None, None
        else:
            ids = plan.history + engine.audio_turn + engine.asst_head
            expected_tokens = len(ids) + plan.audio_extra_tokens + extra
            if expected_tokens + opts.max_tokens > opts.max_model_len:
                raise ContextLimit(f'expanded prompt={expected_tokens} + max_tokens={opts.max_tokens} '
                                   f'> max_model_len={opts.max_model_len}; session stopped')
            request = engine.TokensPrompt(
                prompt_token_ids=ids, multi_modal_data={'audio': plan.audios + [audio]})
            return audio, extra, plan, request, expected_tokens

    def _finish_generation(self, prepared, output=None):
        """Decode one output into this session's pending slot, never commit it."""
        if self._closed or self.engine._closed:
            raise RuntimeError('model session is closed')
        if self._pending is not None:
            raise RuntimeError('previous generation must be committed or session closed')
        audio, extra, plan, _, expected_tokens = prepared
        engine = self.engine
        if engine.backend == 'mock':
            result = self._mock_generate(audio, plan, extra)
        else:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            while token_ids and token_ids[-1] in engine.stop_ids:
                token_ids.pop()
            reply = engine.tok.decode(token_ids, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
            actual_ids = getattr(output, 'prompt_token_ids', None)
            prompt_tokens = len(actual_ids) if actual_ids is not None else expected_tokens
            if prompt_tokens != expected_tokens:
                raise RuntimeError(f'context expansion mismatch: counted={expected_tokens}, '
                                   f'engine={prompt_tokens}; reply NOT committed')
            result = Generation(reply, token_ids, prompt_tokens, completion.finish_reason, extra)
        self._pending = (result, audio, plan)
        return result

    def _mock_generate(self, audio, plan, extra):
        """Deterministic protocol exercise, deliberately not speech recognition."""
        turn = self.total_committed_turns
        stage = turn % 3
        if stage == 1:
            reply = '<|WAIT|>'
        elif self.target_lang == 'zh':
            reply = ('<|FLEX|>【演示】收到音频，' if stage == 0
                     else '<|DIRECT|>这是链路测试，并非实际翻译。')
        else:
            reply = ('<|FLEX|>[DEMO] Audio received; ' if stage == 0
                     else '<|DIRECT|> this is a transport test, not a translation. ')
        # Mock token counts are synthetic and must not be used for GPU sizing.
        prompt_tokens = 200 + len(plan.history) + plan.audio_extra_tokens + extra
        opts = self.engine.options
        if prompt_tokens + opts.max_tokens > opts.max_model_len:
            raise ContextLimit('mock context budget reached; start a new session')
        time.sleep(0.03)
        return Generation(reply, list(reply.encode('utf-8')), prompt_tokens, 'stop', extra)

    def commit(self, audio, generation):
        """Commit only this session's valid, complete pending generation once."""
        if self._closed:
            raise RuntimeError('model session is closed')
        if self._pending is None or self._pending[0] is not generation:
            raise RuntimeError('generation does not belong to this pending session turn')
        if generation.finish_reason != 'stop':
            raise ProtocolError(f'finish_reason={generation.finish_reason!r}; reply NOT committed')
        parse_reply(generation.reply)
        snapshot = self._pending[1]
        supplied = np.asarray(audio, dtype=np.float32)
        if not np.array_equal(snapshot, supplied):
            raise ValueError('commit audio differs from generated audio')
        plan = self._pending[2]
        # Window changes and the new reply are one transaction. Backend errors,
        # truncation, invalid protocol, or mismatched audio leave old history
        # intact; no rollback can change a translation already shown to users.
        if plan.remove_count:
            dropped = self._turns[:plan.remove_count]
            dropped_samples = sum(len(turn.audio) for turn in dropped)
            self._turns = self._turns[plan.remove_count:]
            self.history = list(plan.history)
            self.audios = list(plan.audios)
            self.committed_audio_extra_tokens = plan.audio_extra_tokens
            self._retained_audio_samples -= dropped_samples
            self._dropped_turns += plan.remove_count
            self._dropped_audio_samples += dropped_samples
            self._rollovers += 1
            self._last_rollover = {
                'at_turn': self.total_committed_turns + 1,
                'reasons': list(plan.reasons),
                'dropped_turns': plan.remove_count,
                'dropped_audio_seconds': dropped_samples / SAMPLE_RATE,
            }
            LOG.info('Context rollover=%s before turn=%s reasons=%s dropped_turns=%s '
                     'dropped_audio_seconds=%.3f retained_old_turns=%s',
                     self._rollovers, self.total_committed_turns + 1, plan.reasons,
                     plan.remove_count, dropped_samples / SAMPLE_RATE, len(self._turns))
        # Keep actual generated token IDs, not decode/strip/re-encoded text.
        turn = CommittedTurn(snapshot, tuple(generation.token_ids), generation.audio_extra_tokens)
        self.history.extend(self._turn_tokens(turn))
        self.audios.append(snapshot)
        self._turns.append(turn)
        self.committed_audio_extra_tokens += generation.audio_extra_tokens
        self.total_committed_turns += 1
        self._total_audio_samples += len(snapshot)
        self._retained_audio_samples += len(snapshot)
        self._pending = None

    def close(self):
        """Release only this session's history; retain shared model GPU weights."""
        self._closed = True
        self._pending = None
        self.audios.clear()
        self._turns.clear()
        self.history.clear()
        self._system_tokens = ()
        self.committed_audio_extra_tokens = 0
        self._retained_audio_samples = 0
