"""Compare multi-turn count cache on/off, using a simulated vLLM engine."""
from types import SimpleNamespace

import numpy as np

from flex_web_d910b import model
from flex_web_d910b.test_audio_tokens_028 import native028  # pytest fixture


def run_turns(native, cache_size):
    preprocessor, _, _ = native
    counts = []
    pad = 777

    def expanded(request):
        audio_iter = iter(request['multi_modal_data']['audio'])
        ids = []
        for token in request['prompt_token_ids']:
            if token == pad:
                audio = next(audio_iter)
                ids.extend([pad] * (1 + len(audio) // 16000))
            else:
                ids.append(token)
        return ids

    def preprocess(request):
        counts.append(len(request['multi_modal_data']['audio'][0]))
        return {'prompt_token_ids': expanded(request)}

    preprocessor.preprocess = preprocess
    engine = object.__new__(model.ModelEngine)
    engine.backend = 'vllm'
    engine.options = model.ModelOptions(max_batch_size=1, max_chunk_seconds=3,
                                        history_window_seconds=4, history_retain_seconds=2,
                                        audio_token_cache_size=cache_size)
    engine._closed = False
    engine.enc = lambda text: [10] * len(text)
    engine.tail, engine.audio_turn, engine.asst_head = [11], [12, pad, 11], [13]
    engine.stop_ids = []
    engine.tok = SimpleNamespace(convert_tokens_to_ids=lambda text: pad,
                                 decode=lambda *args, **kwargs: '<|WAIT|>')
    engine.TokensPrompt = dict
    engine.counter = preprocessor
    engine._audio_token_counter = model.AudioTokenCounter.for_preprocessor(
        preprocessor, vllm_version='0.28.0', max_entries=cache_size)
    engine._audio_count_cache_hit = False
    engine._audio_count_mode = engine._audio_token_counter.mode
    engine.metadata = {}
    engine.sampling = object()
    requests = []

    def generate(batch, sampling, use_tqdm):
        assert sampling is engine.sampling and use_tqdm is False
        requests.extend(batch)
        return [SimpleNamespace(prompt_token_ids=expanded(request), outputs=[
            SimpleNamespace(token_ids=[901], finish_reason='stop')]) for request in batch]

    engine.llm = SimpleNamespace(generate=generate, llm_engine=SimpleNamespace(
        engine_core=SimpleNamespace(shutdown=lambda: None)))
    session = engine.new_session('zh')
    trace = []
    try:
        for size, amplitude in [(16000, 0), (16000, .5), (32000, .25),
                                (48000, -.5), (16000, .75)]:
            audio = np.full(size, amplitude, dtype=np.float32)
            result = session.generate(audio)
            model.parse_reply(result.reply)
            session.commit(audio, result)
            trace.append((result.reply, result.prompt_tokens,
                          session.context_metadata['retained_audio_seconds'],
                          session.context_metadata['rollovers']))
        assert any(turn[-1] for turn in trace), 'test must cross a window boundary'
        assert all(len(request['multi_modal_uuids']['audio']) ==
                   len(request['multi_modal_data']['audio']) for request in requests)
        return trace, counts
    finally:
        session.close()
        engine.close()


def test_028_cache_preserves_multiturn_outputs_and_window(native028):
    without_cache, measurements_off = run_turns(native028, 0)
    with_cache, measurements_on = run_turns(native028, 64)
    assert with_cache == without_cache
    assert measurements_off == [16000, 16000, 32000, 48000, 16000]
    assert measurements_on == [16000, 32000, 48000]
