"""Run real checkpoint audio turns without starting the Web server."""
import argparse
import wave

import numpy as np

from .model import ModelEngine, ModelOptions, SAMPLE_RATE, parse_reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Local merged checkpoint')
    parser.add_argument('audio', help='Mono 16 kHz, 16-bit PCM WAV')
    parser.add_argument('--target', choices=['zh', 'en'], default='zh')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--cpu-offload-gb', type=float, default=0.0,
                        help='Experimental CPU weight offload in GiB (default 0)')
    parser.add_argument('--audio-token-cache-size', type=int, default=0,
                        help='Experimental exact-count cache size; default 0 disables it')
    parser.add_argument('--mm-processor-cache-gb', type=float, default=0.0)
    parser.add_argument('--enable-prefix-caching', action='store_true')
    parser.add_argument('--enable-graphs', action='store_true')
    parser.add_argument('--turns', type=int, default=3, help='Number of one-second turns (default 3)')
    args = parser.parse_args()
    if args.turns < 1:
        parser.error('--turns must be positive')
    with wave.open(args.audio, 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, 'NONE'):
            parser.error('audio must be a mono 16 kHz 16-bit PCM WAV')
        if wav.getnframes() < SAMPLE_RATE * args.turns:
            parser.error('audio must contain at least --turns seconds')
        chunks = [np.frombuffer(wav.readframes(SAMPLE_RATE), dtype='<i2').astype(np.float32) / 32768.0
                  for _ in range(args.turns)]
    options = ModelOptions(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                           cpu_offload_gb=args.cpu_offload_gb,
                           mm_processor_cache_gb=args.mm_processor_cache_gb,
                           audio_token_cache_size=args.audio_token_cache_size,
                           prefix_caching=args.enable_prefix_caching,
                           enforce_eager=not args.enable_graphs, history_window_seconds=20,
                           history_retain_seconds=10, max_audio_turns=32, image_slots=1)
    engine = ModelEngine(options)
    session = engine.new_session(args.target)
    try:
        for index, audio in enumerate(chunks, 1):
            generation = session.generate(audio)
            tag, delta = parse_reply(generation.reply)
            session.commit(audio, generation)
            timings = generation.timings
            print(f'{index}: {tag}{delta}  [engine={timings["engine_generate_s"]:.3f}s '
                  f'count={timings.get("audio_count_s", 0):.3f}s '
                  f'count_mode={timings.get("audio_count_mode")} '
                  f'count_hit={timings.get("audio_count_cache_hit")}]', flush=True)
    finally:
        session.close()
        engine.close()


if __name__ == '__main__':
    main()
