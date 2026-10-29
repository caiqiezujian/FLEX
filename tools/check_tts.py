#!/usr/bin/env python3
"""Verify the configured voice/API without loading the interpretation model."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web.tts import TTSOptions, TTSService, TTSError


async def check(args):
    service = TTSService(TTSOptions(url=args.url, assets_dir=args.assets_dir,
                                    voices_config=args.voices_config, timeout_seconds=args.timeout))
    text = args.text or ('Hello, this is the interpretation voice preview.' if args.language == 'en'
                         else '你好，这是同声传译的音色试听。')
    try:
        await service.start()
        audio, rate, duration = await service.synthesize(text, args.language)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as out:
            out.write(audio)
        print(json.dumps(dict(ok=True, language=args.language, sample_rate=rate,
                              duration_seconds=duration, output=str(args.output.resolve())), ensure_ascii=False))
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='TTS base URL, e.g. http://7.242.105.6:9100')
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--voices-config', type=Path)
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    parser.add_argument('--text', default='')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', type=Path, required=True, help='New WAV file; existing files are never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists; select another WAV path')
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(check(args))
    except (TTSError, ValueError, OSError) as exc:
        print(f'TTS check failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
