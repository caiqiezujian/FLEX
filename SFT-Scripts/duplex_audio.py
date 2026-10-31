"""Read virtual SFT chunks without creating WAV files on disk.

The reference format is shared with FLEX GSPO, but this module has no RL or
SWIFT dependency. Crop source samples BEFORE resampling or feature extraction;
processing the whole recording first can leak future audio across a boundary.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import io
import json
import math
from pathlib import Path

PREFIX = "flexaudio:v1:"
SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 8  # Project recipe limit, not a universal model limit.


def encode_chunk_ref(audio_path, start_seconds, end_seconds, input_seconds):
    payload = {"path": str(audio_path), "start": start_seconds,
               "end": end_seconds, "input_seconds": input_seconds}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    ref = PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")
    decode_chunk_ref(ref)
    return ref


def decode_chunk_ref(value):
    if not isinstance(value, str) or not value.startswith(PREFIX):
        raise ValueError("Expected flexaudio:v1: virtual audio reference")
    try:
        raw = base64.b64decode(value[len(PREFIX):], altchars=b"-_", validate=True)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed virtual audio reference") from exc
    if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"]:
        raise ValueError("Virtual audio requires a local path")
    for name in ("start", "end", "input_seconds"):
        number = data.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"Invalid virtual audio {name}")
    span = data["end"] - data["start"]
    if data["start"] < 0 or span <= 0:
        raise ValueError("Invalid virtual audio time range")
    if not 0 < data["input_seconds"] <= MAX_CHUNK_SECONDS or data["input_seconds"] + 1e-8 < span:
        raise ValueError("Chunk must fit its input duration (project maximum 8 seconds)")
    return data


def chunk_wav_bytes(ref):
    """16 kHz mono FLOAT WAV in memory; never a temporary audio file.

    File inventory is checked on every call, including cache hits. The cache
    contains at most 64 small chunks per worker, not complete source recordings.
    """
    data = decode_chunk_ref(ref)
    path = Path(data["path"])
    if not path.is_absolute():
        raise ValueError("Audio path must be absolute on the training server")
    info = path.stat()
    for name, actual in (("audio_size_bytes", info.st_size), ("audio_mtime_ns", info.st_mtime_ns)):
        if name in data and data[name] != actual:
            raise ValueError(f"Source audio changed after preparation ({name}): {path}")
    return _read_chunk(ref, info.st_size, info.st_mtime_ns)


@lru_cache(maxsize=64)
def _read_chunk(ref, size, mtime_ns):
    import numpy as np
    import soundfile as sf

    data = decode_chunk_ref(ref)
    with sf.SoundFile(data["path"]) as stream:
        sr = stream.samplerate
        start = round(data["start"] * sr)
        end = round(data["end"] * sr)
        if data["end"] > len(stream) / sr + 1e-8 or start < 0 or end > len(stream) or end <= start:
            raise ValueError("Chunk bounds exceed the actual source audio")
        stream.seek(start)
        pcm = stream.read(end - start, dtype="float32", always_2d=True)
    if len(pcm) != end - start or not np.isfinite(pcm).all():
        raise ValueError("Truncated or non-finite audio samples")
    mono = pcm.mean(axis=1, dtype=np.float32)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    actual_count = round((data["end"] - data["start"]) * SAMPLE_RATE)
    output_count = round(data["input_seconds"] * SAMPLE_RATE)
    if output_count < actual_count or output_count < 160:
        raise ValueError("Invalid padded chunk length")
    result = np.zeros(output_count, dtype=np.float32)
    count = min(actual_count, len(mono))
    result[:count] = mono[:count]
    out = io.BytesIO()
    sf.write(out, result, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return out.getvalue()


def materialize_chunk_refs(refs):
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("A virtual session needs at least one audio chunk")
    return [chunk_wav_bytes(ref) for ref in refs]
