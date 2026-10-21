"""Virtual audio chunks: JSON stores ranges, never copies the user's WAVs.

Only the selected source samples are read and resampled. In particular, do NOT
encode/resample a full utterance and then slice contextual features: that can
give an earlier turn access to future speech. Encoded chunk bytes are cached
per process; no CUDA tensors, credentials or reference text enter this cache.
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


def encode_chunk_ref(task, index):
    chunk = task["chunks"][index]
    payload = {
        "path": task["audio_path"],
        "start": chunk["start_seconds"],
        "end": chunk["end_seconds"],
        "input_seconds": chunk["input_seconds"],
    }
    for name in ("audio_size_bytes", "audio_mtime_ns"):
        if name in task:
            payload[name] = task[name]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PREFIX + base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_chunk_ref(value):
    if not isinstance(value, str) or not value.startswith(PREFIX):
        raise ValueError("Expected a flexaudio:v1 virtual chunk, not a full audio path")
    try:
        raw = base64.b64decode(value[len(PREFIX):], altchars=b"-_", validate=True)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed virtual audio chunk") from exc
    if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"]:
        raise ValueError("Virtual chunk requires a local audio path")
    for key in ("start", "end", "input_seconds"):
        number = data.get(key)
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            raise ValueError(f"Invalid virtual audio {key}")
    if data["start"] < 0 or data["end"] <= data["start"]:
        raise ValueError("Invalid virtual audio time range")
    if not 0 < data["input_seconds"] <= 8 or data["input_seconds"] + 1e-8 < data["end"] - data["start"]:
        raise ValueError("Audio chunk must fit its input duration (maximum 8 seconds)")
    return data


def chunk_wav_bytes(ref):
    """Return a 16 kHz mono float WAV containing only this chunk plus silence.

    stat is checked even on cache hits, so edits to the underlying WAV invalidate
    the cache and (when manifest metadata exists) fail rather than mix episodes.
    """
    data = decode_chunk_ref(ref)
    path = Path(data["path"])
    if not path.is_absolute():
        raise ValueError("Audio path must be absolute on the training server")
    info = path.stat()
    for name, actual in (("audio_size_bytes", info.st_size), ("audio_mtime_ns", info.st_mtime_ns)):
        if name in data and data[name] != actual:
            raise ValueError(f"Audio changed after preparation ({name}): {path}")
    return _read_chunk(ref, info.st_size, info.st_mtime_ns)


@lru_cache(maxsize=256)
def _read_chunk(ref, size, mtime_ns):
    # Cache key includes inventory, but no full-waveform cache or physical files.
    import numpy as np
    import soundfile as sf

    data = decode_chunk_ref(ref)
    with sf.SoundFile(data["path"]) as stream:
        sr = stream.samplerate
        start = round(data["start"] * sr)
        end = round(data["end"] * sr)
        if start < 0 or end > len(stream) + 1 or end <= start:
            raise ValueError("Chunk bounds do not match the real WAV samples")
        end = min(end, len(stream))
        stream.seek(start)
        pcm = stream.read(end - start, dtype="float32", always_2d=True)
    if len(pcm) != end - start or not np.isfinite(pcm).all():
        raise ValueError("Truncated or non-finite WAV samples")
    mono = pcm.mean(axis=1, dtype=np.float32)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    actual_count = round((data["end"] - data["start"]) * SAMPLE_RATE)
    output_count = round(data["input_seconds"] * SAMPLE_RATE)
    if output_count < actual_count or output_count < 160:
        raise ValueError("Invalid padded chunk sample count")
    result = np.zeros(output_count, dtype=np.float32)
    count = min(actual_count, len(mono))
    result[:count] = mono[:count]
    out = io.BytesIO()
    sf.write(out, result, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return out.getvalue()


def materialize_chunk_refs(refs):
    """Used by the same template in rollout AND teacher-forced training."""
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("FLEX requires at least one virtual audio chunk")
    return [chunk_wav_bytes(ref) for ref in refs]
