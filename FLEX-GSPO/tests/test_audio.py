"""Real CPU waveform tests for causal virtual chunk materialization."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from flex_gspo.audio import (PREFIX, _read_chunk, chunk_wav_bytes, decode_chunk_ref,
                             encode_chunk_ref, materialize_chunk_refs)

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None


def task_for(path, duration=2.25, inventory=True):
    task = {
        "audio_path": str(path.resolve()),
        "chunks": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": duration, "input_seconds": 1.0},
        ],
    }
    if inventory:
        info = path.stat()
        task.update(audio_size_bytes=info.st_size, audio_mtime_ns=info.st_mtime_ns)
    return task


class ChunkReferenceTests(unittest.TestCase):
    def test_uri_contains_only_audio_range_not_transcript_or_reference(self):
        task = {"audio_path": "/audio/file.wav", "source_text": "secret", "reference_text": "参考",
                "chunks": [{"start_seconds": 0, "end_seconds": 1, "input_seconds": 1}]}
        uri = encode_chunk_ref(task, 0)
        decoded = decode_chunk_ref(uri)
        self.assertEqual(set(decoded), {"path", "start", "end", "input_seconds"})
        self.assertNotIn("secret", json.dumps(decoded))
        self.assertEqual(decoded["end"], 1)

    def test_bad_uri_is_rejected(self):
        for value in (None, [], "file.wav", PREFIX + "not_base64!", PREFIX):
            with self.assertRaises(ValueError):
                decode_chunk_ref(value)

    def test_invalid_ranges_are_rejected(self):
        task = {"audio_path": "/audio/file.wav", "chunks": [{"start_seconds": 0, "end_seconds": 1, "input_seconds": 1}]}
        for key, value in (("start_seconds", -1), ("end_seconds", 0), ("input_seconds", 0.5),
                           ("input_seconds", 9), ("end_seconds", True)):
            changed = {**task, "chunks": [{**task["chunks"][0], key: value}]}
            with self.assertRaises(ValueError):
                decode_chunk_ref(encode_chunk_ref(changed, 0))
        with self.assertRaises(ValueError):
            materialize_chunk_refs([])


@unittest.skipUnless(np is not None and sf is not None, "Install numpy and soundfile for real waveform tests")
class AudioMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _read_chunk.cache_clear()

    def write(self, name, data, rate=16000):
        path = self.root / name
        sf.write(path, data, rate, subtype="FLOAT")
        return path

    def decode(self, data):
        pcm, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        self.assertEqual(sr, 16000)
        self.assertEqual(pcm.shape[1], 1)
        return pcm[:, 0]

    def test_future_changes_cannot_change_first_chunk(self):
        prefix = np.linspace(-0.3, 0.3, 16000, dtype=np.float32)
        first = self.write("a.wav", np.concatenate((prefix, np.zeros(20000, dtype=np.float32))))
        second = self.write("b.wav", np.concatenate((prefix, np.ones(20000, dtype=np.float32) * 0.9)))
        output_a = chunk_wav_bytes(encode_chunk_ref(task_for(first), 0))
        output_b = chunk_wav_bytes(encode_chunk_ref(task_for(second), 0))
        self.assertEqual(output_a, output_b)
        np.testing.assert_array_equal(self.decode(output_a), prefix)

    def test_stereo_is_mixed_and_last_chunk_padded_without_future_audio(self):
        stereo = np.column_stack((np.full(36000, 0.2, dtype=np.float32), np.full(36000, 0.6, dtype=np.float32)))
        path = self.write("stereo.wav", stereo)
        output = self.decode(chunk_wav_bytes(encode_chunk_ref(task_for(path), 2)))
        self.assertEqual(len(output), 16000)
        np.testing.assert_allclose(output[:4000], 0.4, atol=1e-7)
        np.testing.assert_array_equal(output[4000:], 0.0)

    def test_resampling_reads_only_the_current_interval(self):
        try:
            import scipy.signal  # noqa: F401
        except ImportError:
            self.skipTest("Install scipy for resampling tests")
        sr = 48000
        prefix = np.sin(np.arange(sr, dtype=np.float32) * (2 * np.pi * 440 / sr)) * 0.25
        first = self.write("a48.wav", np.concatenate((prefix, np.zeros(sr, dtype=np.float32))), sr)
        second = self.write("b48.wav", np.concatenate((prefix, np.ones(sr, dtype=np.float32))), sr)
        data_a = chunk_wav_bytes(encode_chunk_ref(task_for(first, 2), 0))
        data_b = chunk_wav_bytes(encode_chunk_ref(task_for(second, 2), 0))
        self.assertEqual(data_a, data_b)
        self.assertEqual(len(self.decode(data_a)), 16000)

    def test_cache_hits_still_check_source_inventory(self):
        path = self.write("audio.wav", np.zeros(36000, dtype=np.float32))
        ref = encode_chunk_ref(task_for(path), 0)
        chunk_wav_bytes(ref)
        before = path.stat()
        sf.write(path, np.ones(36000, dtype=np.float32) * 0.25, 16000, subtype="FLOAT")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "changed after preparation"):
            chunk_wav_bytes(ref)

    def test_cache_without_inventory_uses_updated_file_metadata(self):
        path = self.write("audio.wav", np.zeros(36000, dtype=np.float32))
        ref = encode_chunk_ref(task_for(path, inventory=False), 0)
        first = chunk_wav_bytes(ref)
        before = path.stat()
        sf.write(path, np.full(36000, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        second = chunk_wav_bytes(ref)
        self.assertNotEqual(first, second)
        np.testing.assert_allclose(self.decode(second), 0.25)

    def test_nonfinite_samples_and_header_corruption_are_not_silently_repaired(self):
        values = np.zeros(36000, dtype=np.float32)
        values[100] = float("nan")
        path = self.write("nan.wav", values)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            chunk_wav_bytes(encode_chunk_ref(task_for(path), 0))
        broken = self.root / "broken.wav"
        broken.write_bytes(b"not a wave file")
        with self.assertRaises((RuntimeError, ValueError, OSError)):
            chunk_wav_bytes(encode_chunk_ref(task_for(broken), 0))

    def test_missing_and_out_of_range_audio_fail(self):
        path = self.write("short.wav", np.zeros(16000, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "bounds"):
            chunk_wav_bytes(encode_chunk_ref(task_for(path), 2))
        path.unlink()
        with self.assertRaises(OSError):
            chunk_wav_bytes(encode_chunk_ref(task_for(path, inventory=False), 0))

    def test_materialize_returns_one_payload_for_each_requested_chunk(self):
        path = self.write("audio.wav", np.zeros(36000, dtype=np.float32))
        task = task_for(path)
        refs = [encode_chunk_ref(task, i) for i in range(2)]
        outputs = materialize_chunk_refs(refs)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(isinstance(value, bytes) for value in outputs))
        self.assertTrue(all(len(self.decode(value)) == 16000 for value in outputs))


if __name__ == "__main__":
    unittest.main()
