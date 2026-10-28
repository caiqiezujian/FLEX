"""Prepared-data relocation contract; real tiny WAVs, no GPU/network services."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

from Scripts import relocate_prepared_data as relocation
from flex_gspo.audio import decode_chunk_ref, encode_chunk_ref
from flex_gspo.prepare import build_row
from flex_gspo.prompts import SYSTEM_PROMPT


def write_wav(path, sample_rate=16000, frames=4000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0\0" * frames)


class RelocatePreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.old_audio = self.root / "old_audio"
        self.new_audio = self.root / "copied_audio"
        self.old_audio.mkdir()
        self.new_audio.mkdir()
        self.source = self.root / "old_train.jsonl"
        self.output = self.root / "new_prepared"

    def row(self, key="example"):
        old = self.old_audio / (key + ".wav")
        write_wav(old)
        result = build_row({
            "id": key, "audio_path": str(old),
            "source_text": "Preserve this literal path: /data/old/file.wav",
            "reference_text": "保留文字中的路径 /data/old/file.wav。",
        }, chunk_seconds=1, max_duration_seconds=40, system_prompt=SYSTEM_PROMPT)
        target = self.new_audio / old.name
        shutil.copyfile(old, target)
        newer = old.stat().st_mtime_ns + 5_000_000_000
        os.utime(target, ns=(newer, newer))
        old.unlink()  # Old audio may be unavailable on the destination server.
        return result

    def write_rows(self, rows):
        self.source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def run_relocation(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return relocation.relocate_dataset(self.source, self.new_audio, self.output, workers=2, **kwargs)

    def test_rewrites_both_references_without_changing_supervision_or_waveform(self):
        original = self.row()
        self.write_rows([original])
        original_bytes = self.source.read_bytes()
        target = self.new_audio / "example.wav"
        target_bytes = target.read_bytes()
        target_mtime = target.stat().st_mtime_ns
        report = self.run_relocation()
        moved = json.loads((self.output / "train.jsonl").read_text(encoding="utf-8"))
        old_task, task = json.loads(original["flex_task"]), json.loads(moved["flex_task"])
        self.assertEqual(task["audio_path"], str(target))
        self.assertEqual(task["audio_mtime_ns"], target_mtime)
        self.assertEqual(task["original_audio"]["mtime_ns"], target_mtime)
        for key in old_task.keys() - {"audio_path", "audio_mtime_ns", "original_audio"}:
            self.assertEqual(task[key], old_task[key], key)
        self.assertEqual(moved["messages"], original["messages"])
        self.assertEqual(moved["audios"], [encode_chunk_ref(task, 0)])
        payload = decode_chunk_ref(moved["audios"][0])
        self.assertEqual(payload["path"], str(target))
        self.assertEqual((payload["start"], payload["end"], payload["input_seconds"]), (0., .25, 1.))
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(target.read_bytes(), target_bytes)
        self.assertEqual(target.stat().st_mtime_ns, target_mtime)
        self.assertEqual((report["tasks"], report["changed_paths"], report["refreshed_mtimes"]), (1, 1, 1))
        self.assertEqual(report["input_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256((self.output / "train.jsonl").read_bytes()).hexdigest())
        self.assertFalse((self.output / "train.jsonl.partial").exists())

    def test_parallel_processing_preserves_manifest_order(self):
        rows = [self.row(f"clip_{i}") for i in range(12)]
        self.write_rows(rows)
        actual = relocation.relocate_row

        def delayed(row, root):
            if json.loads(row["flex_task"])["id"] == "clip_0":
                time.sleep(.05)
            return actual(row, root)

        with patch.object(relocation, "relocate_row", side_effect=delayed):
            report = self.run_relocation(progress_seconds=.01)
        records = [json.loads(line) for line in (self.output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([json.loads(row["flex_task"])["id"] for row in records], [f"clip_{i}" for i in range(12)])
        self.assertEqual(report["tasks"], 12)

    def test_bad_input_never_publishes_partial_manifest(self):
        good = self.row()
        inconsistent = copy.deepcopy(good)
        inconsistent["audios"] = ["flexaudio:v1:corrupted"]
        for label, bad in (("raw", {"key": "example", "en": {"txt": "Hello"}}),
                           ("ref", inconsistent), ("malformed", None)):
            with self.subTest(case=label):
                self.output = self.root / label
                self.write_rows([good] if label == "malformed" else [good, bad])
                if label == "malformed":
                    with self.source.open("a", encoding="utf-8") as stream:
                        stream.write("not JSON\n")
                with self.assertRaises(ValueError):
                    self.run_relocation()
                self.assertFalse((self.output / "train.jsonl").exists())
                self.assertFalse((self.output / "train.jsonl.partial").exists())
                self.assertFalse(json.loads((self.output / "relocate_failed.json").read_text())["manifest_published"])

    def test_missing_truncated_and_different_audio_are_rejected(self):
        row = self.row()
        self.write_rows([row])
        wav = self.new_audio / "example.wav"
        data = wav.read_bytes()
        for label in ("missing", "truncated", "different_rate"):
            with self.subTest(case=label):
                self.output = self.root / label
                if label == "missing":
                    wav.unlink()
                elif label == "truncated":
                    wav.write_bytes(data[:-2])
                else:
                    write_wav(wav, sample_rate=8000)  # Same bytes, incompatible duration/rate.
                with self.assertRaises(ValueError):
                    self.run_relocation()
                self.assertFalse((self.output / "train.jsonl").exists())

    def test_existing_output_is_preserved(self):
        self.write_rows([self.row()])
        self.output.mkdir()
        marker = self.output / "train.jsonl"
        marker.write_text("preserve existing run\n")
        with self.assertRaises(FileExistsError):
            self.run_relocation()
        self.assertEqual(marker.read_text(), "preserve existing run\n")
        self.assertEqual([p.name for p in self.output.iterdir()], ["train.jsonl"])

    def test_console_failure_after_publish_reports_actual_manifest_state(self):
        self.write_rows([self.row()])

        def broken_console(*args, **kwargs):
            if args and str(args[0]).startswith("[relocate] SUCCESS:"):
                raise OSError("stdout disconnected after publication")

        with patch("builtins.print", side_effect=broken_console), self.assertRaisesRegex(OSError, "stdout disconnected"):
            self.run_relocation()
        self.assertTrue((self.output / "train.jsonl").is_file())
        self.assertTrue(json.loads((self.output / "relocate_failed.json").read_text())["manifest_published"])

    def test_unsafe_id_and_guessed_filename_are_rejected(self):
        original = self.row()
        for key, old_path in (("../example", "/old/example.wav"), ("example", "/old/unrelated.wav")):
            row = copy.deepcopy(original)
            task = json.loads(row["flex_task"])
            task.update(id=key, audio_path=old_path)
            row["flex_task"] = json.dumps(task)
            row["audios"] = [encode_chunk_ref(task, 0)]
            with self.subTest(key=key, old_path=old_path), self.assertRaises(ValueError):
                relocation.relocate_row(row, self.new_audio)


if __name__ == "__main__":
    unittest.main()
