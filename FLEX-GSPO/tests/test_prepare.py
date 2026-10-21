"""Dependency-free preparation tests; no checkpoints, GPUs, or network calls."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from flex_gspo.prepare import PreparationError, PrepareOptions, normalize_record, plan_chunks, prepare_dataset, probe_wav
from flex_gspo.prompts import SYSTEM_PROMPT


def write_pcm(path: Path, seconds=2.25, sample_rate=16000, channels=1):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\0\0" * int(seconds * sample_rate) * channels)


def raw_row(key="example", source="The meeting starts at nine.", target="会议九点开始。"):
    return {"lang": "en", "key": key, "en": {"txt": source}, "zh": {"txt": target}}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "wav"
        self.audio.mkdir()
        self.source = self.root / "metadata.jsonl"
        self.output = self.root / "prepared"

    def prepare(self, rows, **kwargs):
        self.source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        return prepare_dataset(PrepareOptions(str(self.source), str(self.audio), str(self.output), workers=2, **kwargs))

    def read_manifest(self):
        return [json.loads(line) for line in (self.output / "train.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_header_probe_supports_multichannel_and_nonmodel_rate(self):
        path = self.audio / "example.wav"
        write_pcm(path, 1.25, 48000, 2)
        result = probe_wav(path)
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(result["channels"], 2)
        self.assertEqual(result["duration_seconds"], 1.25)
        self.assertEqual(result["num_frames"], 60000)

    def test_wave_payload_truncation_is_rejected(self):
        path = self.audio / "example.wav"
        write_pcm(path)
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 100)
        with self.assertRaisesRegex(PreparationError, "Truncated"):
            probe_wav(path)

    def test_float_and_rf64_headers(self):
        for rf64 in (False, True):
            with self.subTest(rf64=rf64):
                samples = struct.pack("<ff", 0.0, 0.5)
                fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
                data = b"data" + struct.pack("<I", 0xFFFFFFFF if rf64 else len(samples)) + samples
                body = b"WAVE" + fmt + data
                if rf64:
                    body = b"WAVE" + b"ds64" + struct.pack("<IQQQI", 28, len(body) + 36, len(samples), 2, 0) + fmt + data
                path = self.audio / "float.wav"
                path.write_bytes((b"RF64" if rf64 else b"RIFF") + struct.pack("<I", 0xFFFFFFFF if rf64 else len(body)) + body)
                info = probe_wav(path)
                self.assertEqual(info["encoding"], "IEEE_FLOAT")
                self.assertEqual(info["num_frames"], 2)

    def test_chunks_cover_real_time_without_counting_padding(self):
        chunks = plan_chunks(2.25, 1)
        self.assertEqual(chunks, [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.25, "input_seconds": 1.0},
        ])
        self.assertEqual(len(plan_chunks(2, 1)), 2)
        self.assertEqual(plan_chunks(0.01, 1)[0]["input_seconds"], 1.0)
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(PreparationError):
                plan_chunks(value)
        for value in (0, 9, 0.5, True):
            with self.assertRaises(PreparationError):
                plan_chunks(1.0, value)

    def test_manifest_has_only_first_audio_and_no_reference_text_in_messages(self):
        path = self.audio / "example.wav"
        write_pcm(path)
        before = path.read_bytes()
        report = self.prepare([raw_row()])
        self.assertTrue(report["manifest_published"])
        row = self.read_manifest()[0]
        self.assertEqual(row["messages"], [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<audio>"},
        ])
        self.assertEqual(len(row["audios"]), 1)
        self.assertTrue(row["audios"][0].startswith("flexaudio:v1:"))
        self.assertIsInstance(row["flex_task"], str)
        task = json.loads(row["flex_task"])
        self.assertEqual(task["id"], "example")
        self.assertEqual(task["audio_path"], str(path.resolve()))
        self.assertEqual(task["source_text"], raw_row()["en"]["txt"])
        self.assertEqual(task["reference_text"], raw_row()["zh"]["txt"])
        self.assertFalse(task["reference_provenance"]["is_gold"])
        self.assertEqual(task["audio_size_bytes"], path.stat().st_size)
        self.assertEqual(task["audio_mtime_ns"], path.stat().st_mtime_ns)
        self.assertEqual(task["sample_rate"], 16000)
        self.assertIsNone(task["source_segments"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.audio.iterdir()), [path])
        self.assertFalse((self.output / ".prepare.lock").exists())
        self.assertFalse(any(path.name.startswith(".prepare-") for path in self.output.iterdir()))

    def test_duplicate_keys_keep_first_and_report_conflicts(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row(), raw_row(), raw_row(target="不同的参考。")])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 2)
        self.assertEqual(report["conflicting_duplicate_rows"], 1)
        task = json.loads(self.read_manifest()[0]["flex_task"])
        self.assertEqual(task["reference_text"], raw_row()["zh"]["txt"])
        duplicate_rows = [json.loads(line) for line in (self.output / "duplicates.jsonl").read_text().splitlines()]
        self.assertEqual(duplicate_rows[0]["first_line"], 1)
        self.assertTrue(duplicate_rows[1]["conflicting_text"])

    def test_invalid_first_record_does_not_hide_later_valid_same_key(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row(source=""), raw_row()], skip_invalid=True)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(json.loads(self.read_manifest()[0]["flex_task"])["source_text"], raw_row()["en"]["txt"])

    def test_malformed_later_duplicate_is_reported_not_silently_ignored(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row(), raw_row(target="")])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertFalse(report["manifest_published"])

    def test_missing_audio_blocks_manifest_in_strict_mode(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row(), raw_row("missing")])
        self.assertFalse(report["manifest_published"])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertFalse((self.output / "train.jsonl").exists())
        rejection = json.loads((self.output / "rejected.jsonl").read_text())
        self.assertEqual(rejection["key"], "missing")

    def test_skipping_bad_rows_requires_explicit_flag(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row("missing"), raw_row()], skip_invalid=True)
        self.assertTrue(report["manifest_published"])
        self.assertEqual(len(self.read_manifest()), 1)

    def test_output_is_not_overwritten(self):
        write_pcm(self.audio / "example.wav")
        self.prepare([raw_row()])
        before = (self.output / "train.jsonl").read_bytes()
        with self.assertRaisesRegex(PreparationError, "overwrite"):
            self.prepare([raw_row()])
        self.assertEqual((self.output / "train.jsonl").read_bytes(), before)

    def test_concurrent_or_stale_lock_is_not_removed(self):
        self.output.mkdir()
        lock = self.output / ".prepare.lock"
        lock.write_text("another process")
        with self.assertRaisesRegex(PreparationError, "lock"):
            self.prepare([raw_row()])
        self.assertEqual(lock.read_text(), "another process")

    def test_manifest_order_is_stable_with_workers(self):
        for key in ("b", "a", "c"):
            write_pcm(self.audio / f"{key}.wav")
        self.prepare([raw_row(key) for key in ("b", "a", "c")])
        self.assertEqual([json.loads(row["flex_task"])["id"] for row in self.read_manifest()], ["b", "a", "c"])

    def test_limit_counts_distinct_eligible_keys(self):
        write_pcm(self.audio / "example.wav")
        report = self.prepare([raw_row(), raw_row("missing")], limit=1)
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rows_read"], 1)
        self.assertEqual(report["rejected_rows"], 0)

    def test_max_duration_rejects_instead_of_trimming(self):
        path = self.audio / "example.wav"
        write_pcm(path, 3)
        before = path.stat().st_size
        report = self.prepare([raw_row()], max_duration_seconds=2)
        self.assertFalse(report["manifest_published"])
        self.assertEqual(path.stat().st_size, before)

    def test_bom_blank_lines_and_invalid_json_report(self):
        write_pcm(self.audio / "example.wav")
        self.source.write_text("\ufeff" + json.dumps(raw_row()) + "\n\nnot json\n", encoding="utf-8")
        report = prepare_dataset(PrepareOptions(str(self.source), str(self.audio), str(self.output), skip_invalid=True))
        self.assertEqual(report["blank_lines"], 1)
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)

    def test_invalid_keys_and_text_cannot_leak_or_escape(self):
        for key in ("../evil", "..\\evil", "a/b", "", " a", "\x00"):
            with self.assertRaises(PreparationError):
                normalize_record(raw_row(key), self.audio.resolve())
        for row in ([], {**raw_row(), "lang": "zh"}, raw_row(source=" "), raw_row(target="")):
            with self.assertRaises(PreparationError):
                normalize_record(row, self.audio.resolve())

    def test_custom_system_prompt_is_rejected_before_preparation(self):
        write_pcm(self.audio / "example.wav")
        custom = self.root / "prompt.txt"
        custom.write_text("My exact SFT prompt.", encoding="utf-8")
        with self.assertRaisesRegex(PreparationError, "exactly match"):
            self.prepare([raw_row()], system_prompt_file=str(custom))
        self.assertFalse((self.output / "train.jsonl").exists())

    def test_matching_sft_prompt_file_is_accepted(self):
        write_pcm(self.audio / "example.wav")
        custom = self.root / "prompt.txt"
        custom.write_text(SYSTEM_PROMPT, encoding="utf-8")
        self.prepare([raw_row()], system_prompt_file=str(custom))
        self.assertEqual(self.read_manifest()[0]["messages"][0]["content"], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
