"""Prepare segmented source records using local WAV fixtures and no model calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from flex_gspo.audio import decode_chunk_ref
from flex_gspo.prepare import PreparationError, PrepareOptions, normalize_record, prepare_dataset
from flex_gspo.prompts import SYSTEM_PROMPT
from test_prepare import raw_row, write_pcm


def segmented_row(audio, **extra):
    return {
        "audio": str(audio),
        "srcs": ["  The train leaves  ", " at nine. "],
        "chooses": ["  火车  ", " 九点出发。 "],
        **extra,
    }


class SegmentedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.source = self.input_dir / "segments.jsonl"
        self.output = self.root / "prepared"

    def wav(self, relative="audio/example.wav", seconds=2.25):
        path = self.input_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_pcm(path, seconds=seconds)
        return path.resolve()

    def prepare(self, rows, **kwargs):
        self.source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        options = dict(input_jsonl=str(self.source), output_dir=str(self.output), workers=2)
        options.update(kwargs)
        return prepare_dataset(PrepareOptions(**options))

    def read_output(self, name="train.jsonl"):
        return [json.loads(line) for line in (self.output / name).read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def tasks(self):
        return [json.loads(row["flex_task"]) for row in self.read_output()]

    def test_default_auto_accepts_segments_without_audio_dir(self):
        path = self.wav()
        before = path.read_bytes()
        report = self.prepare([segmented_row(path)])
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 0)
        task = self.tasks()[0]
        self.assertEqual(task["id"], "example")
        self.assertEqual(task["audio_path"], str(path))
        self.assertEqual(task["source_text"], "The train leaves at nine.")
        self.assertEqual(task["reference_text"], "火车九点出发。")
        self.assertEqual((task["source_lang"], task["target_lang"]), ("en", "zh"))
        self.assertTrue(task["quality_ready"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_normalization_joins_stripped_segments_in_order(self):
        path = self.wav()
        row = segmented_row(path, srcs=["  First sentence.\n", " Second   sentence. ", "Third."],
                            chooses=[" 第一。 ", " 第二。\n", "第三。"])
        normalized = normalize_record(row, None, input_format="segments", input_dir=self.input_dir)
        self.assertEqual(normalized["source_text"], "First sentence. Second   sentence. Third.")
        self.assertEqual(normalized["reference_text"], "第一。第二。第三。")
        self.assertEqual(normalized["audio_path"], str(path))
        self.assertEqual(row["srcs"][0], "  First sentence.\n")
        self.assertEqual(row["chooses"][0], " 第一。 ")

    def test_auto_legacy_extra_audio_field_does_not_change_its_path_contract(self):
        path = self.wav("legacy/example.wav")
        row = raw_row("example")
        row["audio"] = "unused-legacy-metadata.wav"
        report = self.prepare([row], audio_dir=str(path.parent))
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["accepted_by_format"], {"legacy": 1, "segments": 0})
        self.assertEqual(self.tasks()[0]["audio_path"], str(path))
        # Presence of partial segment fields must still surface bad new input,
        # never silently choose legacy text after a segments parsing failure.
        with self.assertRaises(PreparationError):
            normalize_record({**row, "srcs": []}, path.parent, input_dir=self.input_dir)

    def test_optional_id_or_key_takes_precedence_over_filename_stem(self):
        path = self.wav("audio/filename.wav")
        for field in ("id", "key"):
            with self.subTest(field=field):
                normalized = normalize_record(segmented_row(path, **{field: "chosen-id"}), None,
                                              input_format="segments", input_dir=self.input_dir)
                self.assertEqual(normalized["id"], "chosen-id")
                self.assertEqual(normalized["audio_path"], str(path))

    def test_invalid_explicit_identifiers_are_rejected(self):
        path = self.wav()
        for field in ("id", "key"):
            for value in ("", " id", "../escape", "a/b", "a\\b", "\x00", 7):
                with self.subTest(field=field, value=value), self.assertRaises(PreparationError):
                    normalize_record(segmented_row(path, **{field: value}), None,
                                     input_format="segments", input_dir=self.input_dir)

    def test_relative_hash_path_resolves_against_input_jsonl_parent(self):
        path = self.wav("nested#directory/part#1.wav")
        report = self.prepare([segmented_row("nested#directory/part#1.wav")], input_format="segments")
        self.assertTrue(report["manifest_published"])
        row = self.read_output()[0]
        task = json.loads(row["flex_task"])
        self.assertEqual(task["id"], "part#1")
        self.assertEqual(task["audio_path"], str(path))
        self.assertEqual(decode_chunk_ref(row["audios"][0])["path"], str(path))

    def test_supplied_audio_dir_never_overrides_segments_audio(self):
        original = self.wav("explicit/example.wav", seconds=2.25)
        legacy_audio = self.root / "legacy_audio"
        legacy_audio.mkdir()
        write_pcm(legacy_audio / "example.wav", seconds=1.0)
        report = self.prepare([segmented_row("explicit/example.wav")],
                              audio_dir=str(legacy_audio), input_format="segments")
        self.assertTrue(report["manifest_published"])
        self.assertEqual(self.tasks()[0]["audio_path"], str(original))
        self.assertEqual(self.tasks()[0]["duration_seconds"], 2.25)

    def test_segments_and_voice_end_do_not_define_audio_chunks_or_actor_prompts(self):
        path = self.wav(seconds=2.25)
        rejected_text = "NEVER_INCLUDE_REJECTED_TRANSLATION_6719"
        row = segmented_row(path, srcs=["SOURCE_TRANSCRIPT_6719"],
                            chooses=["参考译文甲乙丙丁"], rejects=[rejected_text], voice_end=False)
        report = self.prepare([row], input_format="segments")
        self.assertEqual(report["accepted_rows"], 1)
        prepared = self.read_output()[0]
        task = json.loads(prepared["flex_task"])
        self.assertEqual(task["chunks"], [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.25, "input_seconds": 1.0},
        ])
        self.assertIsNone(task["source_segments"])
        self.assertEqual(prepared["messages"], [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<audio>"},
        ])
        self.assertEqual(len(prepared["audios"]), 1)
        first_chunk = decode_chunk_ref(prepared["audios"][0])
        self.assertEqual((first_chunk["start"], first_chunk["end"]), (0.0, 1.0))
        messages = json.dumps(prepared["messages"], ensure_ascii=False)
        self.assertNotIn(task["source_text"], messages)
        self.assertNotIn(task["reference_text"], messages)
        self.assertNotIn(rejected_text, json.dumps(prepared, ensure_ascii=False))
        self.assertNotIn("rejects", task)
        self.assertNotIn("voice_end", task)
        self.assertNotIn("trajectory", prepared)

    def test_missing_audio_file_blocks_publication_in_strict_mode(self):
        good = self.wav()
        report = self.prepare([segmented_row(good), segmented_row("missing.wav")],
                              input_format="segments")
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertFalse(report["manifest_published"])
        self.assertFalse((self.output / "train.jsonl").exists())
        self.assertEqual(self.read_output("rejected.jsonl")[0]["line"], 2)

    def test_missing_or_malformed_audio_field_is_rejected(self):
        path = self.wav()
        missing = segmented_row(path)
        del missing["audio"]
        rows = [missing] + [segmented_row(value, id=f"bad-{index}")
                            for index, value in enumerate(("", "  "))]
        rows.extend({**segmented_row(path), "audio": value} for value in (None, 123, [], {}))
        for row in rows:
            with self.subTest(audio=row.get("audio")), self.assertRaises(PreparationError):
                normalize_record(row, None, input_format="segments", input_dir=self.input_dir)

    def test_malformed_segment_lists_are_reported_while_valid_rows_can_be_published(self):
        path = self.wav()
        invalid = []
        for field in ("srcs", "chooses"):
            for value in (None, "text", {}, [], [""], ["  "], [17], [None], ["valid", []]):
                row = segmented_row(path, id=f"invalid-{len(invalid)}")
                row[field] = value
                invalid.append(row)
            row = segmented_row(path, id=f"invalid-{len(invalid)}")
            del row[field]
            invalid.append(row)
        report = self.prepare(invalid + [segmented_row(path, id="valid")],
                              input_format="segments", skip_invalid=True)
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["rejected_rows"], len(invalid))
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(self.tasks()[0]["id"], "valid")
        self.assertEqual(len(self.read_output("rejected.jsonl")), len(invalid))

    def test_mismatched_segment_counts_are_rejected_without_truncation(self):
        path = self.wav()
        report = self.prepare([
            segmented_row(path, id="too-many-srcs", srcs=["one", "two"], chooses=["一"]),
            segmented_row(path, id="too-many-chooses", srcs=["one"], chooses=["一", "二"]),
        ], input_format="segments")
        self.assertEqual(report["rejected_rows"], 2)
        self.assertEqual(report["accepted_rows"], 0)
        self.assertFalse(report["manifest_published"])
        self.assertFalse((self.output / "train.jsonl").exists())

    def test_identical_audio_and_text_are_deduplicated(self):
        path = self.wav()
        original = segmented_row(path)
        duplicate = copy.deepcopy(original)
        duplicate.update(voice_end=False, rejects=["Unused alternative"])
        report = self.prepare([original, duplicate], input_format="segments")
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["conflicting_duplicate_rows"], 0)
        self.assertEqual(report["rejected_rows"], 0)
        self.assertEqual(len(self.read_output("duplicates.jsonl")), 1)

    def test_same_explicit_id_different_audio_is_a_rejection_not_a_duplicate(self):
        first, second = self.wav("one.wav"), self.wav("two.wav")
        report = self.prepare([segmented_row(first, id="same-id"),
                               segmented_row(second, id="same-id")], input_format="segments")
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertFalse(report["manifest_published"])
        self.assertEqual(self.read_output("rejected.jsonl")[0]["line"], 2)

    def test_colliding_filename_stems_are_rejected_and_skip_invalid_preserves_first(self):
        first = self.wav("one/same.wav", seconds=1.25)
        second = self.wav("two/same.wav", seconds=2.25)
        report = self.prepare([segmented_row(first), segmented_row(second)],
                              input_format="segments", skip_invalid=True)
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(self.tasks()[0]["audio_path"], str(first))
        self.assertEqual(self.tasks()[0]["duration_seconds"], 1.25)

    def test_auto_accepts_mixed_formats_and_retains_input_order(self):
        segment_audio = self.wav("segments.wav")
        legacy_audio = self.root / "legacy_audio"
        legacy_audio.mkdir()
        write_pcm(legacy_audio / "legacy.wav")
        report = self.prepare([segmented_row(segment_audio), raw_row(key="legacy")],
                              audio_dir=str(legacy_audio), input_format="auto")
        self.assertTrue(report["manifest_published"])
        self.assertEqual(report["accepted_rows"], 2)
        self.assertEqual(report["rejected_rows"], 0)
        tasks = self.tasks()
        self.assertEqual([task["id"] for task in tasks], ["segments", "legacy"])
        self.assertEqual(tasks[0]["audio_path"], str(segment_audio))
        self.assertEqual(tasks[1]["audio_path"], str((legacy_audio / "legacy.wav").resolve()))
        self.assertEqual(tasks[1]["source_text"], raw_row()["en"]["txt"])
        self.assertEqual(tasks[1]["reference_text"], raw_row()["zh"]["txt"])

    def test_explicit_format_does_not_silently_accept_the_other_schema(self):
        path = self.wav()
        for row, selected in ((segmented_row(path), "legacy"), (raw_row(), "segments")):
            with self.subTest(selected=selected), self.assertRaises(PreparationError):
                normalize_record(row, path.parent, input_format=selected, input_dir=self.input_dir)


if __name__ == "__main__":
    unittest.main()
