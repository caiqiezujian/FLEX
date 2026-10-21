"""Contracts for the existing task manifests consumed by GSPO."""

from __future__ import annotations

from copy import deepcopy
import unittest

from flex_gspo.data import DataPreparationError, validate_task


def make_task():
    return {
        "schema_version": 1,
        "id": "example",
        "audio_path": "/server/audio/example.wav",
        "source_lang": "en",
        "target_lang": "zh",
        "duration_seconds": 2.25,
        "sample_rate": 16000,
        "chunks": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.25, "input_seconds": 1.0},
        ],
        "source_text": "We are here.",
        "reference_text": "我们在这里。",
        "source_segments": None,
        "transcript_provenance": {"kind": "provided", "is_gold": False},
        "quality_ready": True,
    }


class TaskValidationTests(unittest.TestCase):
    def assert_invalid(self, task):
        with self.assertRaises(DataPreparationError):
            validate_task(task)

    def test_existing_manifest_is_validated_without_reading_audio(self):
        task = make_task()
        original = deepcopy(task)
        self.assertIs(validate_task(task), task)
        self.assertEqual(task, original)

    def test_version_id_audio_and_direction_are_checked(self):
        for field, values in (
            ("schema_version", [None, True, 2]),
            ("id", [None, "", " ", 3]),
            ("audio_path", [None, "", " ", []]),
            ("source_lang", ["auto", "de", "zh"]),
            ("target_lang", ["auto", "de", "en"]),
            ("sample_rate", [8000, True, None]),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    task = make_task()
                    task[field] = value
                    self.assert_invalid(task)
        self.assert_invalid([])

    def test_reverse_direction_remains_schema_compatible(self):
        task = make_task()
        task.update(source_lang="zh", target_lang="en")
        self.assertIs(validate_task(task), task)

    def test_duration_must_be_positive_finite(self):
        for value in (None, True, 0, -1, float("nan"), float("inf"), "2.25"):
            with self.subTest(value=value):
                task = make_task()
                task["duration_seconds"] = value
                self.assert_invalid(task)

    def test_chunks_must_cover_audio_without_gaps_overlaps_or_overflow(self):
        variants = []
        for field, value in (
            ("start_seconds", 0.1),
            ("start_seconds", -1),
            ("end_seconds", 1.1),
            ("end_seconds", 0),
            ("input_seconds", 0.5),
            ("input_seconds", float("inf")),
        ):
            task = make_task()
            task["chunks"][0][field] = value
            variants.append(task)
        task = make_task()
        task["chunks"][-1]["end_seconds"] = 2.3
        variants.append(task)
        task = make_task()
        task["chunks"] = task["chunks"][:-1]
        variants.append(task)
        for chunks in (None, [], {}, [None]):
            task = make_task()
            task["chunks"] = chunks
            variants.append(task)
        for task in variants:
            with self.subTest(chunks=task["chunks"]):
                self.assert_invalid(task)

    def test_readiness_reflects_source_text_not_reference(self):
        task = make_task()
        task["reference_text"] = None
        validate_task(task)
        task["source_text"] = " "
        self.assert_invalid(task)
        task["quality_ready"] = False
        validate_task(task)
        task["quality_ready"] = 0
        self.assert_invalid(task)

    def test_text_types_are_not_silently_coerced(self):
        for field in ("source_text", "reference_text"):
            for value in (False, [], {}):
                with self.subTest(field=field, value=value):
                    task = make_task()
                    task[field] = value
                    self.assert_invalid(task)

    def test_source_segments_preserve_word_spacing(self):
        task = make_task()
        segments = [
            {"start_seconds": 0, "end_seconds": 0.5, "text": " We"},
            {"start_seconds": 0.5, "end_seconds": 1, "text": "'re"},
            {"start_seconds": 1, "end_seconds": 2, "text": " here."},
        ]
        task["source_segments"] = deepcopy(segments)
        validate_task(task)
        self.assertEqual(task["source_segments"], segments)

    def test_source_segments_reject_invalid_order_time_and_text(self):
        for segments in (
            {},
            [None],
            [{"start_seconds": 0, "end_seconds": 1, "text": ""}],
            [{"start_seconds": 0, "end_seconds": 3, "text": "late"}],
            [{"start_seconds": -1, "end_seconds": 1, "text": "bad"}],
            [{"start_seconds": 0, "end_seconds": 0, "text": "zero"}],
            [{"start_seconds": 0, "end_seconds": 1, "text": "one"},
             {"start_seconds": 0.5, "end_seconds": 2, "text": "overlap"}],
        ):
            with self.subTest(segments=segments):
                task = make_task()
                task["source_segments"] = segments
                self.assert_invalid(task)

    def test_transcript_provenance_remains_explicit_and_not_gold_by_assumption(self):
        task = make_task()
        task["transcript_provenance"] = {
            "kind": "asr", "is_gold": False, "segment_granularity": "word"
        }
        validate_task(task)
        for provenance in (
            [], {}, {"kind": ""},
            {"kind": "asr", "is_gold": True},
            {"kind": "provided", "is_gold": 1},
            {"kind": "provided", "segment_granularity": ""},
        ):
            with self.subTest(provenance=provenance):
                task["transcript_provenance"] = provenance
                self.assert_invalid(task)


if __name__ == "__main__":
    unittest.main()
