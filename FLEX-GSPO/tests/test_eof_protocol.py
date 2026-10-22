"""Bounded finalization, timing provenance and backward-compatible audit tests."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from flex_gspo.online import STATE_KEY
from flex_gspo.prompts import EOF_PROMPT
from flex_gspo.rewards import RewardError
from flex_gspo.trajectory import validate_turn_schedule
from test_online import Hooks, choice_fixture, request_fixture, task_fixture


def short_task():
    task = task_fixture()
    task["duration_seconds"] = 0.8
    task["chunks"] = [{"start_seconds": 0.0, "end_seconds": 0.8, "input_seconds": 1.0}]
    return task


class EofProtocolTests(unittest.TestCase):
    def test_short_all_wait_gets_one_text_only_drain_and_preserves_causal_history(self):
        task = short_task()
        hooks, request = Hooks(), request_fixture(task)
        asyncio.run(hooks.on_trajectory_start([request]))
        request.messages.append({"role": "assistant", "content": "<|WAIT|>"})
        last_audio = asyncio.run(hooks.on_turn_end(request, choice_fixture(), 1))
        trajectory = last_audio["rollout_infos"]["flex_trajectory"]
        self.assertFalse(last_audio["done"])
        self.assertTrue(trajectory["complete_input"])
        self.assertFalse(trajectory["episode_complete"])
        before = deepcopy(request.messages)
        audios = list(request.audios)
        with patch("flex_gspo.online.time.monotonic", return_value=10.0):
            hooks.step(request, choice_fixture(), 1)
        self.assertEqual(request.messages[:-1], before)
        self.assertEqual(request.messages[-1], {"role": "user", "content": EOF_PROMPT})
        self.assertEqual(request.audios, audios)
        serialized = json.dumps(request.messages, ensure_ascii=False)
        self.assertNotIn(task["source_text"], serialized)
        self.assertNotIn(task["reference_text"], serialized)
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            hooks.step(request, choice_fixture(), 1)
        request.messages.append({"role": "assistant", "content": "<|DIRECT|>生日是七月四日。"})
        with patch("flex_gspo.online.time.monotonic", return_value=11.25):
            final = asyncio.run(hooks.on_turn_end(
                request, choice_fixture("<|DIRECT|>生日是七月四日。"), 2))
        result = final["rollout_infos"]["flex_trajectory"]
        self.assertTrue(final["done"])
        self.assertEqual(result["schema_version"], 2)
        self.assertTrue(result["episode_complete"])
        self.assertEqual(result["eof_elapsed_seconds"], 1.25)
        self.assertEqual(result["turns"][-1]["audio_end_seconds"], 0.8)
        self.assertEqual(final["rollout_infos"]["audios"], audios)
        with self.assertRaises(RuntimeError):
            hooks.step(request, choice_fixture(), 2)
        with self.assertRaises(RuntimeError):
            asyncio.run(hooks.on_turn_end(request, choice_fixture(), 3))

    def test_short_audio_requires_room_for_eof_before_inference(self):
        hooks = Hooks()
        hooks.max_turns = 1
        with self.assertRaisesRegex(ValueError, "needs 2 turns"):
            asyncio.run(hooks.on_trajectory_start([request_fixture(short_task())]))

    def test_eof_wait_is_allowed_and_not_forced_to_direct(self):
        hooks, request = Hooks(), request_fixture(short_task())
        asyncio.run(hooks.on_trajectory_start([request]))
        asyncio.run(hooks.on_turn_end(request, choice_fixture("<|DIRECT|>生日。"), 1))
        hooks.step(request, choice_fixture(), 1)
        result = asyncio.run(hooks.on_turn_end(request, choice_fixture(), 2))
        trajectory = result["rollout_infos"]["flex_trajectory"]
        self.assertTrue(trajectory["valid"])
        self.assertTrue(trajectory["eof_completed"])
        self.assertEqual(trajectory["turns"][-1]["response"], "<|WAIT|>")

    def test_invalid_finalization_stops_but_preserves_raw_output(self):
        hooks, request = Hooks(), request_fixture(short_task())
        asyncio.run(hooks.on_trajectory_start([request]))
        asyncio.run(hooks.on_turn_end(request, choice_fixture(), 1))
        hooks.step(request, choice_fixture(), 1)
        result = asyncio.run(hooks.on_turn_end(request, choice_fixture("wrong format"), 2))
        trajectory = result["rollout_infos"]["flex_trajectory"]
        self.assertTrue(result["done"])
        self.assertFalse(trajectory["valid"])
        self.assertTrue(trajectory["complete_input"])
        self.assertTrue(trajectory["episode_complete"])
        self.assertEqual(trajectory["turns"][-1]["response"], "wrong format")

    def test_eof_observation_must_not_be_forged_or_missing(self):
        hooks, request = Hooks(), request_fixture(short_task())
        asyncio.run(hooks.on_trajectory_start([request]))
        asyncio.run(hooks.on_turn_end(request, choice_fixture(), 1))
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            asyncio.run(hooks.on_turn_end(request, choice_fixture(), 2))
        hooks.step(request, choice_fixture(), 1)
        request.messages[-1]["content"] = "fabricated future source"
        with self.assertRaisesRegex(RuntimeError, "explicit text-only EOF"):
            asyncio.run(hooks.on_turn_end(request, choice_fixture(), 2))


class ScheduleValidationTests(unittest.TestCase):
    def test_old_audio_only_record_remains_auditable_but_is_not_eof_complete(self):
        task = short_task()
        legacy = [{"audio_end_seconds": 0.8, "response": "<|WAIT|>", "finish_reason": "stop"}]
        result = validate_turn_schedule(task, legacy)
        self.assertTrue(result["complete_input"])
        self.assertFalse(result["eof_completed"])
        self.assertIsNone(result["eof_elapsed_seconds"])

    def test_one_explicit_eof_after_audio_is_valid(self):
        task = short_task()
        turns = [{"audio_end_seconds": 0.8, "kind": "audio"},
                 {"audio_end_seconds": 0.8, "kind": "eof", "eof_elapsed_seconds": 1.2}]
        result = validate_turn_schedule(task, turns)
        self.assertTrue(result["eof_completed"])
        self.assertEqual(result["audio_turns"], 1)

    def test_infrastructure_errors_raise_not_fake_quality_rewards(self):
        audio = {"audio_end_seconds": 0.8, "kind": "audio"}
        eof = {"audio_end_seconds": 0.8, "kind": "eof", "eof_elapsed_seconds": 0.1}
        bad_cases = [
            [eof], [audio, audio], [audio, eof, eof],
            [{**audio, "audio_end_seconds": 0.7}],
            [audio, {**eof, "audio_end_seconds": 0.9}],
            [audio, {**eof, "eof_elapsed_seconds": float("nan")}],
            [audio, {**eof, "eof_elapsed_seconds": -1}],
            [audio, {**eof, "eof_elapsed_seconds": True}],
            [audio, {"audio_end_seconds": 0.8, "kind": "eof"}],
            [{**audio, "eof_elapsed_seconds": 0.1}],
        ]
        for turns in bad_cases:
            with self.subTest(turns=turns), self.assertRaises(RewardError):
                validate_turn_schedule(short_task(), turns)

    def test_failed_audio_prefix_is_valid_infrastructure_but_not_complete(self):
        result = validate_turn_schedule(task_fixture(), [{"kind": "audio", "audio_end_seconds": 1.0}])
        self.assertFalse(result["complete_input"])
        self.assertFalse(result["episode_complete"])


if __name__ == "__main__":
    unittest.main()
