"""Optional telemetry must preserve calls, exceptions and context semantics."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo import performance


class PerformanceTests(unittest.TestCase):
    def make_trainer(self):
        class Trainer:
            def __init__(self):
                self.calls = []
                self.state = SimpleNamespace(global_step=7)
                self.error = None

            def _move_model_to_vllm(self):
                self.calls.append("sync")
                return "synced"

            def _rollout_requests(self, requests, config=None):
                self.calls.append(("infer", requests, config))
                if self.error:
                    raise self.error
                return requests

            @contextmanager
            def offload_context(self):
                self.calls.append("offload")
                try:
                    yield "context-result"
                finally:
                    self.calls.append("reload")
        return Trainer()

    def read_rows(self, directory):
        return [json.loads(line) for path in Path(directory).glob("performance-*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()]

    def test_disabled_does_not_wrap_or_create_files(self):
        trainer = self.make_trainer()
        original = type(trainer)._rollout_requests
        with patch.dict(os.environ, {}, clear=True):
            performance.install_phase_timers(type(trainer))
            self.assertIsNone(performance.start_generation(trainer))
            self.assertIs(type(trainer)._rollout_requests, original)
            performance.finish_generation(None, status="ok")

    def test_phase_calls_and_metadata_preserved_with_one_row_per_generation(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "FLEX_PERF_LOG_DIR": directory, "RANK": "2"}):
            performance.install_phase_timers(type(trainer))
            session = performance.start_generation(trainer)
            self.assertEqual(trainer._move_model_to_vllm(), "synced")
            requests = [object()]
            with trainer.offload_context() as context_result:
                self.assertEqual(context_result, "context-result")
                self.assertIs(trainer._rollout_requests(requests, "config"), requests)
                self.assertEqual(trainer._rollout_requests([], "config"), [])
            performance.finish_generation(session, status="ok", local_trajectories=8)
            rows = self.read_rows(directory)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row["rank"], row["global_step"], row["step_source"]),
                             ("2", 7, "trainer.state.global_step"))
            self.assertEqual((row["call"], row["status"], row["local_trajectories"]), (1, "ok", 8))
            self.assertEqual(row["rollout_requests_count"], 2)
            for name in (*performance._SECONDS, "generation_seconds"):
                self.assertGreaterEqual(row[name], 0)
            self.assertEqual(trainer.calls, ["sync", "offload", ("infer", requests, "config"),
                                             ("infer", [], "config"), "reload"])
            self.assertIsNone(performance._ACTIVE.get())

    def test_outside_scope_and_other_trainer_do_not_pollute_metrics(self):
        trainer = self.make_trainer()
        other = type(trainer)()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            self.assertEqual(trainer._rollout_requests([1]), [1])
            session = performance.start_generation(trainer)
            self.assertEqual(other._rollout_requests([2]), [2])
            performance.finish_generation(session, status="ok")
            self.assertEqual(self.read_rows(directory)[0]["rollout_requests_count"], 0)

    def test_original_exception_and_context_cleanup_preserved(self):
        trainer = self.make_trainer()
        error = RuntimeError("generation failed")
        trainer.error = error
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            session = performance.start_generation(trainer)
            try:
                with self.assertRaises(RuntimeError) as caught:
                    with trainer.offload_context():
                        trainer._rollout_requests([])
                self.assertIs(caught.exception, error)
            finally:
                performance.finish_generation(session, status="error")
            row = self.read_rows(directory)[0]
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["rollout_requests_count"], 1)
            self.assertEqual(trainer.calls[-1], "reload")
            self.assertIsNone(performance._ACTIVE.get())

    def test_context_exception_suppression_return_is_unchanged(self):
        trainer = self.make_trainer()
        seen = []
        class SuppressingContext:
            def __enter__(self):
                return 23
            def __exit__(self, *exc):
                seen.append(exc)
                return True
        type(trainer).offload_context = lambda self: SuppressingContext()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            session = performance.start_generation(trainer)
            with trainer.offload_context() as value:
                self.assertEqual(value, 23)
                raise ValueError("suppressed by upstream, not telemetry")
            performance.finish_generation(session, status="ok")
            self.assertIs(seen[0][0], ValueError)

    def test_file_failure_is_nonfatal_and_resets_context(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            session = performance.start_generation(trainer)
            with patch.object(Path, "open", side_effect=OSError("disk full")), self.assertLogs("flex.performance", "WARNING"):
                performance.finish_generation(session, status="error")
            self.assertIsNone(performance._ACTIVE.get())

    def test_installer_is_idempotent_and_step_fallback_is_labelled(self):
        trainer = self.make_trainer()
        trainer.state.global_step = None
        trainer._step = 12
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            original = type(trainer)._rollout_requests
            performance.install_phase_timers(type(trainer))
            self.assertIs(type(trainer)._rollout_requests, original)
            for _ in range(2):
                session = performance.start_generation(trainer)
                trainer._rollout_requests([])
                performance.finish_generation(session, status="ok")
            rows = self.read_rows(directory)
            self.assertEqual([row["call"] for row in rows], [1, 2])
            self.assertEqual([row["rollout_requests_count"] for row in rows], [1, 1])
            self.assertEqual(rows[0]["global_step"], 12)
            self.assertEqual(rows[0]["step_source"], "trainer._step")


if __name__ == "__main__":
    unittest.main()
