"""Optional telemetry must preserve calls, exceptions and context semantics."""
from contextlib import contextmanager
from functools import wraps
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

    def read_phases(self, directory):
        return [json.loads(line) for path in Path(directory).glob("phases-*.jsonl")
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
            events = self.read_phases(directory)
            self.assertEqual([(row["phase"], row["event"]) for row in events], [
                ("generation", "begin"), ("weight_sync", "begin"), ("weight_sync", "end"),
                ("offload_to_cpu", "begin"), ("offload_to_cpu", "end"),
                ("rollout_requests", "begin"), ("rollout_requests", "end"),
                ("rollout_requests", "begin"), ("rollout_requests", "end"),
                ("reload_to_gpu", "begin"), ("reload_to_gpu", "end"), ("generation", "end")])
            self.assertEqual([row["event_sequence"] for row in events], list(range(1, 13)))
            self.assertEqual(len({row["phase_id"] for row in events}), 6)
            self.assertTrue(all(row["call"] == 1 and row["global_step"] == 7 for row in events))
            self.assertTrue(all(row["status"] == "ok" for row in events if row["event"] == "end"))

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
            self.assertEqual([row["phase"] for row in self.read_phases(directory)],
                             ["generation", "generation"])

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
            rollout_end = next(row for row in self.read_phases(directory)
                               if row["phase"] == "rollout_requests" and row["event"] == "end")
            self.assertEqual(rollout_end["status"], "error")

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

    def test_begin_is_visible_before_underlying_call_finishes(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            def rollout(owner, requests):
                events = self.read_phases(directory)
                self.assertEqual((events[-1]["phase"], events[-1]["event"]),
                                 ("rollout_requests", "begin"))
                self.assertEqual(self.read_rows(directory), [])
                return requests
            type(trainer)._rollout_requests = rollout
            performance.install_phase_timers(type(trainer))
            session = performance.start_generation(trainer)
            try:
                self.assertEqual(trainer._rollout_requests([1]), [1])
            finally:
                performance.finish_generation(session, status="ok")

    def test_decorated_sleep_call_site_is_scoped_and_errors_propagate(self):
        trainer = self.make_trainer()
        trainer.engine = SimpleNamespace(engine=object())
        calls = []
        failure = RuntimeError("upstream cleanup failed")
        def original_sleep(engine, level, **kwargs):
            calls.append((engine, level, kwargs))
            if level == 99:
                raise failure
            return "slept"
        namespace = {"sleep_vllm_engine": original_sleep}
        exec("def generate(self, level):\n"
             "    return sleep_vllm_engine(self.engine.engine, level, suppress_errors=False)\n", namespace)
        @wraps(namespace["generate"])
        def decorated(*args, **kwargs):
            return namespace["generate"](*args, **kwargs)
        type(trainer)._generate_completions = decorated
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            installed = namespace["sleep_vllm_engine"]
            performance.install_phase_timers(type(trainer))
            self.assertIs(namespace["sleep_vllm_engine"], installed)
            self.assertEqual(trainer._generate_completions(2), "slept")
            self.assertEqual(self.read_phases(directory), [])
            session = performance.start_generation(trainer)
            try:
                other_engine = object()
                self.assertEqual(installed(other_engine, 2), "slept")
                self.assertEqual(trainer._generate_completions(2), "slept")
                with self.assertRaises(RuntimeError) as caught:
                    trainer._generate_completions(99)
                self.assertIs(caught.exception, failure)
            finally:
                performance.finish_generation(session, status="error")
            events = self.read_phases(directory)
            sleep_events = [row for row in events if row["phase"] == "vllm_sleep"]
            self.assertEqual([row["event"] for row in sleep_events], ["begin", "end", "begin", "end"])
            self.assertEqual([row["status"] for row in sleep_events if row["event"] == "end"], ["ok", "error"])
            self.assertEqual(calls[-1], (trainer.engine.engine, 99, {"suppress_errors": False}))
            self.assertGreaterEqual(self.read_rows(directory)[0]["vllm_sleep_seconds"], 0)

    def test_phase_write_failure_does_not_replace_underlying_result_or_error(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            performance.install_phase_timers(type(trainer))
            with patch.object(Path, "open", side_effect=OSError("disk full")), self.assertLogs("flex.performance", "WARNING"):
                session = performance.start_generation(trainer)
                try:
                    self.assertEqual(trainer._rollout_requests([1]), [1])
                    trainer.error = ValueError("original failure")
                    with self.assertRaises(ValueError) as caught:
                        trainer._rollout_requests([])
                    self.assertIs(caught.exception, trainer.error)
                finally:
                    performance.finish_generation(session, status="error")
            self.assertIsNone(performance._ACTIVE.get())

    def test_disabling_after_installation_leaves_calls_unlogged(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
                performance.install_phase_timers(type(trainer))
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(performance.start_generation(trainer))
                self.assertEqual(trainer._rollout_requests([1]), [1])
                with trainer.offload_context():
                    pass
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
