import json
from pathlib import Path
import tempfile
import unittest

from flex_gspo.monitoring import Monitor, SourceLookup, read_complete_lines, reward_summary, sample_text, slim_response


class Writer:
    def __init__(self, path):
        self.path = path
        self.events = []

    def scalar(self, *args):
        self.events.append(("scalar", *args))

    def text(self, *args):
        self.events.append(("text", *args))

    def flush(self):
        pass

    def close(self):
        pass


def record(call=1, reward=.5, step=0):
    return {"task_id": "example", "call": call, "global_step": step, "timestamp_unix": 1000 + call,
            "reward": reward, "translation": "我们开会。<script>bad</script>",
            "components": {"quality": .9, "metricx_error": 2.5, "latency_seconds": 1.5,
                           "mean_lag_seconds": 1.5, "length_penalty": 0., "tier": "feasible",
                           "protocol_valid": True, "quality_pass": True, "latency_pass": True, "feasible": True},
            "trajectory": {"complete_input": True, "turns": [{"response": "<|WAIT|>", "audio_end_seconds": 1.},
                                      {"response": "<|FLEX|>我们", "audio_end_seconds": 2.},
                                      {"response": "<|DIRECT|>开会。", "audio_end_seconds": 3.}]}}


class MonitorTests(unittest.TestCase):
    def test_slim_action_state_drops_raw_prose_and_keeps_invalid_wait_distinct(self):
        self.assertEqual(slim_response("secret words without a tag"), "[unrecognized]")
        self.assertEqual(slim_response(" <|FLEX|>abc"), "<|FLEX|>[text]")
        self.assertNotEqual(slim_response("<|WAIT|>illegal text"), "<|WAIT|>")
        self.assertEqual(slim_response(" <|WAIT|> "), "<|WAIT|>")
    def test_partial_utf8_line_waits_and_clean_restart_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank0-pid1.jsonl"
            data = (json.dumps(record(), ensure_ascii=False) + "\n").encode("utf-8")
            path.write_bytes(data[:-4])
            monitor = Monitor(root, writer_factory=Writer)
            self.assertEqual(monitor.poll(), 0)
            self.assertFalse(monitor.writers)
            with path.open("ab") as handle:
                handle.write(data[-4:])
            self.assertEqual(monitor.poll(), 1)
            self.assertEqual(monitor.state["samples"], 1)
            self.assertEqual(monitor.poll(), 0)
            monitor.close()
            monitor = Monitor(root, writer_factory=Writer)
            self.assertEqual(monitor.poll(), 0)
            self.assertFalse(monitor.writers)
            monitor.close()
            self.assertEqual(path.read_bytes(), data)

    def test_callback_fragments_accumulate_without_fake_global_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank2-pid10.jsonl"
            path.write_text(json.dumps(record(reward=0)) + "\n", encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer)
            monitor.poll()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record(reward=1)) + "\n")
            monitor.poll()
            rows = monitor.state["files"]["rewards/rewards-rank2-pid10.jsonl"]["pending"]["rows"]
            self.assertEqual(reward_summary(rows)["reward/mean"], .5)
            self.assertEqual(reward_summary(rows)["candidates"], 2)
            events = monitor.writers[path.stem].events
            self.assertEqual([e[2] for e in events if e[1].endswith("reward/mean")], [0., .5])
            self.assertTrue(all("group_advantage" not in e[1] for e in events))
            self.assertNotIn("我们", (root / "monitor" / "cursor.json").read_text(encoding="utf-8"))
            monitor.close()

    def test_text_sampling_continues_in_later_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank0-pid1.jsonl"
            rows = [record(call=i, step=i) for i in (1, 1, 2, 10, 10, 20)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer, max_samples=1, sample_every=10)
            monitor.poll()
            texts = [e for e in monitor.writers[path.stem].events if e[0] == "text"]
            self.assertEqual([e[3] for e in texts], [1, 10, 20])
            self.assertIn("&lt;script&gt;", texts[0][2])
            self.assertNotIn("<script>", texts[0][2])
            monitor.close()

    def test_invalid_metrics_not_zero_filled_and_action_ratios(self):
        invalid = record(reward=-4)
        invalid["components"] = {"quality": None, "metricx_error": None, "tier": "invalid", "protocol_valid": False}
        result = reward_summary([invalid, record()])
        self.assertEqual(result["quality/mean"], .9)
        self.assertEqual(result["quality/scored_count"], 1)
        self.assertEqual(result["tier/invalid_fraction"], .5)
        self.assertAlmostEqual(result["actions/WAIT_fraction"], 1 / 3)

    def test_diagnostics_and_exact_final_wait(self):
        row = record()
        row["components"]["diagnostics"] = {"first_output_seconds": 2., "max_no_text_gap_seconds": 2.,
                                             "eof_text_fraction": .5, "emitted_units": 4, "reference_units": 4}
        row["trajectory"]["turns"][-1]["response"] = "<|WAIT|>"
        result = reward_summary([row])
        self.assertEqual(result["diagnostics/eof_text_fraction/mean"], .5)
        self.assertEqual(result["complete_input_fraction"], 1.)
        self.assertEqual(result["final_wait_fraction"], 1.)
        row["trajectory"]["turns"][-1]["response"] = "<|WAIT|>illegal text"
        self.assertEqual(reward_summary([row])["final_wait_fraction"], 0.)

    def test_non_string_response_skipped_without_stalling_observer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank0-pid1.jsonl"
            row = record()
            row["trajectory"]["turns"][0]["response"] = None
            path.write_text(json.dumps(row) + "\n" + json.dumps(record()) + "\n", encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer)
            monitor.poll()
            self.assertEqual(monitor.state["samples"], 1)
            self.assertEqual(monitor.poll(), 0)
            monitor.close()

    def test_performance_axis_is_explicit_if_optimizer_step_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "performance").mkdir()
            row = {"call": 2, "global_step": None, "generation_seconds": 11.5,
                   "rollout_requests_count": 12, "timestamp_unix": 1000, "rank": "0", "pid": 123}
            path = root / "performance" / "performance-rank0-pid123.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer)
            monitor.poll()
            events = monitor.writers[path.stem].events
            self.assertEqual(events[0][1], "performance_by_callback/generation_seconds")
            self.assertEqual(events[0][3], 2)
            self.assertTrue((root / "monitor" / "summary.latest.json").is_file())
            monitor.close()

    def test_replacement_opens_separate_series(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank0-pid1.jsonl"
            path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer)
            monitor.poll()
            path.write_text(json.dumps({"reward": -4., "call": 1, "components": {"tier": "invalid"},
                                       "trajectory": {"turns": []}}) + '\n', encoding="utf-8")
            monitor.poll()
            self.assertIn(path.stem + "-replacement1", monitor.writers)
            monitor.close()

    def test_only_one_observer_owns_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(directory, writer_factory=Writer)
            try:
                with self.assertRaisesRegex(ValueError, "Another observer"):
                    Monitor(directory, writer_factory=Writer)
            finally:
                monitor.close()

    def test_optional_source_lookup_reads_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            source.write_text(json.dumps({"flex_task": json.dumps({"id": "example", "source_text": "We meet.",
                                                                  "reference_text": "我们开会。", "audio_path": "/missing.wav"})}) + "\n", encoding="utf-8")
            lookup = SourceLookup(source, root)
            self.assertEqual(lookup.get("example"), ("We meet.", "我们开会。"))
            self.assertIn("We meet.", sample_text(record(), lookup))
            lookup.close()
            lookup = SourceLookup(source, root)
            self.assertEqual(lookup.get("example")[0], "We meet.")
            lookup.close()

    def test_reader_skips_bad_complete_line_not_unfinished_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.jsonl"
            path.write_bytes(b'not-json\n{"ok":true}\n{"pending":')
            rows, offset = read_complete_lines(path, 0)
            self.assertEqual(rows, [{"ok": True}])
            self.assertEqual(path.read_bytes()[offset:], b'{"pending":')

    def test_oneshot_keeps_draining_after_a_blank_read_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rewards").mkdir()
            path = root / "rewards" / "rewards-rank0-pid1.jsonl"
            path.write_text("\n" * 10000 + json.dumps(record()) + "\n", encoding="utf-8")
            monitor = Monitor(root, writer_factory=Writer)
            self.assertEqual(monitor.poll(), 0)
            self.assertTrue(monitor.last_progress)
            self.assertEqual(monitor.poll(), 1)
            self.assertTrue(monitor.last_progress)
            self.assertEqual(monitor.poll(), 0)
            self.assertFalse(monitor.last_progress)
            monitor.close()


if __name__ == "__main__":
    unittest.main()
