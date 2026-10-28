import argparse
from contextlib import redirect_stdout
import copy
import csv
import io
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Scripts import collect_reward_revision as audit


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class RewardRevisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.launch = self.root / "launch_001.json"
        self.native = self.root / "v0-test" / "logging.jsonl"
        self.dataset = self.root / "train.jsonl"
        options = {"--num_generations": "8", "--steps_per_generation": "1", "--num_iterations": "1",
                   "--scale_rewards": "group", "--reward_funcs": "qalco_online", "--reward_weights": "1",
                   "--global_batch_size": "32", "--nproc_per_node": "4", "--nnodes": "1"}
        plan = {"session_id": "session-a", "run_id": "run-a", "argv": [x for kv in options.items() for x in kv]}
        self.launch.write_text(json.dumps(plan), encoding="utf-8")
        self.rows = [[] for _ in range(4)]
        self.original = {}
        tasks = {}
        for gs in range(2):
            for group in range(4):
                task_id = "repeated-empty" if group == 0 else f"task-{gs}-{group}"
                task = {"id": task_id, "source_text": "This is a test.", "reference_text": "这是测试。",
                        "duration_seconds": 2.0, "audio_path": "/unread/audio.wav"}
                tasks[task_id] = task
                for index in range(8):
                    empty = group == 0 or (group == 1 and index < 2)
                    invalid = group == 2 and index == 0
                    text = "" if empty else "这是测试。" + "补充" * index
                    response = "<|WAIT|>" if empty else "<|DIRECT|>" + text
                    turns = [{"kind": "audio", "response": response, "finish_reason": "stop", "audio_end_seconds": 1.0},
                             {"kind": "audio", "response": "<|WAIT|>", "finish_reason": "eos", "audio_end_seconds": 2.0},
                             {"kind": "eof", "response": "MALFORMED..." if invalid else "<|WAIT|>",
                              "finish_reason": "stop", "audio_end_seconds": 2.0}]
                    reward = -4.0 if empty or invalid else .2 + index / 100
                    request = f"request-{gs}-{group}-{index}"
                    row = {"session_id": "session-a", "run_id": "run-a", "global_step": gs,
                           "call": gs + 1, "timestamp_unix": 1000 + gs, "task_id": task_id,
                           "task_hash": audit.digest(task), "request_id": request, "reward": reward,
                           "duration_seconds": 2.0, "translation": text,
                           "components": {"global_group_id": f"gid-{gs}-{group}", "group_candidate_index": index,
                                          "group_size": 8, "group_finalized": True, "quality": None if empty or invalid else .85 + index / 1000,
                                          "latency_penalty": None if empty or invalid else .05, "latency_seconds": None if empty or invalid else 1.0,
                                          "protocol_valid": not invalid, "empty_translation": empty,
                                          "quality_pass": not empty and not invalid, "feasible": not empty and not invalid,
                                          "weighted_brevity_penalty": 0.0, "tier": "invalid" if empty or invalid else "feasible"},
                           "trajectory": {"valid": not invalid, "complete_input": True, "episode_complete": True,
                                          "eof_completed": True, "turns": turns}}
                    self.rows[index % 4].append(row)
                    self.original[request] = copy.deepcopy(row)
        self.rows[0].append({"session_id": "foreign", "run_id": "run-a", "global_step": 0})
        self.rows[1].append({"session_id": "session-a", "run_id": "run-a", "global_step": 2})
        write_rows(self.dataset, [{"flex_task": json.dumps(t, ensure_ascii=False)} for t in tasks.values()])
        self.flush_rewards()
        native_rows = []
        for gs in range(2):
            rewards = [r["reward"] for r in self.original.values() if r["global_step"] == gs]
            native_rows.append({"iteration": f"{gs + 1}/3000", "reward": statistics.mean(rewards),
                                "loss": .0001, "learning_rate": 1e-7})
        write_rows(self.native, native_rows)
        self.args = argparse.Namespace(run_dir=str(self.root), native_log=str(self.native),
                                       launch_file=str(self.launch), dataset=str(self.dataset),
                                       groups=20, block_size=1, output_dir=str(self.root / "audit"))

    def flush_rewards(self):
        for rank, rows in enumerate(self.rows):
            write_rows(self.root / "rewards" / f"rewards-rank{rank}-pid{100 + rank}.jsonl", rows)

    def test_v4_export_uses_center_only_from_verified_launch(self):
        plan = json.loads(self.launch.read_text(encoding="utf-8"))
        index = plan["argv"].index("--scale_rewards") + 1
        plan["argv"][index] = "none"
        self.launch.write_text(json.dumps(plan), encoding="utf-8")
        for rows in self.rows:
            for row in rows:
                if "components" in row:
                    row["reward_version"] = 4
        self.flush_rewards()
        out, report, selected = self.run_collect()
        self.assertEqual(report["advantage_scale"], "none")
        expected = {}
        for group in selected:
            mean = group["group"]["reward_mean"]
            for row in group["candidates"]:
                expected.setdefault(row["components"]["group_candidate_index"], set()).add(round(row["reward"] - mean, 12))
        seen = 0
        for path in out.glob("review_part_*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("候选: "):
                    value = json.loads(line[len("候选: "):])
                    self.assertIn(round(value["estimated_advantage"], 12), expected[value["candidate_index"]])
                    seen += 1
        self.assertEqual(seen, len(selected) * 8)

    def test_v4_wrong_launch_scaling_rejected(self):
        self.rows[0][0]["reward_version"] = 4
        self.flush_rewards()
        with self.assertRaisesRegex(ValueError, "scaling disagree"):
            self.run_collect()

    def run_collect(self):
        with redirect_stdout(io.StringIO()):
            out = audit.collect(self.args)
        report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        selected = [json.loads(line) for line in (out / "selected_groups.jsonl").read_text(encoding="utf-8").splitlines()]
        return out, report, selected

    def test_full_groups_preserve_original_rows_missing_scores_and_eof(self):
        out, report, selected = self.run_collect()
        self.assertEqual(report["rewards"]["groups"], 8)
        self.assertEqual(report["rewards"]["candidates"], 64)
        self.assertEqual(report["all_wait_completed_groups"], 2)
        self.assertEqual(report["mixed_empty_groups"], 2)
        self.assertEqual(report["ignored_records"], {"other_session_or_run": 1, "not_completed_native_step": 1})
        self.assertEqual(report["first_failure_kind"], {"eof": 2})
        self.assertEqual(report["native_reward_mismatch_count"], 0)
        self.assertEqual(report["selected_groups"], 7)
        self.assertEqual(report["selected_candidates"], 56)
        self.assertEqual(report["top_empty_tasks"][0]["groups_seen"], 2)
        self.assertEqual(report["top_empty_tasks"][0]["all_wait_groups"], 2)
        null_count = 0
        for group in selected:
            self.assertEqual(len(group["candidates"]), 8)
            self.assertEqual({r["components"]["group_candidate_index"] for r in group["candidates"]}, set(range(8)))
            for row in group["candidates"]:
                self.assertEqual(row, self.original[row["request_id"]])
                self.assertEqual(len(row["trajectory"]["turns"]), 3)
                null_count += row["components"]["quality"] is None
        self.assertGreater(null_count, 0)
        groups = [json.loads(line) for line in (out / "group_statistics.jsonl").read_text(encoding="utf-8").splitlines()]
        mixed = next(g for g in groups if g["global_group_id"] == "gid-0-1")
        self.assertEqual(mixed["quality_count"], 6)
        self.assertIsNone(mixed["quality_std_population"])
        self.assertEqual(mixed["component_statistics"]["quality"]["missing"], 2)
        text = "".join(p.read_text(encoding="utf-8") for p in out.glob("review_part_*.txt"))
        self.assertIn("MALFORMED...", text)
        self.assertIn("<|WAIT|>", text)
        self.assertEqual(text.count("逐轮输出:"), 56 * 3)
        with (out / "all_steps.csv").open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(float(csv_rows[0]["learning_rate"]), 1e-7)

    def test_incomplete_step_excluded_not_shortened_to_g7(self):
        self.rows[3] = [r for r in self.rows[3] if r.get("request_id") != "request-0-3-7"]
        self.flush_rewards()
        _, report, selected = self.run_collect()
        self.assertIn("1", report["excluded_reward_steps"])
        self.assertEqual(report["rewards"]["groups"], 4)
        self.assertTrue(all(g["group"]["native_step"] == 2 for g in selected))
        self.assertTrue(all(len(g["candidates"]) == 8 for g in selected))

    def test_duplicate_indices_reject_entire_step(self):
        row = next(r for r in self.rows[3] if r.get("request_id") == "request-0-3-7")
        row["components"]["group_candidate_index"] = 6
        self.flush_rewards()
        _, report, selected = self.run_collect()
        self.assertIn("1", report["excluded_reward_steps"])
        self.assertEqual(report["rewards"]["groups"], 4)
        self.assertTrue(all(g["group"]["native_step"] == 2 for g in selected))

    def test_task_hash_mismatch_fails_before_output(self):
        data = [json.loads(line) for line in self.dataset.read_text(encoding="utf-8").splitlines()]
        task = json.loads(data[0]["flex_task"])
        task["source_text"] = "Changed transcript."
        data[0]["flex_task"] = json.dumps(task)
        write_rows(self.dataset, data)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "differs from original"):
            audit.collect(self.args)
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_stale_native_refused(self):
        stamp = self.launch.stat().st_mtime - 100
        os.utime(self.native, (stamp, stamp))
        with self.assertRaisesRegex(ValueError, "predates launch"):
            audit.collect(self.args)

    def test_incomplete_wait_only_trajectory_is_not_completed_all_wait(self):
        for rows in self.rows:
            for row in rows:
                if row.get("task_id") == "repeated-empty":
                    row["trajectory"]["eof_completed"] = False
                    row["trajectory"]["turns"] = row["trajectory"]["turns"][:-1]
        self.flush_rewards()
        _, report, _ = self.run_collect()
        self.assertEqual(report["all_wait_completed_groups"], 0)
        self.assertEqual(report["observed_turn_statistics"]["empty_without_eof"], 16)

    def test_partial_tail_is_ignored_but_corrupt_complete_line_fails(self):
        path = self.root / "rewards" / "rewards-rank0-pid100.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"unfinished":')
        out, report, _ = self.run_collect()
        self.assertEqual(report["rewards"]["candidates"], 64)
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(any(m.get("partial_tail_ignored") for m in provenance["log_snapshots"]))
        self.args.output_dir = str(self.root / "second-audit")
        with path.open("ab") as stream:
            stream.write(b"\n")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Corrupt complete"):
            audit.collect(self.args)

    def test_selected_row_mutation_detected_by_offset_hash(self):
        path = self.root / "probe.jsonl"
        write_rows(path, [{"x": 1}])
        row, pointer = next(audit.indexed_snapshot(path, []))
        self.assertEqual(row, {"x": 1})
        write_rows(path, [{"x": 2}])
        with self.assertRaisesRegex(ValueError, "changed during audit"):
            audit.read_original({"pointer": pointer})


if __name__ == "__main__":
    unittest.main()
