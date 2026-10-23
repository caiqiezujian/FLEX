"""Stopped-run analysis is text-only and must not mix runs or round LR to 0."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import analyze_training_run as audit


class TrainingAuditTests(unittest.TestCase):
    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def candidate(self, index=0, call=1, text="你好", tier="quality_fail", latency=1):
        return {"call": call, "global_step": call-1, "timestamp_unix": call,
                "candidate_index_in_local_batch": index, "task_id": "sample",
                "translation": text, "reward": -2.0588235294117645,
                "components": {"tier": tier, "quality": .8, "latency_seconds": latency,
                               "metricx_error": 5., "length_penalty": 0., "error": "no measurable translation"},
                "trajectory": {"valid": True, "complete_input": True, "turns": [
                    {"response": "<|DIRECT|>" + text, "audio_end_seconds": 2., "finish_reason": "stop"}]}}

    def test_lr_and_nonfinite_are_preserved(self):
        w = audit.native_window([{"iteration": "1/4100", "learning_rate": 5.5e-7, "loss": float("nan")}])
        self.assertEqual(w["metrics"]["learning_rate"]["mean"], 5.5e-7)
        self.assertEqual(w["metrics"]["loss"]["count"], 0)
        self.assertEqual(audit.json_safe({"v": float("nan")}), {"v": "nan"})

    def test_stream_windows_timing_pair_and_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rewards-rank0-pid1.jsonl"
            records = [self.candidate(), self.candidate(1, latency=12), self.candidate(call=2, text="再见")]
            self.write_rows(path, records)
            cfg = {"quality_floor": .85, "latency_budget_seconds": 3., "latency_weight": .2, "length_weight": .1}
            report, examples, _ = audit.audit_rewards(path, 1, cfg, 2)
            self.assertEqual(report["early"]["candidates"], 2)
            self.assertEqual(report["recent"]["candidates"], 1)
            self.assertFalse(report["windows_overlap"])
            self.assertEqual(report["formula_recheck"]["checked"], 3)
            self.assertEqual(report["formula_recheck"]["mismatches"], 0)
            self.assertEqual(report["same_text_different_timing_local_groups"], 1)
            self.assertIn("faster", examples[0])

    def test_invalid_errors_partial_indices_and_bad_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rewards-rank0-pid1.jsonl"
            row = self.candidate(index=2, text="", tier="invalid")
            row["trajectory"]["turns"][0]["response"] = "<|WAIT|>"
            self.write_rows(path, [row])
            with path.open("ab") as f:
                f.write(b'{"unfinished":')
            result, examples, _ = audit.audit_rewards(path, 10, None, 2)
            self.assertEqual(result["invalid_reasons"], {"no measurable translation": 1})
            self.assertEqual(result["source"]["bad_lines"], 1)
            self.assertEqual(result["noncontiguous_candidate_index_callbacks"], 1)
            self.assertEqual(result["all_wait_candidates"], 1)
            self.assertEqual(examples[0]["turn_count"], 1)

    def test_exact_wait_preserved_not_malformed_wait_text(self):
        r = self.candidate()
        r["trajectory"]["turns"][0]["response"] = "<|WAIT|>not allowed"
        self.assertNotEqual(audit.compact(r)["trajectory"]["turns"][0]["response"], "<|WAIT|>")

    def test_config_needs_matching_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/"config.json"
            config.write_text(json.dumps({"reward": {"provider": "qalco", "quality_floor": .85,
                "latency_metric": "mean_lag", "latency_budget_seconds": 3., "reference_length_ratio": 1.2,
                "latency_weight": .2, "length_weight": .1, "metricx": {"mode": "reference"}}}), encoding="utf-8")
            self.assertIsNone(audit.load_settings(config, root)[0])
            (root/"launch_1.json").write_text(json.dumps({"run_identity": {
                "reward_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}}), encoding="utf-8")
            settings, result = audit.load_settings(config, root)
            self.assertEqual(settings["quality_floor"], .85)
            self.assertNotIn("service_url", json.dumps(result))

    def test_checkpoint_presence_is_not_resume_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = root/"v0"/"checkpoint-50"
            (cp/"iter_0000050").mkdir(parents=True)
            (cp/"latest_checkpointed_iteration.txt").write_text("50")
            (cp/"iter_0000050"/"common.pt").write_bytes(b"metadata")
            inventory = audit.checkpoint_inventory(root)
            self.assertTrue(inventory[0]["common_pt_present"])
            self.assertFalse(inventory[0]["nonmetadata_payload_present"])
            self.assertFalse(inventory[0]["resumability_verified"])

    def test_v3_arithmetic_recheck_uses_logged_group_cost_not_old_formula(self):
        from flex_gspo.gated_reward import combine_v3
        from flex_gspo.qalco import validate_settings
        config = Path(__file__).resolve().parents[1] / "configs" / "qalco_v3.json"
        settings = validate_settings(json.loads(config.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch_1.json").write_text(json.dumps({"run_identity": {
                "reward_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}}), encoding="utf-8")
            loaded, info = audit.load_settings(config, root)
            self.assertEqual(loaded["version"], 3)
            self.assertIn("v3", info["settings"])
            result = combine_v3(.9, 1, .25, .1, settings)
            row = self.candidate()
            row.update(reward_version=3, reward=result["reward"])
            row["components"].update(result, quality=.9, latency_seconds=1,
                                     repetition_penalty=.1, group_finalized=True)
            path = root / "rewards-rank0-pid1.jsonl"
            self.write_rows(path, [row])
            report, _, _ = audit.audit_rewards(path, 1, loaded, 2)
            self.assertEqual(report["formula_recheck"]["checked"], 1)
            self.assertEqual(report["formula_recheck"]["mismatches"], 0)
            self.assertIn("logged C", report["formula_recheck"]["note"])

    def test_end_to_end_no_overwrite_and_pid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"run"
            log = root/"v0"/"logging.jsonl"
            self.write_rows(log, [{"iteration": "1/4100", "learning_rate": 5.5e-7}])
            self.write_rows(root/"rewards"/"rewards-rank0-pid1.jsonl", [self.candidate()])
            self.write_rows(root/"rewards"/"rewards-rank0-pid2.jsonl", [self.candidate()])
            data = Path(tmp)/"train.jsonl"
            self.write_rows(data, [{"flex_task": json.dumps({"id": "sample", "source_text": "Hello", "reference_text": "你好"})}])
            out = Path(tmp)/"audit"
            args = audit.parser().parse_args([str(root), "--native-log", str(log), "--output-dir", str(out)])
            with self.assertRaisesRegex(ValueError, "Multiple PID"):
                audit.analyze(args)
            args.pids, args.dataset = "1", str(data)
            with contextlib.redirect_stdout(io.StringIO()):
                audit.analyze(args)
            self.assertTrue((out/"summary.json").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                audit.analyze(args)
            self.assertTrue((root/"rewards"/"rewards-rank0-pid2.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
