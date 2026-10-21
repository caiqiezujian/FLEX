from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from flex_gspo.artifacts import (
    ArtifactError, discover_trajectory_files, file_digest, inspect_sampler_runs,
    iter_validated_groups, load_training_tasks, normalize_task, strict_json_loads,
    validate_sampling_group,
)


def make_task(task_id="one"):
    return {
        "schema_version": 1, "id": task_id, "key": task_id,
        "audio_path": f"/server/audio/{task_id}.wav", "source_lang": "en", "target_lang": "zh",
        "duration_seconds": 2.1, "sample_rate": 16000, "chunks": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.1, "input_seconds": 1.0},
        ],
        "source_text": "This is a short test.", "reference_text": "这是一个简短测试。",
        "quality_ready": True, "source_segments": None, "split": "train",
        "transcript_provenance": {"kind": "provided", "is_gold": False, "source_field": "en.txt"},
        "reference_provenance": {"kind": "machine_translation", "is_gold": False},
        "semantic_verified": False,
    }


def make_candidate(task, index):
    replies = ["<|WAIT|>", "<|FLEX|>这是", "<|DIRECT|>测试。"]
    if index == 1:
        replies = ["<|DIRECT|>这是", "<|WAIT|>", "<|DIRECT|>一个测试。"]
    turns = []
    for i, (chunk, reply) in enumerate(zip(task["chunks"], replies)):
        tag, delta = reply[2:reply.index("|>")], reply[reply.index("|>") + 2:]
        turns.append({"turn_id": i, "audio_start_seconds": chunk["start_seconds"],
                      "audio_end_seconds": chunk["end_seconds"], "input_seconds": chunk["input_seconds"],
                      "response": reply, "tag": tag, "delta": delta, "committed": True,
                      "token_ids": [3, 4], "raw_token_ids": [3, 4, 5],
                      "token_logprobs": None, "finish_reason": "stop"})
    seed_text = f"42\0{task['id']}\0{index}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], "big")
    return {"id": task["id"], "group_index": index, "seed": seed,
            "turns": turns, "translation": "".join(t["delta"] for t in turns),
            "valid": True, "error": None, "complete_input": True, "final_wait": False,
            "mock": False, "offline_gspo_replay_ready": False,
            "timing_basis": "decoded_source_audio_not_live_wall_clock"}


def write_fixture(root, tasks=None, *, shards=1):
    tasks = tasks or [make_task()]
    manifest = root / "train.tasks.jsonl"
    manifest.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    paths = []
    for shard in range(shards):
        directory = root / f"worker_{shard}"
        directory.mkdir()
        local_tasks = tasks[shard::shards]
        signature = {
            "format_version": 1, "purpose": "sft_rollout_audit_and_reward_calibration",
            "offline_gspo_replay_ready": False, "manifest": str(manifest),
            "manifest_sha256": file_digest(manifest), "backend": "vllm",
            "model_options": {"temperature": 0.8, "model": "/server/checkpoint"},
            "checkpoint_identity": {"path": "/server/checkpoint"},
            "group_size": 2, "seed": 42, "limit": None,
            "shard_index": shard, "num_shards": shards,
            "audio_inventory": [{"id": t["id"], "path": t["audio_path"],
                                 "bytes": 999, "mtime_ns": 1} for t in local_tasks],
        }
        run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
        (directory / "run.json").write_text(json.dumps({"run_id": run_id, "signature": signature}), encoding="utf-8")
        groups = [{"schema_version": 1, "run_id": run_id, "id": task["id"], "task": task,
                   "trajectories": [make_candidate(task, 0), make_candidate(task, 1)],
                   "purpose": "sft_rollout_audit_and_reward_calibration",
                   "offline_gspo_replay_ready": False} for task in local_tasks]
        output = directory / "trajectories.jsonl"
        output.write_text("".join(json.dumps(group) + "\n" for group in groups), encoding="utf-8")
        paths.append(output.resolve())
    return manifest, paths


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.paths = write_fixture(self.root)
        self.tasks = load_training_tasks(self.manifest)
        self.runs, _ = inspect_sampler_runs(self.paths, file_digest(self.manifest))
        self.record = json.loads(self.paths[0].read_text())

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, record):
        validate_sampling_group(record, self.tasks[0], self.runs[self.paths[0]])

    def test_compatible_schema_preserves_provenance(self):
        task = normalize_task(make_task())
        self.assertEqual(task["reference_provenance"]["kind"], "machine_translation")
        self.assertIsNone(task["source_segments"])
        self.assertFalse(task["semantic_verified"])
        self.validate(self.record)

    def test_remote_audio_paths_need_not_exist_for_text_scoring(self):
        self.assertEqual(len(self.tasks), 1)
        self.assertEqual(self.tasks[0]["audio_path"], "/server/audio/one.wav")

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for value in ('{"id":"a","id":"b"}', '{"number":NaN}', '{"number":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ArtifactError):
                strict_json_loads(value)

    def test_duplicate_tasks_rejected(self):
        self.manifest.write_text(json.dumps(make_task()) + "\n" + json.dumps(make_task()) + "\n")
        with self.assertRaisesRegex(ArtifactError, "duplicate task"):
            load_training_tasks(self.manifest)

    def test_raw_metadata_rejected(self):
        with self.assertRaisesRegex(ArtifactError, "prepare"):
            normalize_task({"key": "one", "en": {"txt": "hi"}, "zh": {"txt": "你好"}})

    def test_bad_fields_rejected(self):
        changes = [
            lambda r: r["trajectories"].pop(),
            lambda r: r["trajectories"][0].update(translation="forged"),
            lambda r: r["trajectories"][0]["turns"][0].update(audio_end_seconds=1.2),
            lambda r: r["trajectories"][0]["turns"][1].update(delta="forged"),
            lambda r: r["trajectories"][0].update(mock=True),
            lambda r: r["trajectories"][0].update(group_index=True),
            lambda r: r["trajectories"][0].update(seed=17),
            lambda r: r["trajectories"][0].update(complete_input=False),
            lambda r: r["trajectories"][0].update(final_wait=True),
            lambda r: r["task"].update(source_text="changed"),
        ]
        for change in changes:
            record = deepcopy(self.record)
            change(record)
            with self.subTest(change=change), self.assertRaises(ArtifactError):
                self.validate(record)

    def test_real_invalid_policy_candidate_is_retained(self):
        candidate = self.record["trajectories"][0]
        candidate.update(valid=False, complete_input=False, final_wait=True, translation="", error="invalid tag")
        candidate["turns"] = candidate["turns"][:2]
        candidate["turns"][-1].update(response="nonsense", committed=False)
        candidate["turns"][-1].pop("tag")
        candidate["turns"][-1].pop("delta")
        self.validate(self.record)

    def test_one_sample_decoder_rounding_allowed(self):
        candidate = self.record["trajectories"][0]
        candidate["turns"][-1]["audio_end_seconds"] += 1 / 16000
        self.validate(self.record)

    def test_directory_discovery_only_exact_export_name(self):
        (self.root / "unrelated.jsonl").write_text("not JSON")
        self.assertEqual(discover_trajectory_files([self.root]), self.paths)
        with self.assertRaisesRegex(ArtifactError, "more than once"):
            discover_trajectory_files([self.root, self.paths[0]])

    def test_duplicate_groups_rejected(self):
        original = self.paths[0].read_text()
        self.paths[0].write_text(original + original)
        with self.assertRaisesRegex(ArtifactError, "duplicate trajectory"):
            list(iter_validated_groups(self.paths, {"one": self.tasks[0]}, self.runs))

    def test_manifest_content_mismatch_rejected(self):
        with self.assertRaisesRegex(ArtifactError, "manifest content differs"):
            inspect_sampler_runs(self.paths, "changed")

    def test_multiple_shards_allow_only_shard_specific_signature_differences(self):
        other = self.root / "parallel"
        other.mkdir()
        manifest, paths = write_fixture(other, [make_task("a"), make_task("b")], shards=2)
        runs, common = inspect_sampler_runs(paths, file_digest(manifest))
        self.assertTrue(common)
        self.assertEqual(len(runs), 2)
        run_path = paths[1].parent / "run.json"
        run = json.loads(run_path.read_text())
        run["signature"]["model_options"]["temperature"] = 0.9
        run["run_id"] = hashlib.sha256(json.dumps(run["signature"], sort_keys=True).encode()).hexdigest()
        run_path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ArtifactError, "Mixed sampler"):
            inspect_sampler_runs(paths, file_digest(manifest))

    def test_worker_specific_continuation_paths_are_provenance_not_config_mix(self):
        other = self.root / "continuation"
        other.mkdir()
        manifest, paths = write_fixture(other, [make_task("a"), make_task("b")], shards=2)
        for index, path in enumerate(paths):
            run_path = path.parent / "run.json"
            run = json.loads(run_path.read_text())
            run["signature"]["continue_from"] = f"/previous/run/worker_{index}"
            run["run_id"] = hashlib.sha256(json.dumps(run["signature"], sort_keys=True).encode()).hexdigest()
            run_path.write_text(json.dumps(run))
        runs, common = inspect_sampler_runs(paths, file_digest(manifest))
        self.assertTrue(common)
        self.assertEqual(runs[paths[1]]["signature"]["continue_from"], "/previous/run/worker_1")
        # Integrity is still checked with the COMPLETE, unmodified signature.
        run_path = paths[0].parent / "run.json"
        run = json.loads(run_path.read_text())
        run["signature"]["continue_from"] = "/forged/path"
        run_path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ArtifactError, "run_id does not match"):
            inspect_sampler_runs(paths, file_digest(manifest))


if __name__ == "__main__":
    unittest.main()
