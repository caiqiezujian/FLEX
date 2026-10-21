import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_megatron_under_test", ROOT / "train_megatron.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TestMegatronLauncher(unittest.TestCase):
    def args(self, *extra):
        return launcher.parser().parse_args(["/srv/model", "/srv/train.jsonl", "/srv/new-run", *extra])

    def test_defaults_are_online_sequence_gspo(self):
        cmd = launcher.build_command(self.args())
        self.assertIn("swift.cli._megatron.rlhf", cmd)
        for flag, value in (("--importance_sampling_level", "sequence"), ("--multi_turn_scheduler", "flex_stream"),
                            ("--reward_funcs", "qalco_online"), ("--num_generations", "8"),
                            ("--global_batch_size", "16"), ("--vllm_mode", "colocate"),
                            ("--remove_unused_columns", "false"), ("--no_save_optim", "false")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn("--new_special_tokens", cmd)
        self.assertNotIn("--adapters", cmd)

    def test_smoke_batch_alignment(self):
        launcher.build_command(self.args("--group-size", "2", "--global-batch-size", "4", "--train-iters", "2"))

    def test_invalid_batch_and_gpus_fail(self):
        for extra in (("--global-batch-size", "8"), ("--gpu-ids", "0,1"), ("--group-size", "1"),
                      ("--gpu-ids", "0,1,2,2"), ("--temperature", "0"), ("--lr", "nan"), ("--kl-beta", "inf")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launcher.build_command(self.args(*extra))

    def test_resume_is_native_and_loads_optimizer(self):
        cmd = launcher.build_command(self.args("--resume-from", "/srv/native-adapter"))
        self.assertEqual(cmd[cmd.index("--mcore_adapter") + 1], "/srv/native-adapter")
        self.assertEqual(cmd[cmd.index("--finetune") + 1], "false")
        self.assertEqual(cmd[cmd.index("--no_load_optim") + 1], "false")

    def test_paths_with_spaces_are_single_arguments(self):
        a = self.args()
        a.model = "/srv/a model; echo nope"
        cmd = launcher.build_command(a)
        self.assertEqual(cmd[cmd.index("--model") + 1], a.model)

    def test_multimodal_limit_matches_turns(self):
        cmd = launcher.build_command(self.args("--max-turns", "42"))
        self.assertEqual(json.loads(cmd[cmd.index("--vllm_limit_mm_per_prompt") + 1]),
                         {"audio": {"count": 42, "length": 32000}, "image": 1, "video": 0})

    def test_environment_keeps_reward_external(self):
        env = launcher.build_environment(self.args())
        self.assertEqual(env["ENABLE_AUDIO_OUTPUT"], "0")
        self.assertIn("FLEX_REWARD_CONFIG", env)
        self.assertIn("FLEX_REWARD_LOG_DIR", env)
        self.assertEqual(env["FLEX_ROLLOUT_LOG_EVERY"], "5")

    def test_rollout_logging_is_local_not_an_unsupported_swift_flag(self):
        a = self.args("--rollout-log-every", "1", "--global-batch-size", "32")
        cmd = launcher.build_command(a)
        self.assertEqual(launcher.build_environment(a)["FLEX_ROLLOUT_LOG_EVERY"], "1")
        self.assertNotIn("--rollout_log_every", cmd)
        self.assertEqual(cmd[cmd.index("--tensor_model_parallel_size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--vllm_tensor_parallel_size") + 1], "4")
        self.assertEqual(cmd[cmd.index("--log_completions") + 1], "true")
        with self.assertRaisesRegex(ValueError, "rollout-log-every"):
            launcher.build_command(self.args("--rollout-log-every", "0"))

    def test_dry_run_no_files_or_framework_import(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nonexistent"
            with redirect_stdout(StringIO()):
                code = launcher.main(["/missing/model", "/missing/data", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def prepared_fixture(self, root):
        from flex_gspo.audio import encode_chunk_ref
        from flex_gspo.prompts import initial_messages
        model = root / "model"
        model.mkdir()
        dataset, config = root / "train.jsonl", root / "reward.json"
        config.write_text("{}", encoding="utf-8")
        rows = []
        for index in range(2):
            task = {"schema_version": 1, "id": str(index), "audio_path": "/srv/audio.wav",
                    "source_lang": "en", "target_lang": "zh", "duration_seconds": 1.0,
                    "sample_rate": 16000, "quality_ready": True, "source_text": "Hello.",
                    "reference_text": "你好。", "chunks": [
                        {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0}]}
            rows.append({"messages": initial_messages(), "audios": [encode_chunk_ref(task, 0)],
                         "flex_task": json.dumps(task)})
        dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return launcher.parser().parse_args([str(model), str(dataset), str(root / "run"), "--reward-config", str(config)])

    def test_manifest_is_metadata_only_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            a = self.prepared_fixture(Path(temp))
            info = launcher.validate_local_inputs(a)
            self.assertEqual(info["tasks"], 2)
            self.assertEqual(info["sha256"], launcher.hash_file(Path(a.dataset)))
            # /srv/audio.wav does not exist: launcher validation must not decode it.

    def test_manifest_rejects_full_audio_or_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            a = self.prepared_fixture(Path(temp))
            path = Path(a.dataset)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["audios"] = ["/srv/audio.wav"]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "first virtual audio"):
                launcher.validate_local_inputs(a)

    def test_resume_identity_rejects_reward_or_data_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = self.prepared_fixture(root)
            Path(a.output).mkdir()
            a.resume_from = str(Path(a.output) / "checkpoint-1")
            original = {"dataset_sha256": "a", "reward_config_sha256": "b"}
            (Path(a.output) / "launch_original.json").write_text(json.dumps({"run_identity": original}), encoding="utf-8")
            launcher.verify_resume_identity(a, original)
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                launcher.verify_resume_identity(a, dict(original, dataset_sha256="changed"))

    def test_merged_export_metadata_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            a = self.prepared_fixture(root)
            resume = root / "checkpoint-1-merged"
            iteration = resume / "iter_0000001"
            iteration.mkdir(parents=True)
            (resume / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
            (iteration / "common.pt").write_bytes(b"metadata")
            (resume / "args.json").write_text(json.dumps({"no_save_optim": False, "no_save_rng": False}), encoding="utf-8")
            a.resume_from = str(resume)
            with self.assertRaisesRegex(ValueError, "native sharded"):
                launcher.validate_local_inputs(a)


if __name__ == "__main__":
    unittest.main()
