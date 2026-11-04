import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout
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

    def test_four_and_eight_gpu_commands_keep_model_sharding_and_reward(self):
        shared = ("--micro-batch-size", "2", "--vllm-memory", "0.5", "--vllm-max-num-seqs", "32",
                  "--reward-config", str(ROOT / "configs" / "qalco_v4.json"))
        before = launcher.build_command(self.args("--gpu-ids", "0,1,2,3", "--global-batch-size", "32", *shared))
        after_args = self.args("--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "64", *shared)
        after = launcher.build_command(after_args)
        self.assertEqual(len(before), len(after))
        changed = {before[index - 1] for index, (old, new) in enumerate(zip(before, after)) if old != new}
        self.assertEqual(changed, {"--nproc_per_node", "--global_batch_size"})
        for flag, value in (("--nproc_per_node", "8"), ("--tensor_model_parallel_size", "2"),
                            ("--expert_model_parallel_size", "2"), ("--vllm_tensor_parallel_size", "4"),
                            ("--num_generations", "8"), ("--scale_rewards", "none"),
                            ("--lr", "1e-06"), ("--beta", "0.01")):
            self.assertEqual(after[after.index(flag) + 1], value)
        self.assertEqual(launcher.build_environment(after_args)["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")

    def test_eight_gpu_batch_alignment_and_invalid_device_lists(self):
        for devices in ("0,1,2,3,4,5", "0,1,2,3,4,5,6,6", "0,1,2,3,", "0,1,,3"):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                launcher.build_command(self.args("--gpu-ids", devices, "--global-batch-size", "64"))
        with self.assertRaisesRegex(ValueError, r"DP\(4\).+64"):
            launcher.build_command(self.args("--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "32",
                                             "--micro-batch-size", "2"))
        # Keeping B32 is possible on eight GPUs when each DP replica gets micro1.
        launcher.build_command(self.args("--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "32",
                                         "--micro-batch-size", "1"))

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

    def test_mm_cache_bypass_default_and_override_do_not_change_other_flags(self):
        baseline = launcher.build_command(self.args())
        override = launcher.build_command(self.args("--vllm-mm-processor-cache-gb", "1.5"))
        cache_index = baseline.index("--vllm_mm_processor_cache_gb") + 1
        self.assertEqual(baseline[cache_index], "0.0")
        self.assertEqual(override[cache_index], "1.5")
        self.assertEqual([i for i, (a, b) in enumerate(zip(baseline, override)) if a != b], [cache_index])
        self.assertEqual(len(baseline), len(override))
        for flag, value in (("--vllm_enable_prefix_caching", "true"), ("--sleep_level", "2"),
                            ("--offload_model", "true"), ("--offload_optimizer", "true")):
            self.assertEqual(baseline[baseline.index(flag) + 1], value)
        self.assertEqual(launcher.build_command(self.args("--vllm-mm-processor-cache-gb", "0")), baseline)

    def test_mm_cache_rejects_negative_and_nonfinite_values(self):
        for value in ("-0.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                launcher.build_command(self.args("--vllm-mm-processor-cache-gb=" + value))

    def test_mm_cache_positive_override_warns_even_in_dry_run(self):
        for value, warning_expected in (("0", False), ("1", True)):
            with self.subTest(value=value), redirect_stdout(StringIO()), redirect_stderr(StringIO()) as err:
                code = launcher.main(["/missing/model", "/missing/data", "/missing/output", "--dry-run",
                                      "--vllm-mm-processor-cache-gb", value])
            self.assertEqual(code, 0)
            self.assertEqual("cache-clear hang" in err.getvalue(), warning_expected)

    def test_environment_keeps_reward_external(self):
        env = launcher.build_environment(self.args())
        self.assertEqual(env["ENABLE_AUDIO_OUTPUT"], "0")
        self.assertIn("FLEX_REWARD_CONFIG", env)
        self.assertIn("FLEX_REWARD_LOG_DIR", env)
        self.assertEqual(env["FLEX_ROLLOUT_LOG_EVERY"], "5")

    def test_v3_config_does_not_change_accepted_engine_arguments(self):
        baseline = self.args("--global-batch-size", "32", "--micro-batch-size", "2",
                             "--vllm-memory", "0.5", "--vllm-max-num-seqs", "32")
        experiment = self.args("--global-batch-size", "32", "--micro-batch-size", "2",
                               "--vllm-memory", "0.5", "--vllm-max-num-seqs", "32",
                               "--reward-config", str(ROOT / "configs" / "qalco_v3.json"))
        self.assertEqual(launcher.build_command(baseline), launcher.build_command(experiment))
        self.assertNotEqual(launcher.build_environment(baseline)["FLEX_REWARD_CONFIG"],
                            launcher.build_environment(experiment)["FLEX_REWARD_CONFIG"])
        command = launcher.build_command(experiment)
        self.assertEqual(command[command.index("--train_iters") + 1], "3000")
        self.assertEqual(command[command.index("--save_steps") + 1], "500")

    def test_v3_shell_explicitly_selects_new_config_and_allows_overrides(self):
        script = (ROOT / "05_train_qalco_v3.sh").read_text(encoding="utf-8")
        tokens = shlex.split(script[script.index("exec bash "):].replace("\\\n", " "))
        self.assertEqual(tokens[2], "${SCRIPT_DIR}/04_train_throughput.sh")
        self.assertEqual(tokens[-1], "$@")
        self.assertEqual(tokens[3:5], ["--reward-config", "${SCRIPT_DIR}/configs/qalco_v3.json"])
        with redirect_stdout(StringIO()) as out:
            code = launcher.main(["/missing/model", "/missing/train.jsonl", "/missing/new-run",
                                  "--reward-config", str(ROOT / "configs" / "qalco_v3.json"), "--dry-run"])
        self.assertEqual(code, 0)
        plan = json.JSONDecoder().raw_decode(out.getvalue())[0]
        self.assertEqual(plan["reward_settings"]["version"], 3)
        self.assertEqual(plan["reward_settings"]["v3"]["brevity_weight"], .08)

    def test_v4_changes_only_reward_scaling_in_accepted_engine(self):
        shared = ("--global-batch-size", "32", "--micro-batch-size", "2",
                  "--vllm-memory", "0.5", "--vllm-max-num-seqs", "32")
        before = launcher.build_command(self.args(*shared, "--reward-config", str(ROOT / "configs" / "qalco_v3.json")))
        after = launcher.build_command(self.args(*shared, "--reward-config", str(ROOT / "configs" / "qalco_v4.json")))
        def flags(command):
            start = command.index("--rlhf_type")
            return dict(zip(command[start::2], command[start + 1::2]))
        old_flags, new_flags = flags(before), flags(after)
        self.assertEqual(old_flags["--scale_rewards"], "group")
        self.assertEqual(new_flags["--scale_rewards"], "none")
        self.assertEqual({key for key in old_flags if old_flags[key] != new_flags[key]}, {"--scale_rewards"})
        self.assertEqual(set(new_flags) - set(old_flags), {"--advantage_estimator", "--kl_in_reward"})
        self.assertEqual(new_flags["--advantage_estimator"], "grpo")
        self.assertEqual(new_flags["--kl_in_reward"], "false")
        for flag, expected in (("--importance_sampling_level", "sequence"), ("--epsilon", "0.0003"),
                               ("--epsilon_high", "0.0004"), ("--num_iterations", "1"),
                               ("--beta", "0.01"), ("--lr", "1e-06"),
                               ("--vllm_mm_processor_cache_gb", "0.0"),
                               ("--log_completions", "false"), ("--max_turns", "65")):
            self.assertEqual(after[after.index(flag) + 1], expected)

    def test_v4_wrapper_chain_preserves_capacity_and_smoke_overrides(self):
        options = []
        for name, target in (("03_train_3000.sh", "02_train.sh"),
                             ("04_train_throughput.sh", "03_train_3000.sh"),
                             ("06_train_qalco_v4.sh", "04_train_throughput.sh")):
            script = (ROOT / name).read_text(encoding="utf-8")
            tokens = shlex.split(script[script.index("exec bash "):].replace("\\\n", " "))
            self.assertEqual(tokens[2], "${SCRIPT_DIR}/" + target)
            self.assertEqual(tokens[-1], "$@")
            options.extend(value.replace("${SCRIPT_DIR}", str(ROOT)) for value in tokens[3:-1])
        parsed = self.args(*options)
        self.assertEqual((parsed.group_size, parsed.global_batch_size, parsed.micro_batch_size,
                          parsed.vllm_max_num_seqs, parsed.vllm_memory, parsed.train_iters,
                          parsed.save_steps, parsed.max_turns), (8, 32, 2, 32, .5, 3000, 500, 65))
        self.assertEqual(parsed.vllm_mm_processor_cache_gb, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-smoke"
            with redirect_stdout(StringIO()) as stdout:
                code = launcher.main(["/missing/model", "/missing/train.jsonl", str(output), *options,
                                      "--train-iters", "2", "--save-steps", "2", "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
        self.assertEqual(plan["reward_settings"]["version"], 4)
        self.assertEqual(plan["reward_normalization"]["upstream_scale_rewards"], "none")
        self.assertTrue(plan["reward_normalization"]["upstream_group_mean_baseline"])
        self.assertEqual(plan["sampling"]["fresh_prompts_per_generation"], 4)
        self.assertEqual(plan["argv"][plan["argv"].index("--train_iters") + 1], "2")

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as stdout:
            output = Path(directory) / "eight-gpu-dry-run"
            with patch.object(launcher.subprocess, "run") as run:
                code = launcher.main(["/missing/model", "/missing/train.jsonl", str(output), *options,
                                      "--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "64", "--dry-run"])
            run.assert_not_called()
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
            self.assertEqual(plan["parallelism"]["world_size"], 8)
            self.assertEqual(plan["parallelism"]["data_parallel_size"], 4)
            self.assertEqual(plan["parallelism"]["vllm_replicas"], 2)
            self.assertEqual(plan["sampling"]["fresh_prompts_per_generation"], 8)
            self.assertEqual(plan["reward_settings"]["version"], 4)

    def test_command_rejects_missing_or_invalid_reward_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for payload in (None, "{}", '{"reward":{"version":99}}'):
                if payload is not None:
                    path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                    launcher.build_command(self.args("--reward-config", str(path)))

    def test_rollout_logging_is_local_not_an_unsupported_swift_flag(self):
        a = self.args("--rollout-log-every", "1", "--global-batch-size", "32")
        cmd = launcher.build_command(a)
        self.assertEqual(launcher.build_environment(a)["FLEX_ROLLOUT_LOG_EVERY"], "1")
        self.assertNotIn("--rollout_log_every", cmd)
        self.assertEqual(cmd[cmd.index("--tensor_model_parallel_size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--vllm_tensor_parallel_size") + 1], "4")
        self.assertEqual(cmd[cmd.index("--log_completions") + 1], "false")
        with self.assertRaisesRegex(ValueError, "rollout-log-every"):
            launcher.build_command(self.args("--rollout-log-every", "0"))

    def test_offload_debug_is_opt_in_and_keeps_upstream_training_command(self):
        normal = self.args("--global-batch-size", "64", "--gpu-ids", "0,1,2,3,4,5,6,7")
        debug = self.args("--global-batch-size", "64", "--gpu-ids", "0,1,2,3,4,5,6,7", "--debug-offload")
        self.assertEqual(launcher.build_command(normal), launcher.build_command(debug))
        with patch.dict(launcher.os.environ, {"FLEX_DEBUG_OFFLOAD": "1", "CUDA_LAUNCH_BLOCKING": "0"}):
            self.assertNotIn("FLEX_DEBUG_OFFLOAD", launcher.build_environment(normal))
            self.assertEqual(launcher.build_environment(normal)["CUDA_LAUNCH_BLOCKING"], "0")
            env = launcher.build_environment(debug)
        self.assertEqual(env["FLEX_DEBUG_OFFLOAD"], "1")
        self.assertEqual(env["CUDA_LAUNCH_BLOCKING"], "1")
        self.assertEqual(env["TORCH_SHOW_CPP_STACKTRACES"], "0")
        self.assertEqual(env["TORCH_DISABLE_ADDR2LINE"], "1")

    def test_formal_3000_recipe_keeps_accepted_batch_and_disables_audio_log_reencode(self):
        script = (ROOT / "03_train_3000.sh").read_text(encoding="utf-8")
        command = script[script.index("exec bash "):].replace("\\\n", " ")
        tokens = shlex.split(command)
        self.assertEqual(tokens[:2], ["exec", "bash"])
        self.assertEqual(tokens[2], "${SCRIPT_DIR}/02_train.sh")
        self.assertEqual(tokens[-1], "$@")
        a = self.args(*tokens[3:-1])
        cmd = launcher.build_command(a)
        for flag, expected in (("--train_iters", "3000"), ("--num_generations", "8"),
                               ("--global_batch_size", "32"), ("--micro_batch_size", "1"),
                               ("--steps_per_generation", "1"), ("--save_steps", "500"),
                               ("--save_total_limit", "3"), ("--log_completions", "false"),
                               ("--offload_model", "true"), ("--offload_optimizer", "true")):
            self.assertEqual(cmd[cmd.index(flag) + 1], expected)
        self.assertEqual(a.global_batch_size * a.steps_per_generation // a.group_size, 4)
        self.assertEqual(a.rollout_log_every, 5)
        self.assertIn("unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_DISTRIBUTED_DEBUG", script)
        self.assertNotIn("--resume-from", script)

    def test_dry_run_no_files_or_framework_import(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nonexistent"
            with redirect_stdout(StringIO()):
                code = launcher.main(["/missing/model", "/missing/data", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_all_supported_versions_launch_without_scoring_preflight(self):
        for config in ("default.json", "qalco_v3.json", "qalco_v4.json"):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as temp, \
                    redirect_stdout(StringIO()) as stdout, \
                    patch.object(launcher, "validate_local_inputs", return_value={}), \
                    patch.object(launcher, "run_identity", return_value={}), \
                    patch.object(launcher, "verify_resume_identity"), \
                    patch.object(launcher.subprocess, "run") as run, \
                    patch("flex_gspo.comet22.ServiceCOMET22.score_batch",
                          side_effect=AssertionError("Startup must not score COMET22 samples")), \
                    patch("flex_gspo.metricx.ServiceMetricX.score_batch",
                          side_effect=AssertionError("Startup must not score MetricX samples")):
                run.return_value.returncode = 0
                self.assertEqual(launcher.main(["/missing/model", "/missing/data", temp,
                                                "--reward-config", str(ROOT / "configs" / config)]), 0)
                self.assertEqual(run.call_count, 2)
                environment_check, training = run.call_args_list
                self.assertEqual(environment_check.args[0][:2],
                                 [launcher.sys.executable, str(ROOT / "Scripts" / "check_environment.py")])
                self.assertTrue(environment_check.kwargs["check"])
                self.assertEqual(training.args[0][:3],
                                 [launcher.sys.executable, "-m", "torch.distributed.run"])
                self.assertFalse(training.kwargs["check"])
                plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
                self.assertEqual(plan["quality_validation"],
                                 {"mode": "manual", "automatic_service_probe": False})
                self.assertEqual(plan["reward_settings"]["quality_provider"], "comet22")
                self.assertNotIn("quality_preflight_cases", plan)
                self.assertEqual(len(list(Path(temp).glob("launch_*.json"))), 1)

    def test_framework_failure_still_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), redirect_stderr(StringIO()), \
                patch.object(launcher, "validate_local_inputs", return_value={}), \
                patch.object(launcher, "run_identity", return_value={}), \
                patch.object(launcher, "verify_resume_identity"), \
                patch.object(launcher.subprocess, "run") as run:
            run.side_effect = launcher.subprocess.CalledProcessError(2, "framework check")
            status = launcher.main(["/missing/model", "/missing/data", temp,
                                    "--reward-config", str(ROOT / "configs" / "qalco_v4.json")])
            self.assertEqual(status, 2)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[:2],
                             [launcher.sys.executable, str(ROOT / "Scripts" / "check_environment.py")])
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertFalse(list(Path(temp).glob("launch_*.json")))

    def test_eight_gpu_launch_preflight_checks_all_selected_gpus(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), \
                patch.object(launcher, "validate_local_inputs", return_value={}), \
                patch.object(launcher, "run_identity", return_value={}), \
                patch.object(launcher, "verify_resume_identity"), \
                patch.object(launcher.subprocess, "run") as run:
            run.return_value.returncode = 0
            status = launcher.main(["/missing/model", "/missing/data", temp,
                                    "--reward-config", str(ROOT / "configs" / "qalco_v4.json"),
                                    "--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "64",
                                    "--micro-batch-size", "2"])
            self.assertEqual(status, 0)
            environment_check, training = run.call_args_list
            check_command, train_command = environment_check.args[0], training.args[0]
            self.assertEqual(check_command[check_command.index("--required-gpus") + 1], "8")
            self.assertEqual(train_command[train_command.index("--nproc_per_node") + 1], "8")
            self.assertEqual(training.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")

    def test_v4_identity_does_not_require_or_hash_manual_quality_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (root / "configs").mkdir()
            (root / "flex_gspo").mkdir()
            for name in ("swift_plugin.py", "train_megatron.py"):
                (root / name).write_text("# identity fixture\n", encoding="utf-8")
            cases = root / "configs" / "quality_regressions_v4.jsonl"
            args = self.args("--reward-config", str(ROOT / "configs" / "qalco_v4.json"))
            args.model = str(model)
            with patch.object(launcher, "ROOT", root):
                # Missing or edited manual diagnostics are not training inputs.
                first = launcher.run_identity(args, {"sha256": "data", "tasks": 4})
                cases.write_text("manually edited diagnostic cases\n", encoding="utf-8")
                second = launcher.run_identity(args, {"sha256": "data", "tasks": 4})
            self.assertEqual(first, second)
            self.assertNotIn("quality_regression_cases_sha256", first)

    def test_throughput_recipe_is_opt_in_and_preserves_objective(self):
        def options(name, target):
            script = (ROOT / name).read_text(encoding="utf-8")
            tokens = shlex.split(script[script.index("exec bash "):].replace("\\\n", " "))
            self.assertEqual(tokens[2], "${SCRIPT_DIR}/" + target)
            self.assertEqual(tokens[-1], "$@")
            return tokens[3:-1]
        stable = options("03_train_3000.sh", "02_train.sh")
        fast = options("04_train_throughput.sh", "03_train_3000.sh")
        a = self.args(*stable, *fast)
        command = launcher.build_command(a)
        baseline = launcher.build_command(self.args(*stable))
        # Ignore only the three explicitly changed runtime capacity flags.
        def flags(cmd):
            start = cmd.index("--rlhf_type")
            return dict(zip(cmd[start::2], cmd[start + 1::2]))
        before, after = flags(baseline), flags(command)
        self.assertEqual({k for k in before if before[k] != after[k]},
                         {"--micro_batch_size", "--vllm_max_num_seqs", "--vllm_gpu_memory_utilization"})
        self.assertEqual((a.group_size, a.global_batch_size, a.micro_batch_size,
                          a.vllm_max_num_seqs, a.vllm_memory, a.train_iters), (8, 32, 2, 32, 0.5, 3000))
        self.assertEqual(a.performance_log, "on")
        self.assertEqual(launcher.build_environment(a)["FLEX_PERF_LOG_DIR"],
                         str(Path(a.output).resolve() / "performance"))
        # Trailing arguments still allow a 5-step isolated smoke or rollback.
        overridden = self.args(*stable, *fast, "--train-iters", "5", "--micro-batch-size", "1")
        self.assertEqual((overridden.train_iters, overridden.micro_batch_size), (5, 1))

    def test_selective_recompute_omits_incompatible_full_options(self):
        full = launcher.build_command(self.args())
        selective = launcher.build_command(self.args("--recompute-granularity", "selective"))
        self.assertEqual(full[full.index("--recompute_granularity") + 1], "full")
        self.assertEqual(full[full.index("--recompute_method") + 1], "uniform")
        self.assertEqual(selective[selective.index("--recompute_granularity") + 1], "selective")
        self.assertNotIn("--recompute_method", selective)
        self.assertNotIn("--recompute_num_layers", selective)

    def test_micro_two_alignment_and_explicit_disabled_observation(self):
        for batch in (32, 64):
            launcher.build_command(self.args("--global-batch-size", str(batch), "--micro-batch-size", "2"))
        with self.assertRaisesRegex(ValueError, "Prompt count"):
            launcher.build_command(self.args("--global-batch-size", "48", "--micro-batch-size", "2"))
        with patch.dict("os.environ", {"FLEX_PERF_LOG_DIR": "/old/experiment"}):
            self.assertNotIn("FLEX_PERF_LOG_DIR", launcher.build_environment(self.args()))
            self.assertNotIn("FLEX_PERF_LOG_DIR", launcher.build_environment(self.args("--performance-log", "off")))

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

    def test_old_prepared_prompt_has_actionable_rebuild_error(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            a = self.prepared_fixture(Path(temp))
            path = Path(a.dataset)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["messages"][0]["content"] = "obsolete SFT instruction"
            content = "\n".join(json.dumps(row) for row in rows) + "\n"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Re-run 01_prepare.sh"):
                launcher.validate_local_inputs(a)
            self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_segmented_prepare_is_accepted_by_current_v4_launcher(self):
        from flex_gspo.prepare import PrepareOptions, prepare_dataset
        from test_prepare import write_pcm

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            model = root / "short-prompt-sft"
            model.mkdir()
            rows = []
            for index in range(4):
                audio = root / f"clip-{index}.wav"
                write_pcm(audio)
                rows.append({"audio": str(audio), "srcs": ["The meeting", "starts at nine."],
                             "chooses": ["会议", "九点开始。"], "voice_end": False})
            raw = root / "raw.jsonl"
            raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "prepared"
            report = prepare_dataset(PrepareOptions(input_jsonl=str(raw), output_dir=str(output),
                                                     input_format="segments", workers=2))
            self.assertTrue(report["manifest_published"])
            args = launcher.parser().parse_args([
                str(model), str(output / "train.jsonl"), str(root / "new-run"),
                "--reward-config", str(ROOT / "configs" / "qalco_v4.json"),
                "--global-batch-size", "32", "--group-size", "8", "--micro-batch-size", "2",
            ])
            info = launcher.validate_local_inputs(args)
            self.assertEqual(info["tasks"], 4)
            command = launcher.build_command(args)
            self.assertEqual(command[command.index("--multi_turn_scheduler") + 1], "flex_stream")
            self.assertEqual(command[command.index("--scale_rewards") + 1], "none")
            reward = json.loads(Path(args.reward_config).read_text(encoding="utf-8"))["reward"]
            self.assertEqual(reward["quality_provider"], "comet22")
            self.assertEqual(reward["v4"]["latency_weight"], 0.25)

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

    def test_resume_identity_tracks_gpu_count_but_not_physical_device_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args("--global-batch-size", "64", "--micro-batch-size", "2")
            args.model, args.output = str(root / "model"), str(root / "run")
            Path(args.model).mkdir()
            Path(args.output).mkdir()
            manifest = {"sha256": "same-data", "tasks": 8}
            args.gpu_ids = "0,1,2,3"
            four = launcher.run_identity(args, manifest)
            args.gpu_ids = "4,5,6,7"
            self.assertEqual(four, launcher.run_identity(args, manifest))
            args.gpu_ids = "0,1,2,3,4,5,6,7"
            eight = launcher.run_identity(args, manifest)
            self.assertNotEqual(four, eight)
            self.assertEqual(eight["parallelism"]["data_parallel_size"], 4)
            (Path(args.output) / "launch_original.json").write_text(json.dumps({"run_identity": four}), encoding="utf-8")
            args.resume_from = str(Path(args.output) / "checkpoint-500")
            with self.assertRaisesRegex(ValueError, "Resume rejected.+parallelism"):
                launcher.verify_resume_identity(args, eight)

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
