"""V5 launch plans retain the accepted engine recipe and never probe quality."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qalco_v5.json"
spec = importlib.util.spec_from_file_location("metricx_v5_launcher_under_test", ROOT / "train_megatron.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class MetricXV5LauncherTests(unittest.TestCase):
    def args(self, *extra):
        return launcher.parser().parse_args(["/srv/model", "/srv/train.jsonl", "/srv/new-run", *extra])

    def test_v5_preserves_v4_hardware_and_center_only_gspo_flags(self):
        shared = ("--global-batch-size", "32", "--micro-batch-size", "2",
                  "--vllm-memory", "0.5", "--vllm-max-num-seqs", "32")
        before = self.args(*shared, "--reward-config", str(ROOT / "configs" / "qalco_v4.json"))
        after = self.args(*shared, "--reward-config", str(CONFIG))
        self.assertEqual(launcher.build_command(before), launcher.build_command(after))
        self.assertNotEqual(launcher.build_environment(before)["FLEX_REWARD_CONFIG"],
                            launcher.build_environment(after)["FLEX_REWARD_CONFIG"])
        command = launcher.build_command(after)
        for flag, expected in (("--scale_rewards", "none"), ("--advantage_estimator", "grpo"),
                               ("--kl_in_reward", "false"), ("--importance_sampling_level", "sequence"),
                               ("--tensor_model_parallel_size", "2"), ("--vllm_tensor_parallel_size", "4"),
                               ("--num_generations", "8"), ("--global_batch_size", "32"),
                               ("--micro_batch_size", "2"), ("--vllm_max_num_seqs", "32"),
                               ("--vllm_gpu_memory_utilization", "0.5"),
                               ("--vllm_mm_processor_cache_gb", "0.0"),
                               ("--epsilon", "0.0003"), ("--epsilon_high", "0.0004"),
                               ("--num_iterations", "1"), ("--beta", "0.01"),
                               ("--lr", "1e-06"), ("--max_turns", "65")):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def wrapper_options(self):
        options = []
        for name, target in (("03_train_3000.sh", "02_train.sh"),
                             ("04_train_throughput.sh", "03_train_3000.sh"),
                             ("07_train_qalco_v5.sh", "04_train_throughput.sh")):
            script = (ROOT / name).read_text(encoding="utf-8")
            tokens = shlex.split(script[script.index("exec bash "):].replace("\\\n", " "))
            self.assertEqual(tokens[:2], ["exec", "bash"])
            self.assertEqual(tokens[2], "${SCRIPT_DIR}/" + target)
            self.assertEqual(tokens[-1], "$@")
            if name == "07_train_qalco_v5.sh":
                self.assertEqual(tokens[3:5], ["--reward-config", "${SCRIPT_DIR}/configs/qalco_v5.json"])
            options.extend(value.replace("${SCRIPT_DIR}", str(ROOT)) for value in tokens[3:-1])
        return options

    def test_v5_wrapper_dry_run_keeps_capacity_and_accepts_smoke_overrides(self):
        options = self.wrapper_options()
        parsed = self.args(*options)
        self.assertEqual((parsed.group_size, parsed.global_batch_size, parsed.micro_batch_size,
                          parsed.vllm_max_num_seqs, parsed.vllm_memory, parsed.train_iters,
                          parsed.save_steps, parsed.max_turns), (8, 32, 2, 32, .5, 3000, 500, 65))
        with tempfile.TemporaryDirectory() as directory, \
                patch("flex_gspo.metricx_service.requests.Session") as session, \
                patch.object(launcher.subprocess, "run") as run:
            output = Path(directory) / "new-smoke"
            with redirect_stdout(StringIO()) as stdout:
                status = launcher.main(["/missing/model", "/missing/data", str(output), *options,
                                        "--train-iters", "2", "--save-steps", "2", "--dry-run"])
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            session.assert_not_called()
            run.assert_not_called()
            plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
        self.assertEqual(plan["reward_settings"]["version"], 5)
        self.assertEqual(plan["reward_settings"]["quality_provider"], "metricx")
        self.assertEqual(plan["reward_settings"]["metricx"]["mode"], "reference")
        self.assertEqual(plan["reward_settings"]["v5"]["max_metricx_error"], 5.)
        self.assertEqual(plan["reward_normalization"]["upstream_scale_rewards"], "none")
        self.assertTrue(plan["reward_normalization"]["component_normalization"])
        self.assertTrue(plan["reward_normalization"]["upstream_group_mean_baseline"])
        self.assertEqual(plan["sampling"]["fresh_prompts_per_generation"], 4)
        for flag in ("--train_iters", "--save_steps"):
            self.assertEqual(plan["argv"][plan["argv"].index(flag) + 1], "2")

    def test_v5_launch_has_framework_check_but_no_automatic_scoring_preflight(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as stdout, \
                patch.object(launcher, "validate_local_inputs", return_value={}), \
                patch.object(launcher, "run_identity", return_value={}), \
                patch.object(launcher, "verify_resume_identity"), \
                patch.object(launcher.subprocess, "run") as run, \
                patch("flex_gspo.metricx_service.requests.Session") as session, \
                patch("flex_gspo.metricx.ServiceMetricX.score_batch",
                      side_effect=AssertionError("Startup must not score MetricX samples")), \
                patch("flex_gspo.comet22.ServiceCOMET22.score_batch",
                      side_effect=AssertionError("Startup must not score COMET22 samples")):
            run.return_value.returncode = 0
            status = launcher.main(["/missing/model", "/missing/data", directory,
                                    "--reward-config", str(CONFIG)])
            self.assertEqual(status, 0)
            self.assertEqual(run.call_count, 2)
            environment_check, training = run.call_args_list
            self.assertEqual(environment_check.args[0][:2],
                             [launcher.sys.executable, str(ROOT / "Scripts" / "check_environment.py")])
            self.assertTrue(environment_check.kwargs["check"])
            self.assertEqual(training.args[0][:3],
                             [launcher.sys.executable, "-m", "torch.distributed.run"])
            self.assertFalse(training.kwargs["check"])
            session.assert_not_called()
            plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
            self.assertEqual(plan["quality_validation"], {"mode": "manual", "automatic_service_probe": False})
            self.assertEqual(plan["reward_settings"]["quality_provider"], "metricx")
            self.assertNotIn("quality_preflight_cases", plan)
            self.assertIn("Training rewards use metricx", stdout.getvalue())
            log_files = list(Path(directory).glob("launch_*.json"))
            self.assertEqual(len(log_files), 1)
            logged = json.loads(log_files[0].read_text(encoding="utf-8"))
            self.assertEqual(logged["reward_settings"]["version"], 5)
            self.assertEqual(logged["reward_settings"]["metricx"]["mode"], "reference")
            self.assertEqual(training.kwargs["env"]["FLEX_RUN_ID"], logged["run_id"])

    def test_framework_failure_stops_v5_before_training_or_launch_log(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                redirect_stderr(StringIO()), \
                patch.object(launcher, "validate_local_inputs", return_value={}), \
                patch.object(launcher, "run_identity", return_value={}), \
                patch.object(launcher, "verify_resume_identity"), \
                patch.object(launcher.subprocess, "run") as run:
            run.side_effect = launcher.subprocess.CalledProcessError(2, "framework check")
            status = launcher.main(["/missing/model", "/missing/data", directory,
                                    "--reward-config", str(CONFIG)])
            self.assertEqual(status, 2)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertEqual(list(Path(directory).glob("launch_*.json")), [])

    def test_v5_wrapper_eight_gpu_override_preserves_sharding_and_reward(self):
        options = self.wrapper_options()
        with redirect_stdout(StringIO()) as stdout:
            status = launcher.main(["/missing/model", "/missing/data", "/missing/output",
                                    *options, "--gpu-ids", "0,1,2,3,4,5,6,7",
                                    "--global-batch-size", "64", "--dry-run"])
        self.assertEqual(status, 0)
        plan = json.JSONDecoder().raw_decode(stdout.getvalue())[0]
        self.assertEqual(plan["parallelism"]["world_size"], 8)
        self.assertEqual(plan["sampling"]["fresh_prompts_per_generation"], 8)
        self.assertEqual(plan["reward_settings"]["version"], 5)
        self.assertEqual(plan["reward_settings"]["v5"]["latency_weight"], .5)
        command = plan["argv"]
        for flag, value in (("--nproc_per_node", "8"), ("--global_batch_size", "64"),
                            ("--micro_batch_size", "2"), ("--num_generations", "8"),
                            ("--tensor_model_parallel_size", "2"), ("--expert_model_parallel_size", "2"),
                            ("--vllm_tensor_parallel_size", "4"), ("--scale_rewards", "none"),
                            ("--vllm_mm_processor_cache_gb", "0.0"), ("--offload_model", "true")):
            self.assertEqual(command[command.index(flag) + 1], value)
        args = self.args(*options, "--gpu-ids", "0,1,2,3,4,5,6,7", "--global-batch-size", "64")
        args.model, args.dataset, args.output = "/missing/model", "/missing/data", "/missing/output"
        args.reward_config = str(ROOT / "configs" / "qalco_v4.json")
        self.assertEqual(command, launcher.build_command(args))


if __name__ == "__main__":
    unittest.main()
