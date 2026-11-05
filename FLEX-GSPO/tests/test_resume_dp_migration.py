"""DP migration retains native training state and rejects semantic changes."""
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Scripts import resume_dp_migration as migration


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dp_migration_launcher_under_test", ROOT / "train_megatron.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

IGNORED_SETTINGS = {
    "output", "dataset", "reward_config", "resume_from", "dry_run", "gpu_ids",
    "master_port", "save_steps", "dataset_num_proc", "rollout_log_every", "performance_log",
}


def original_plan():
    args = launcher.parser().parse_args([
        "/srv/original SFT", "/srv/prepared data/train.jsonl", "/srv/original run",
        "--reward-config", str(ROOT / "configs" / "qalco_v4.json"),
        "--gpu-ids", "0,1,2,3,4,5,6,7", "--group-size", "8",
        "--global-batch-size", "64", "--micro-batch-size", "2",
        "--steps-per-generation", "1", "--train-iters", "3000",
        "--save-steps", "500", "--lr", "1e-6", "--kl-beta", "0.01",
        "--lora-rank", "8", "--lora-alpha", "32", "--max-length", "8192",
        "--max-new-tokens", "128", "--max-turns", "65", "--vllm-memory", "0.5",
        "--vllm-max-num-seqs", "32", "--vllm-mm-processor-cache-gb", "0",
        "--recompute-granularity", "full", "--dataset-num-proc", "3",
        "--rollout-log-every", "7", "--performance-log", "on",
    ])
    identity = {
        "schema": 1,
        "framework_commit": launcher.AUDITED_SWIFT_COMMIT,
        "model_path": args.model,
        "parallelism": launcher.parallel_layout(args),
        "model_metadata_sha256": {"config.json": "model-config-hash"},
        "weight_file_metadata": {"model-00001.safetensors": {"bytes": 12345, "mtime_ns": 67890}},
        "dataset_sha256": "dataset-hash", "dataset_tasks": 12000,
        "reward_config_sha256": "v4-reward-hash",
        "settings": {k: v for k, v in vars(args).items() if k not in IGNORED_SETTINGS},
        "code_sha256": {"train_megatron.py": "launcher-hash", "flex_gspo/qalco.py": "reward-code-hash"},
    }
    return {
        "run_identity": identity,
        "argv": launcher.build_command(args),
        "reward_settings": launcher.load_settings(args),
        "parallelism": copy.deepcopy(identity["parallelism"]),
        "environment": {"CUDA_VISIBLE_DEVICES": args.gpu_ids, "FLEX_ROLLOUT_LOG_EVERY": "7"},
        "performance": {"phase_logging": "on"},
    }


def four_gpu_identity(original):
    current = copy.deepcopy(original)
    current["parallelism"].update(world_size=4, data_parallel_size=2, vllm_replicas=1)
    return current


class DPMigrationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.original = original_plan()["run_identity"]
        self.current = four_gpu_identity(self.original)

    def test_only_supported_eight_to_four_layout_is_accepted_without_mutation(self):
        before = copy.deepcopy((self.original, self.current))
        self.assertIsNone(migration.validate_dp_migration(self.original, self.current))
        self.assertEqual((self.original, self.current), before)

    def test_reverse_noop_and_inconsistent_layouts_are_rejected(self):
        pairs = [(self.current, self.original), (self.original, self.original), (self.current, self.current)]
        for field, bad in (("world_size", 2), ("data_parallel_size", 4), ("vllm_replicas", 2),
                           ("tensor_parallel_size", 4), ("expert_parallel_size", 1),
                           ("pipeline_parallel_size", 2), ("context_parallel_size", 2),
                           ("vllm_tensor_parallel_size", 2)):
            current = copy.deepcopy(self.current)
            current["parallelism"][field] = bad
            pairs.append((self.original, current))
        for index, (original, current) in enumerate(pairs):
            with self.subTest(case=index), self.assertRaises(ValueError):
                migration.validate_dp_migration(original, current)

    def test_unsupported_model_parallelism_is_rejected_even_if_unchanged(self):
        for field, bad in (("tensor_parallel_size", 4), ("expert_parallel_size", 1),
                           ("pipeline_parallel_size", 2), ("context_parallel_size", 2),
                           ("vllm_tensor_parallel_size", 2)):
            original, current = copy.deepcopy((self.original, self.current))
            original["parallelism"][field] = current["parallelism"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(original, current)

    def test_training_semantics_must_remain_exactly_equal(self):
        changes = {
            "model": "/srv/checkpoint-500-merged", "group_size": 4, "global_batch_size": 32,
            "micro_batch_size": 1, "steps_per_generation": 2, "train_iters": 2500,
            "lr": 2e-6, "kl_beta": 0.0, "temperature": 0.8, "lora_rank": 16,
            "lora_alpha": 64.0, "max_length": 4096, "max_new_tokens": 64,
            "max_turns": 64, "max_chunk_seconds": 4.0, "seed": 100,
            "vllm_memory": 0.4, "vllm_max_num_seqs": 16,
            "vllm_mm_processor_cache_gb": 1.0, "recompute_granularity": "selective",
            "swift_source": "/srv/different-swift",
        }
        for field, bad in changes.items():
            current = copy.deepcopy(self.current)
            current["settings"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(self.original, current)

    def test_model_data_reward_code_and_framework_evidence_cannot_change(self):
        changes = {
            "schema": 2, "framework_commit": "another-commit", "model_path": "/srv/another-model",
            "model_metadata_sha256": {"config.json": "changed"},
            "weight_file_metadata": {"model-00001.safetensors": {"bytes": 12345, "mtime_ns": 67891}},
            "dataset_sha256": "changed-data", "dataset_tasks": 12001,
            "reward_config_sha256": "v5-reward-hash", "code_sha256": {"train_megatron.py": "changed"},
        }
        for field, bad in changes.items():
            current = copy.deepcopy(self.current)
            current[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(self.original, current)

    def test_removed_or_added_evidence_is_not_silently_ignored(self):
        for field in ("model_path", "dataset_sha256", "reward_config_sha256", "code_sha256", "settings"):
            current = copy.deepcopy(self.current)
            del current[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(self.original, current)
        current = copy.deepcopy(self.current)
        current["new_semantic_setting"] = "different"
        with self.assertRaises(ValueError):
            migration.validate_dp_migration(self.original, current)


class DPMigrationArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.document = original_plan()
        self.checkpoint = Path("/srv/original run/v0/checkpoint-500")
        self.output = Path("/srv/migrated run")
        self.config = ROOT / "configs" / "qalco_v4.json"

    def parsed(self):
        argv = migration.build_resume_argv(self.document, self.checkpoint, self.output, "4,5,6,7", self.config)
        self.assertIsInstance(argv, list)
        self.assertTrue(all(isinstance(value, str) for value in argv))
        return launcher.parser().parse_args(argv)

    def test_restores_original_settings_paths_and_metadata_without_mutating_plan(self):
        before = copy.deepcopy(self.document)
        args = self.parsed()
        retained = {k: v for k, v in vars(args).items() if k not in IGNORED_SETTINGS}
        self.assertEqual(retained, self.document["run_identity"]["settings"])
        self.assertEqual(args.model, "/srv/original SFT")
        self.assertEqual(args.dataset, "/srv/prepared data/train.jsonl")
        self.assertEqual(args.output, str(self.output))
        self.assertEqual(args.resume_from, str(self.checkpoint))
        self.assertEqual(args.reward_config, str(self.config))
        self.assertEqual(args.gpu_ids, "4,5,6,7")
        self.assertEqual((args.save_steps, args.dataset_num_proc, args.rollout_log_every, args.performance_log),
                         (500, 3, 7, "on"))
        self.assertFalse(args.dry_run)
        self.assertEqual(self.document, before)

    def test_native_resume_keeps_optimizer_rng_reference_base_and_lr_horizon(self):
        args = self.parsed()
        command = launcher.build_command(args)
        expected = {
            "--model": "/srv/original SFT", "--mcore_adapter": str(self.checkpoint),
            "--finetune": "false", "--no_load_optim": "false", "--no_load_rng": "false",
            "--no_save_optim": "false", "--no_save_rng": "false", "--train_iters": "3000",
            "--global_batch_size": "64", "--micro_batch_size": "2", "--num_generations": "8",
            "--steps_per_generation": "1", "--lr": "1e-06", "--beta": "0.01",
            "--tensor_model_parallel_size": "2", "--expert_model_parallel_size": "2",
            "--vllm_tensor_parallel_size": "4", "--nproc_per_node": "4",
            "--scale_rewards": "none", "--kl_in_reward": "false",
        }
        for flag, value in expected.items():
            with self.subTest(flag=flag):
                self.assertEqual(command.count(flag), 1)
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--adapters", command)
        self.assertNotIn("--ref_model", command)
        self.assertNotIn("--no_load_optim true", " ".join(command))
        current = copy.deepcopy(self.document["run_identity"])
        current["parallelism"] = launcher.parallel_layout(args)
        migration.validate_dp_migration(self.document["run_identity"], current)

    def test_nondefault_saved_semantic_settings_are_not_replaced_with_recipe_defaults(self):
        changes = {"lr": 8e-7, "kl_beta": 0.02, "seed": 17, "temperature": 0.9,
                   "max_new_tokens": 96, "recompute_granularity": "selective"}
        self.document["run_identity"]["settings"].update(changes)
        args = self.parsed()
        for field, expected in changes.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(args, field), expected)

    def test_missing_saved_dataset_is_not_replaced_with_a_guessed_path(self):
        argv = self.document["argv"]
        index = argv.index("--dataset")
        del argv[index:index + 2]
        with self.assertRaises(ValueError):
            self.parsed()

    def test_only_comet_v4_plans_can_build_a_migration_command(self):
        for reward in ({}, {"version": 5, "quality_provider": "metricx"},
                       {"version": 4, "quality_provider": "metricx"},
                       {"version": 3, "quality_provider": "comet22"}):
            self.document["reward_settings"] = reward
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                self.parsed()


class DPMigrationHistoricalSourceTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()

    def prepare_repository(self, directory):
        repo = Path(directory) / "current checkout"
        repo.mkdir()
        self.git(repo, "init", "--quiet")
        self.git(repo, "config", "user.name", "Migration Test")
        self.git(repo, "config", "user.email", "migration-test@example.invalid")
        self.git(repo, "config", "commit.gpgsign", "false")
        self.git(repo, "config", "core.autocrlf", "false")
        root = repo / "FLEX-GSPO"
        source = {
            "train_megatron.py": b"# original launcher\n",
            "swift_plugin.py": b"# original plugin\n",
            "flex_gspo/__init__.py": b"",
            "flex_gspo/qalco.py": b"# original reward\n",
        }
        for name, contents in source.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        self.git(repo, "add", "FLEX-GSPO")
        self.git(repo, "commit", "--quiet", "-m", "Original training source")
        old_commit = self.git(repo, "rev-parse", "HEAD")
        expected = {name: hashlib.sha256(contents).hexdigest() for name, contents in source.items()}
        (root / "flex_gspo" / "qalco.py").write_bytes(b"# newer reward\n")
        self.git(repo, "add", "FLEX-GSPO/flex_gspo/qalco.py")
        self.git(repo, "commit", "--quiet", "-m", "New checkout source")
        new_commit = self.git(repo, "rev-parse", "HEAD")
        return repo, old_commit, new_commit, expected

    def test_historical_source_is_exact_and_main_checkout_and_dirty_worktree_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            repo, old_commit, new_commit, expected = self.prepare_repository(directory)
            selected = migration.find_original_commit(repo, expected)
            self.assertEqual(selected, old_commit)
            self.assertNotEqual(selected, new_commit)
            root = migration.historical_root(repo, selected, expected)
            self.assertNotEqual(root, repo / "FLEX-GSPO")
            self.assertEqual((root / "flex_gspo" / "qalco.py").read_bytes(), b"# original reward\n")
            self.assertEqual(self.git(repo, "rev-parse", "HEAD"), new_commit)
            self.assertEqual((repo / "FLEX-GSPO" / "flex_gspo" / "qalco.py").read_bytes(), b"# newer reward\n")
            self.assertEqual(self.git(repo, "status", "--porcelain"), "")
            # Reusing an unchanged historical checkout is safe and idempotent.
            self.assertEqual(migration.historical_root(repo, selected, expected), root)
            dirty = root / "swift_plugin.py"
            dirty.write_bytes(b"# user changed this historical file\n")
            with self.assertRaises(ValueError):
                migration.historical_root(repo, selected, expected)
            self.assertEqual(dirty.read_bytes(), b"# user changed this historical file\n")
            self.assertEqual(self.git(repo, "rev-parse", "HEAD"), new_commit)
            self.assertEqual(self.git(repo, "status", "--porcelain"), "")

    def test_absent_fingerprint_and_explicit_wrong_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            repo, old_commit, new_commit, expected = self.prepare_repository(directory)
            absent = {**expected, "swift_plugin.py": hashlib.sha256(b"# never committed\n").hexdigest()}
            with self.assertRaises(ValueError):
                migration.find_original_commit(repo, absent)
            with self.assertRaises(ValueError):
                migration.find_original_commit(repo, expected, explicit=new_commit)
            self.assertEqual(migration.find_original_commit(repo, expected, explicit=old_commit), old_commit)
            self.assertEqual(self.git(repo, "rev-parse", "HEAD"), new_commit)
            self.assertEqual(self.git(repo, "status", "--porcelain"), "")


class DPMigrationMainTests(unittest.TestCase):
    def test_workers_import_historical_code_and_write_audit_without_touching_original_run(self):
        previous_cwd, previous_sys_path = Path.cwd(), sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            current_root, historical = temporary / "current", temporary / "historical"
            for root, marker in ((current_root, "CURRENT"), (historical, "HISTORICAL")):
                package = root / "flex_gspo"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(f"ORIGIN = {marker!r}\n", encoding="utf-8")
            old_run = temporary / "old run"
            checkpoint = old_run / "v0" / "checkpoint-500"
            payload = checkpoint / "iter_0000500"
            payload.mkdir(parents=True)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("500", encoding="utf-8")
            (payload / "common.pt").write_bytes(b"fake metadata: no torch deserialization in this CPU test")
            (payload / "__0_0.distcp").write_bytes(b"fake shard: native GPU loading is outside this test")
            (checkpoint / "args.json").write_text(json.dumps({
                "no_save_optim": False, "no_save_rng": False,
                "global_batch_size": 64, "train_iters": 3000, "lora_rank": 8,
            }), encoding="utf-8")
            document = original_plan()
            model = str(temporary / "original SFT")
            dataset = str(temporary / "prepared data" / "train.jsonl")
            document["run_identity"]["settings"]["model"] = model
            document["run_identity"]["model_path"] = model
            for flag, value in (("--model", model), ("--dataset", dataset)):
                document["argv"][document["argv"].index(flag) + 1] = value
            launch = old_run / "launch_original.json"
            launch.write_text(json.dumps(document), encoding="utf-8")
            original_files = {p.relative_to(old_run): p.read_bytes() for p in old_run.rglob("*") if p.is_file()}
            output = temporary / "new migration"
            restored_identity = four_gpu_identity(document["run_identity"])
            observed = {}
            test = self

            class HistoricalLauncherLoader:
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    def main(forwarded):
                        args = launcher.parser().parse_args(forwarded)
                        # A fresh python -m worker would put cwd before PYTHONPATH.
                        # Execute a real child interpreter with both package copies present.
                        env = dict(os.environ)
                        env["PYTHONPATH"] = os.pathsep.join((str(historical), str(current_root)))
                        worker = subprocess.run(
                            [sys.executable, "-c", "import flex_gspo; print(flex_gspo.ORIGIN)"],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                        observed["worker_origin"] = worker.stdout.decode().strip()
                        test.assertEqual(observed["worker_origin"], "HISTORICAL")
                        test.assertEqual(Path.cwd(), historical)
                        test.assertEqual(args.model, model)
                        test.assertEqual(args.resume_from, str(checkpoint))
                        test.assertEqual(args.master_port, 29611)
                        module.verify_resume_identity(args, restored_identity)
                        (output / "launch_migrated.json").write_text(
                            json.dumps({"run_identity": restored_identity}), encoding="utf-8")
                        return 0
                    module.main = main

            spec = importlib.util.spec_from_loader("migration_historical_test", HistoricalLauncherLoader())
            try:
                os.chdir(current_root)
                with redirect_stdout(StringIO()), \
                        patch.object(migration, "git", return_value=str(current_root).encode()), \
                        patch.object(migration, "find_original_commit", return_value="a" * 40), \
                        patch.object(migration, "historical_root", return_value=historical), \
                        patch.object(migration, "select_reward_config", return_value=ROOT / "configs" / "qalco_v4.json"), \
                        patch.object(migration.importlib.util, "spec_from_file_location", return_value=spec):
                    status = migration.main([
                        "--launch-json", str(launch), "--resume-from", str(checkpoint),
                        "--output-dir", str(output), "--gpu-ids", "0,2,4,6",
                        "--master-port", "29611", "--repo", str(current_root),
                    ])
                self.assertEqual(status, 0)
                self.assertEqual(observed["worker_origin"], "HISTORICAL")
                audit = json.loads((output / "dp_migration.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["original_identity"], document["run_identity"])
                self.assertEqual(audit["new_identity"], restored_identity)
                self.assertEqual(audit["restored_iteration"], 500)
                self.assertEqual(audit["original_launch_sha256"], hashlib.sha256(launch.read_bytes()).hexdigest())
                self.assertEqual(json.loads((output / "launch_migrated.json").read_text(encoding="utf-8"))
                                 ["run_identity"], restored_identity)
                self.assertEqual({p.relative_to(old_run): p.read_bytes() for p in old_run.rglob("*") if p.is_file()},
                                 original_files)
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_sys_path


if __name__ == "__main__":
    unittest.main()
