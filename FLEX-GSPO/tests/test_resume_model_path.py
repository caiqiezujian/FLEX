"""Explicit SFT relocation preserves native resume and model identity checks."""
import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from Scripts import resume_dp_migration as migration
from test_resume_dp_migration import ROOT, four_gpu_identity, launcher, original_plan
from test_resume_copied_source import forbid_git, migration_fixture, printed_plan


def relocated_identity(original, target):
    current = four_gpu_identity(original)
    current["model_path"] = str(target.resolve())
    current["settings"]["model"] = str(target.resolve())
    for metadata in current["weight_file_metadata"].values():
        metadata["mtime_ns"] += 1000
    return current


class ResumeModelPathTests(unittest.TestCase):
    def test_explicit_relocation_accepts_only_timestamp_difference_without_mutating_identities(self):
        original = original_plan()["run_identity"]
        target = ROOT / "models" / "copied SFT"
        current = relocated_identity(original, target)
        snapshots = copy.deepcopy((original, current))
        migration.validate_dp_migration(original, current, relocated_model=target)
        self.assertEqual((original, current), snapshots)
        current["code_sha256"]["train_megatron.py"] = "current-launcher"
        migration.validate_dp_migration(original, current, relocated_model=target, allow_code_change=True)
        with self.assertRaises(ValueError):
            migration.validate_dp_migration(original, current, relocated_model=target)

    def test_no_explicit_override_retains_strict_model_path_and_timestamp_checks(self):
        original = original_plan()["run_identity"]
        with self.assertRaises(ValueError):
            migration.validate_dp_migration(original, relocated_identity(original, ROOT / "models"))
        current = four_gpu_identity(original)
        next(iter(current["weight_file_metadata"].values()))["mtime_ns"] += 1
        with self.assertRaises(ValueError):
            migration.validate_dp_migration(original, current)

    def test_override_requires_both_identity_model_fields_to_match_resolved_target(self):
        original = original_plan()["run_identity"]
        target = ROOT / "models" / "copied SFT"
        for field in ("model_path", "settings.model", "missing settings.model"):
            with self.subTest(field=field):
                current = relocated_identity(original, target)
                if field == "model_path":
                    current["model_path"] = original["model_path"]
                elif field == "settings.model":
                    current["settings"]["model"] = original["settings"]["model"]
                else:
                    current["settings"].pop("model")
                with self.assertRaises(ValueError):
                    migration.validate_dp_migration(original, current, relocated_model=target)

    def test_relocation_rejects_changed_or_missing_model_metadata(self):
        original = original_plan()["run_identity"]
        target = ROOT / "models" / "copied SFT"
        cases = ("size", "extra shard", "renamed shard", "extra metadata", "config hash",
                 "missing config", "missing identity field", "empty weights", "invalid weights",
                 "missing size", "zero size", "string size")
        for case in cases:
            with self.subTest(case=case):
                current = relocated_identity(original, target)
                weights = current["weight_file_metadata"]
                name = next(iter(weights))
                if case == "size":
                    weights[name]["bytes"] += 1
                elif case == "extra shard":
                    weights["extra.safetensors"] = copy.deepcopy(weights[name])
                elif case == "renamed shard":
                    weights["renamed.safetensors"] = weights.pop(name)
                elif case == "extra metadata":
                    weights[name]["digest"] = "new metadata must not be ignored"
                elif case == "config hash":
                    current["model_metadata_sha256"]["config.json"] = "different model"
                elif case == "missing config":
                    current["model_metadata_sha256"] = {}
                elif case == "missing identity field":
                    current.pop("weight_file_metadata")
                elif case == "empty weights":
                    current["weight_file_metadata"] = {}
                elif case == "invalid weights":
                    current["weight_file_metadata"] = []
                elif case == "missing size":
                    weights[name].pop("bytes")
                elif case == "zero size":
                    weights[name]["bytes"] = 0
                elif case == "string size":
                    weights[name]["bytes"] = "12345"
                with self.assertRaises(ValueError):
                    migration.validate_dp_migration(original, current, relocated_model=target)

    def test_model_relocation_does_not_relax_training_data_reward_or_parallelism(self):
        original = original_plan()["run_identity"]
        target = ROOT / "models" / "copied SFT"
        for case in ("dataset_sha256", "reward_config_sha256", "global_batch_size", "train_iters", "tensor_parallel_size"):
            with self.subTest(case=case):
                current = relocated_identity(original, target)
                if case in ("dataset_sha256", "reward_config_sha256"):
                    current[case] = "different"
                elif case == "tensor_parallel_size":
                    current["parallelism"][case] = 1
                else:
                    current["settings"][case] += 1
                with self.assertRaises(ValueError):
                    migration.validate_dp_migration(original, current, relocated_model=target, allow_code_change=True)

    def test_forwarded_model_override_retains_native_checkpoint_and_original_document(self):
        document = original_plan()
        before = copy.deepcopy(document)
        target = ROOT / "models" / "copied SFT"
        checkpoint = ROOT / "old run" / "checkpoint-500"
        forwarded = migration.build_resume_argv(document, checkpoint, ROOT / "new run", "0,1,2,3",
                                               ROOT / "configs" / "qalco_v4.json", model_path=target)
        args = launcher.parser().parse_args(forwarded)
        self.assertEqual(args.model, str(target.resolve()))
        self.assertEqual(args.dataset, migration.flag_value(document["argv"], "--dataset"))
        self.assertEqual((args.global_batch_size, args.train_iters, args.lora_rank), (64, 3000, 8))
        command = launcher.build_command(args)
        self.assertEqual(migration.flag_value(command, "--model"), str(target.resolve()))
        self.assertEqual(migration.flag_value(command, "--mcore_adapter"), str(checkpoint))
        for flag in ("--finetune", "--no_load_optim", "--no_load_rng", "--no_save_optim", "--no_save_rng"):
            self.assertEqual(migration.flag_value(command, flag), "false")
        self.assertEqual(document, before)

    def test_cli_requires_absolute_existing_model_before_source_execution_and_reports_dry_run(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, original, checkpoint, launch, output, document, argv = migration_fixture(directory)
            guards = forbid_git(stack)
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Dry run must not import source")))
            old_cwd = Path.cwd()
            for value in ("relative/model", str(Path(directory) / "missing model")):
                with self.subTest(value=value), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(migration.main(argv + ["--model-path", value, "--dry-run"]), 2)
                self.assertEqual(Path.cwd(), old_cwd)
            target = Path(directory).resolve() / "models" / "sft_checkpoint-26508"
            target.mkdir(parents=True)
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(migration.main(argv + ["--model-path", str(target), "--dry-run"]), 0)
            plan = printed_plan(stdout.getvalue())
            self.assertEqual(plan["model_path"], str(target))
            self.assertEqual(plan["model_relocation"]["from"], document["run_identity"]["settings"]["model"])
            self.assertEqual(plan["model_relocation"]["to"], str(target))
            self.assertIn("weight shard names/sizes", plan["model_relocation"]["verification"])
            self.assertFalse(output.exists())
            loader.assert_not_called()
            for guard in guards:
                guard.assert_not_called()

    def test_main_passes_relocated_base_and_audits_native_resume_without_git_or_original_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, original, checkpoint, launch, output, document, argv = migration_fixture(directory)
            target = Path(directory).resolve() / "models" / "sft_checkpoint-26508"
            target.mkdir(parents=True)
            current = relocated_identity(document["run_identity"], target)
            original_files = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
            called = []
            test = self

            class Loader:
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    def main(forwarded):
                        args = launcher.parser().parse_args(forwarded)
                        test.assertEqual(Path.cwd(), source)
                        test.assertEqual(args.model, str(target))
                        test.assertEqual(Path(args.resume_from), checkpoint)
                        test.assertEqual((args.global_batch_size, args.train_iters), (64, 3000))
                        command = launcher.build_command(args)
                        test.assertEqual(migration.flag_value(command, "--model"), str(target))
                        test.assertEqual(migration.flag_value(command, "--mcore_adapter"), str(checkpoint))
                        for flag in ("--finetune", "--no_load_optim", "--no_load_rng"):
                            test.assertEqual(migration.flag_value(command, flag), "false")
                        module.verify_resume_identity(args, current)
                        (output / "launch_new.json").write_text(json.dumps({"run_identity": current}), encoding="utf-8")
                        called.append(args)
                        return 0
                    module.main = main

            spec = importlib.util.spec_from_loader("model_relocation_launcher_test", Loader())
            old_cwd, old_path = Path.cwd(), sys.path[:]
            try:
                with ExitStack() as stack, redirect_stdout(StringIO()):
                    guards = forbid_git(stack)
                    stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location", return_value=spec))
                    status = migration.main(argv + ["--model-path", str(target)])
                    for guard in guards:
                        guard.assert_not_called()
                self.assertEqual(status, 0)
                self.assertEqual(len(called), 1)
                audit = json.loads((output / "dp_migration.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["model_path"], str(target))
                self.assertEqual(audit["model_relocation"]["from"], document["run_identity"]["settings"]["model"])
                self.assertEqual(audit["model_relocation"]["to"], str(target))
                self.assertEqual(audit["original_identity"], document["run_identity"])
                self.assertEqual(audit["new_identity"], current)
                expected_mtimes = {name: {"original": info["mtime_ns"], "current": current["weight_file_metadata"][name]["mtime_ns"]}
                                   for name, info in document["run_identity"]["weight_file_metadata"].items()}
                self.assertEqual(audit["model_weight_mtime_changes"], expected_mtimes)
                self.assertEqual(audit["original_launch_sha256"], hashlib.sha256(launch.read_bytes()).hexdigest())
                self.assertEqual(json.loads((output / "launch_new.json").read_text(encoding="utf-8"))["run_identity"], current)
                self.assertEqual({p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}, original_files)
            finally:
                os.chdir(old_cwd)
                sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
