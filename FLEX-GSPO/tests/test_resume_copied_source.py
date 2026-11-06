"""Copied-source migration needs exact source bytes, never a Git executable."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
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
from test_resume_dp_migration import ROOT, four_gpu_identity, launcher, original_plan


def source_fixture(root):
    sources = {
        "train_megatron.py": b"raise AssertionError('source validation must not execute this launcher')\n",
        "swift_plugin.py": b"raise AssertionError('source validation must not execute this plugin')\n",
        "flex_gspo/__init__.py": b"ORIGIN = 'COPIED'\n",
        "flex_gspo/qalco.py": b"# original V4 reward source\n",
    }
    for name, contents in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for name in ("Scripts/check_environment.py", "Scripts/_bootstrap.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# copied runtime support\n", encoding="utf-8")
    return {name: hashlib.sha256(contents).hexdigest() for name, contents in sources.items()}


def migration_fixture(directory):
    root = Path(directory).resolve()
    source = root / "copied FLEX-GSPO"
    expected = source_fixture(source)
    config = source / "configs" / "qalco_v4.json"
    config.parent.mkdir()
    config.write_bytes((ROOT / "configs" / "qalco_v4.json").read_bytes())
    original = root / "old run"
    checkpoint = original / "v0" / "checkpoint-500"
    payload = checkpoint / "iter_0000500"
    payload.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("500", encoding="utf-8")
    (payload / "common.pt").write_bytes(b"metadata placeholder; GPU checkpoint decoding is not exercised")
    (payload / "__0_0.distcp").write_bytes(b"native shard placeholder")
    (checkpoint / "args.json").write_text(json.dumps({
        "no_save_optim": False, "no_save_rng": False,
        "global_batch_size": 64, "train_iters": 3000, "lora_rank": 8,
    }), encoding="utf-8")
    document = original_plan()
    model, dataset = str(root / "original SFT"), str(root / "prepared" / "train.jsonl")
    document["run_identity"].update(
        model_path=model, code_sha256=expected,
        reward_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    document["run_identity"]["settings"]["model"] = model
    for flag, value in (("--model", model), ("--dataset", dataset)):
        document["argv"][document["argv"].index(flag) + 1] = value
    launch = original / "launch_original.json"
    launch.write_text(json.dumps(document), encoding="utf-8")
    output = root / "new migration"
    argv = ["--launch-json", str(launch), "--resume-from", str(checkpoint),
            "--output-dir", str(output), "--gpu-ids", "0,2,4,6",
            "--master-port", "29613", "--source-dir", str(source)]
    return source, original, checkpoint, launch, output, document, argv


def forbid_git(stack):
    return [stack.enter_context(patch.object(migration, name, side_effect=AssertionError("Git path was invoked")))
            for name in ("git", "find_original_commit", "historical_root")]


class CopiedSourceValidationTests(unittest.TestCase):
    def test_exact_bytes_and_set_accept_without_git_import_or_rejecting_extra_support_scripts(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "plain source"
            expected = source_fixture(root)
            (root / "Scripts" / "resume_dp_migration.py").write_text("# helper outside training hash\n", encoding="utf-8")
            guards = forbid_git(stack)
            self.assertFalse((root / ".git").exists())
            self.assertEqual(migration.copied_root(root, expected), root.resolve())
            for guard in guards:
                guard.assert_not_called()

    def test_missing_extra_and_changed_training_files_are_rejected_without_repair(self):
        for kind in ("missing", "extra", "changed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = source_fixture(root)
                target = root / "flex_gspo" / ("metricx_reward.py" if kind == "extra" else "qalco.py")
                if kind == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"# different source\n")
                snapshot = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(ValueError, kind):
                    migration.copied_root(root, expected)
                self.assertEqual({p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}, snapshot)

    def test_runtime_support_scripts_are_required_even_outside_the_old_hash_set(self):
        for name in ("Scripts/check_environment.py", "Scripts/_bootstrap.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = source_fixture(root)
                (root / name).unlink()
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    migration.copied_root(root, expected)

    def test_invalid_fingerprints_and_wrong_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "FLEX-GSPO"
            expected = source_fixture(root)
            for bad in ({**expected, "../outside.py": "a" * 64},
                        {**expected, "swift_plugin.py": "not-a-sha"}):
                with self.subTest(expected=bad), self.assertRaises(ValueError):
                    migration.copied_root(root, bad)
            with self.assertRaises(ValueError):
                migration.copied_root(root.parent, expected)
            with self.assertRaises(ValueError):
                migration.copied_root(root / "absent", expected)


class CopiedSourceMainTests(unittest.TestCase):
    def test_dry_run_verifies_source_and_reports_copied_provenance_without_git_or_execution(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, _, _, _, output, _, argv = migration_fixture(directory)
            guards = forbid_git(stack)
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Dry run imported source")))
            stdout = stack.enter_context(redirect_stdout(StringIO()))
            self.assertEqual(migration.main([*argv, "--dry-run"]), 0)
            text = stdout.getvalue()
            plan = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
            self.assertEqual(plan["source_mode"], "copied")
            self.assertIsNone(plan["source_commit"])
            self.assertEqual(plan["source_dir"], str(source.resolve()))
            self.assertEqual((plan["restored_iteration"], plan["global_batch_size"], plan["total_train_iters"]),
                             (500, 64, 3000))
            self.assertFalse(output.exists())
            self.assertFalse((source / ".git").exists())
            loader.assert_not_called()
            for guard in guards:
                guard.assert_not_called()

    def test_source_mismatch_stops_before_any_source_execution_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, _, _, _, output, _, argv = migration_fixture(directory)
            (source / "train_megatron.py").write_bytes(b"raise AssertionError('do not execute changed source')\n")
            guards = forbid_git(stack)
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Mismatched source imported")))
            stderr = stack.enter_context(redirect_stderr(StringIO()))
            self.assertEqual(migration.main(argv), 2)
            self.assertIn("differs from the original", stderr.getvalue())
            self.assertFalse(output.exists())
            loader.assert_not_called()
            for guard in guards:
                guard.assert_not_called()

    def test_source_commit_cannot_supply_fictitious_git_provenance_in_copied_mode(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, _, _, _, output, _, argv = migration_fixture(directory)
            guards = forbid_git(stack)
            stderr = stack.enter_context(redirect_stderr(StringIO()))
            self.assertEqual(migration.main([*argv, "--source-commit", "a" * 40]), 2)
            self.assertIn("source-commit", stderr.getvalue())
            self.assertFalse(output.exists())
            for guard in guards:
                guard.assert_not_called()

    def test_main_uses_copied_code_in_real_child_and_records_native_migration_without_git(self):
        previous_cwd, previous_sys_path = Path.cwd(), sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            source, original, checkpoint, launch, output, document, argv = migration_fixture(directory)
            shadow = Path(directory).resolve() / "current checkout"
            (shadow / "flex_gspo").mkdir(parents=True)
            (shadow / "flex_gspo" / "__init__.py").write_text("ORIGIN = 'CURRENT'\n", encoding="utf-8")
            original_files = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
            restored_identity = four_gpu_identity(document["run_identity"])
            observed = {}
            test = self

            class CopiedLauncherLoader:
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    def main(forwarded):
                        args = launcher.parser().parse_args(forwarded)
                        env = dict(os.environ)
                        env["PYTHONPATH"] = os.pathsep.join((str(source), str(shadow)))
                        worker = subprocess.run(
                            [sys.executable, "-c", "import flex_gspo; print(flex_gspo.ORIGIN)"],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                        observed["origin"] = worker.stdout.decode().strip()
                        test.assertEqual(observed["origin"], "COPIED")
                        test.assertEqual(Path.cwd(), source.resolve())
                        test.assertEqual(args.resume_from, str(checkpoint))
                        test.assertEqual(args.model, document["run_identity"]["settings"]["model"])
                        test.assertEqual((args.global_batch_size, args.train_iters, args.master_port), (64, 3000, 29613))
                        command = launcher.build_command(args)
                        for flag, expected in (("--finetune", "false"), ("--no_load_optim", "false"),
                                               ("--no_load_rng", "false"), ("--mcore_adapter", str(checkpoint))):
                            test.assertEqual(command[command.index(flag) + 1], expected)
                        module.verify_resume_identity(args, restored_identity)
                        (output / "launch_new.json").write_text(json.dumps({"run_identity": restored_identity}), encoding="utf-8")
                        return 0
                    module.main = main

            spec = importlib.util.spec_from_loader("copied_launcher_test", CopiedLauncherLoader())
            try:
                os.chdir(shadow)
                with ExitStack() as stack:
                    guards = forbid_git(stack)
                    stack.enter_context(redirect_stdout(StringIO()))
                    loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                              return_value=spec))
                    self.assertEqual(migration.main(argv), 0)
                    loader.assert_called_once_with("flex_original_launcher", source / "train_megatron.py")
                    for guard in guards:
                        guard.assert_not_called()
                self.assertEqual(observed["origin"], "COPIED")
                audit = json.loads((output / "dp_migration.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["source_mode"], "copied")
                self.assertIsNone(audit["source_commit"])
                self.assertEqual(audit["source_dir"], str(source.resolve()))
                self.assertEqual(audit["original_identity"], document["run_identity"])
                self.assertEqual(audit["new_identity"], restored_identity)
                self.assertEqual(audit["original_launch_sha256"], hashlib.sha256(launch.read_bytes()).hexdigest())
                self.assertEqual(json.loads((output / "launch_new.json").read_text(encoding="utf-8"))
                                 ["run_identity"], restored_identity)
                self.assertEqual({p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()},
                                 original_files)
                self.assertFalse((source / ".git").exists())
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_sys_path


if __name__ == "__main__":
    unittest.main()
