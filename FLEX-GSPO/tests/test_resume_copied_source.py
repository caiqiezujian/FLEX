"""Current-code V4 resume records source changes while retaining training state."""
import copy
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


def printed_plan(text):
    # Source-change diagnostics may print their own JSON before the full plan.
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "migration" in value:
            return value
    raise AssertionError("No migration plan was printed")


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

    def test_optional_missing_extra_and_changed_sources_are_allowed_and_recorded_without_repair(self):
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
                self.assertEqual(migration.copied_root(root, expected), root.resolve())
                actual = migration.source_hashes(root)
                changes = migration.compare_source_hashes(expected, actual)
                expected_changes = {"missing": [], "extra": [], "changed": []}
                expected_changes[kind] = [target.relative_to(root).as_posix()]
                self.assertEqual(changes, expected_changes)
                if kind != "missing":
                    self.assertEqual(actual[target.relative_to(root).as_posix()],
                                     hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertEqual({p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}, snapshot)

    def test_essential_source_and_runtime_support_files_are_still_required(self):
        for name in ("train_megatron.py", "swift_plugin.py", "flex_gspo/__init__.py",
                     "Scripts/check_environment.py", "Scripts/_bootstrap.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = source_fixture(root)
                (root / name).unlink()
                with self.assertRaises(ValueError):
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
    def test_dry_run_reports_current_source_provenance_without_git_or_execution(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, _, _, _, output, _, argv = migration_fixture(directory)
            guards = forbid_git(stack)
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Dry run imported source")))
            stdout = stack.enter_context(redirect_stdout(StringIO()))
            self.assertEqual(migration.main([*argv, "--dry-run"]), 0)
            text = stdout.getvalue()
            plan = printed_plan(text)
            self.assertEqual(plan["source_mode"], "current")
            self.assertEqual(plan["source_policy"], "use_current_v4")
            self.assertEqual(plan["source_changes"], {"missing": [], "extra": [], "changed": []})
            self.assertEqual(plan["source_code_sha256"], migration.source_hashes(source))
            self.assertIsNone(plan["source_commit"])
            self.assertEqual(plan["source_dir"], str(source.resolve()))
            self.assertEqual((plan["restored_iteration"], plan["global_batch_size"], plan["total_train_iters"]),
                             (500, 64, 3000))
            self.assertFalse(output.exists())
            self.assertFalse((source / ".git").exists())
            loader.assert_not_called()
            for guard in guards:
                guard.assert_not_called()

    def test_source_changes_are_allowed_and_reported_by_dry_run(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, _, _, launch, output, document, argv = migration_fixture(directory)
            original_launch = launch.read_bytes()
            (source / "train_megatron.py").write_bytes(b"raise AssertionError('dry run must not execute source')\n")
            (source / "flex_gspo" / "metricx_reward.py").write_bytes(b"# new V5 module alongside retained V4\n")
            guards = forbid_git(stack)
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Dry run imported source")))
            stdout = stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
            self.assertEqual(migration.main([*argv, "--dry-run"]), 0)
            text = stdout.getvalue()
            plan = printed_plan(text)
            self.assertEqual(plan["source_changes"], {"missing": [], "extra": ["flex_gspo/metricx_reward.py"],
                                                      "changed": ["train_megatron.py"]})
            self.assertNotEqual(plan["source_code_sha256"], document["run_identity"]["code_sha256"])
            self.assertEqual(launch.read_bytes(), original_launch)
            self.assertFalse(output.exists())
            loader.assert_not_called()
            for guard in guards:
                guard.assert_not_called()

    def test_default_source_is_the_current_helper_directory_without_any_git_lookup(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source, _, _, _, output, _, argv = migration_fixture(directory)
            helper = source / "Scripts" / "resume_dp_migration.py"
            helper.write_bytes(Path(migration.__file__).read_bytes())
            # The default must use the helper's FLEX-GSPO, not a surrounding Git checkout.
            del argv[-2:]
            guards = forbid_git(stack)
            stack.enter_context(patch.object(migration, "__file__", str(helper)))
            loader = stack.enter_context(patch.object(migration.importlib.util, "spec_from_file_location",
                                                      side_effect=AssertionError("Dry run imported source")))
            stdout = stack.enter_context(redirect_stdout(StringIO()))
            self.assertEqual(migration.main([*argv, "--dry-run"]), 0)
            text = stdout.getvalue()
            plan = printed_plan(text)
            self.assertEqual(plan["source_mode"], "current")
            self.assertEqual(plan["source_dir"], str(source.resolve()))
            self.assertIsNone(plan["source_commit"])
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

    def test_main_uses_current_code_and_preserves_both_original_and_new_source_identities(self):
        previous_cwd, previous_sys_path = Path.cwd(), sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            source, original, checkpoint, launch, output, document, argv = migration_fixture(directory)
            shadow = Path(directory).resolve() / "current checkout"
            (shadow / "flex_gspo").mkdir(parents=True)
            (shadow / "flex_gspo" / "__init__.py").write_text("ORIGIN = 'CURRENT'\n", encoding="utf-8")
            original_files = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
            (source / "flex_gspo" / "qalco.py").write_bytes(b"# updated dispatch retaining V4\n")
            (source / "flex_gspo" / "metricx_reward.py").write_bytes(b"# independent V5 implementation\n")
            restored_identity = four_gpu_identity(document["run_identity"])
            restored_identity["code_sha256"] = migration.source_hashes(source)
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
                self.assertEqual(audit["source_mode"], "current")
                self.assertEqual(audit["source_policy"], "use_current_v4")
                self.assertEqual(audit["source_changes"], {"missing": [], "extra": ["flex_gspo/metricx_reward.py"],
                                                          "changed": ["flex_gspo/qalco.py"]})
                self.assertEqual(audit["source_code_sha256"], restored_identity["code_sha256"])
                self.assertNotEqual(audit["source_code_sha256"], document["run_identity"]["code_sha256"])
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


class CurrentCodeIdentityTests(unittest.TestCase):
    def test_source_change_requires_explicit_opt_in_and_does_not_rewrite_original_identity(self):
        original = original_plan()["run_identity"]
        current = four_gpu_identity(original)
        current["code_sha256"] = {"train_megatron.py": "updated-launcher", "flex_gspo/metricx_reward.py": "new-module"}
        before = copy.deepcopy((original, current))
        with self.assertRaises(ValueError):
            migration.validate_dp_migration(original, current)
        self.assertIsNone(migration.validate_dp_migration(original, current, allow_code_change=True))
        self.assertEqual((original, current), before)

    def test_allowing_current_code_does_not_allow_model_data_reward_or_training_changes(self):
        original = original_plan()["run_identity"]
        for field, value in (("model_path", "/different-base"), ("dataset_sha256", "different-data"),
                             ("reward_config_sha256", "different-reward"), ("framework_commit", "different-framework")):
            current = four_gpu_identity(original)
            current["code_sha256"] = {"train_megatron.py": "updated-launcher"}
            current[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(original, current, allow_code_change=True)
        for field, value in (("global_batch_size", 32), ("train_iters", 2500), ("group_size", 4),
                             ("kl_beta", 0), ("model", "/checkpoint-500-merged")):
            current = four_gpu_identity(original)
            current["code_sha256"] = {"train_megatron.py": "updated-launcher"}
            current["settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.validate_dp_migration(original, current, allow_code_change=True)


if __name__ == "__main__":
    unittest.main()
