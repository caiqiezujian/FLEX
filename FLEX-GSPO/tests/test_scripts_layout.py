"""Relocated diagnostic entry points work without network, models, or GPUs."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from Scripts import check_comet22_service, check_metricx_service


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "analyze_training_run", "audit_reward_candidates", "check_comet22_service",
    "check_environment", "check_metricx_service", "collect_brevity_samples",
    "collect_training_status", "monitor_training", "replay_reward_audit", "relocate_prepared_data",
)


class ScriptLayoutTests(unittest.TestCase):
    def test_help_by_absolute_path_and_module(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in SCRIPTS:
                commands = (
                    ([sys.executable, str(ROOT / "Scripts" / (name + ".py")), "--help"], directory),
                    ([sys.executable, "-m", "Scripts." + name, "--help"], ROOT),
                )
                for command, cwd in commands:
                    with self.subTest(script=name, invocation=command[1]):
                        result = subprocess.run(command, cwd=cwd, capture_output=True,
                                                text=True, encoding="utf-8", errors="replace", timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("usage:", result.stdout)

    def test_probe_default_configs_resolve_from_another_directory(self):
        expected = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8-sig"))
        report = {"quality_regression_passed": True, "batch_order_consistency_passed": True}
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for module in (check_comet22_service, check_metricx_service):
                    with self.subTest(module=module.__name__), \
                            patch.object(module, "probe", return_value=report) as probe, \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(module.main([]), 0)
                        self.assertEqual(probe.call_args.args[0], expected)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
