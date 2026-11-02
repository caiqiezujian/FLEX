"""Exercise the standalone legacy preparation CLI with local WAV fixtures."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from flex_gspo.audio import decode_chunk_ref
from flex_gspo.prompts import SYSTEM_PROMPT
from test_prepare import raw_row, write_pcm


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "prepare_legacy_data.py"
# Historical GitHub prompt: the legacy input adapter must not restore this.
OLD_SYSTEM_PROMPT = (
    "你是一个同声传译专家，负责将用户音频中的语音同声翻译为中文。"
    "用户将按时间顺序逐块发送音频。每收到一块音频后，立即输出响应："
    "若该块暂无可翻译内容，输出<|WAIT|>；若仅能确认部分译文、语义尚不完整，输出<|FLEX|>并紧跟"
    "确信的译文（只输出有把握的部分，宁少勿错）；若译文语义已完整，输出<|DIRECT|>并紧跟新识别"
    "的译文。所有译文均为增量输出：只输出新增内容，不重复、不修改此前已输出的任何译文。"
)


class LegacyPreparationCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "original wavs"
        self.audio.mkdir()
        self.source = self.root / "legacy input.jsonl"
        self.output = self.root / "prepared"
        self.cwd = self.root / "unrelated working directory"
        self.cwd.mkdir()

    def write_rows(self, rows):
        self.source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def run_cli(self, *extra, output=None, include_audio=True):
        command = [sys.executable, str(CLI), "--input-jsonl", str(self.source),
                   "--output-dir", str(output or self.output), "--workers", "2"]
        if include_audio:
            command.extend(["--audio-dir", str(self.audio)])
        command.extend(extra)
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(command, cwd=self.cwd, env=environment, capture_output=True,
                              text=True, encoding="utf-8", timeout=30)

    def read_jsonl(self, name="train.jsonl", output=None):
        return [json.loads(line) for line in ((output or self.output) / name)
                .read_text(encoding="utf-8").splitlines() if line.strip()]

    def report(self, output=None):
        return json.loads(((output or self.output) / "prepare_report.json")
                          .read_text(encoding="utf-8"))

    def assert_exit(self, result, expected):
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_real_cli_from_other_cwd_uses_latest_prompt_and_virtual_chunks(self):
        path = self.audio / "example.wav"
        write_pcm(path, seconds=2.25)
        before, before_mtime = path.read_bytes(), path.stat().st_mtime_ns
        source = "SOURCE_TRANSCRIPT_8421 is metadata only."
        reference = "参考译文甲乙丙丁仅用于评分。"
        self.write_rows([raw_row(source=source, target=reference)])

        self.assert_exit(self.run_cli(), 0)
        row, = self.read_jsonl()
        self.assertEqual(row["messages"], [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<audio>"},
        ])
        messages = json.dumps(row["messages"], ensure_ascii=False)
        self.assertNotIn(source, messages)
        self.assertNotIn(reference, messages)
        self.assertIsInstance(row["flex_task"], str)
        task = json.loads(row["flex_task"])
        self.assertEqual((task["id"], task["input_format"]), ("example", "legacy"))
        self.assertEqual(task["audio_path"], str(path.resolve()))
        self.assertEqual((task["source_text"], task["reference_text"]), (source, reference))
        self.assertEqual(task["chunks"], [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.25, "input_seconds": 1.0},
        ])
        self.assertEqual(len(row["audios"]), 1)
        chunk = decode_chunk_ref(row["audios"][0])
        self.assertEqual((chunk["start"], chunk["end"]), (0.0, 1.0))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, before_mtime)
        self.assertEqual(list(self.audio.iterdir()), [path])
        self.assertEqual(list(self.output.rglob("*.wav")), [])
        self.assertNotIn("trajectory", row)
        report = self.report()
        self.assertEqual(report["input_format"], "legacy")
        self.assertFalse(report["offline_trajectories_required"])

    def test_latest_prompt_matches_current_sft_template(self):
        source = ROOT.parent / "SFT-Scripts" / "prepare_duplex_data.py"
        if not source.is_file():
            self.skipTest("SFT-Scripts is not included in this standalone checkout")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        template = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(name, ast.Name)
                                and name.id == "PROTOCOL_SYSTEM_PROMPT_TEMPLATE"
                                for name in node.targets))
        self.assertEqual(SYSTEM_PROMPT, ast.literal_eval(template).format(target_lang="中文"))
        self.assertNotEqual(SYSTEM_PROMPT, OLD_SYSTEM_PROMPT)

    def test_duplicate_reporting_and_invalid_rows_require_explicit_skip(self):
        write_pcm(self.audio / "example.wav")
        self.write_rows([raw_row(), raw_row(), raw_row(target="另一条参考。"), raw_row("missing")])
        strict_output = self.root / "strict prepared"
        self.assert_exit(self.run_cli(output=strict_output), 2)
        self.assertFalse((strict_output / "train.jsonl").exists())
        self.assertFalse(self.report(strict_output)["manifest_published"])

        self.assert_exit(self.run_cli("--skip-invalid"), 0)
        report = self.report()
        self.assertEqual((report["accepted_rows"], report["duplicate_rows"],
                          report["conflicting_duplicate_rows"], report["rejected_rows"]),
                         (1, 2, 1, 1))
        row, = self.read_jsonl()
        self.assertEqual(json.loads(row["flex_task"])["reference_text"], raw_row()["zh"]["txt"])
        duplicates = self.read_jsonl("duplicates.jsonl")
        self.assertEqual([row["conflicting_text"] for row in duplicates], [False, True])
        rejection, = self.read_jsonl("rejected.jsonl")
        self.assertEqual(rejection["key"], "missing")

    def test_historical_prompt_assertion_cannot_restore_old_prompt(self):
        write_pcm(self.audio / "example.wav")
        self.write_rows([raw_row()])
        prompt = self.root / "historical prompt.txt"
        prompt.write_text(OLD_SYSTEM_PROMPT, encoding="utf-8")
        result = self.run_cli("--system-prompt-file", str(prompt))
        self.assert_exit(result, 2)
        self.assertIn("exactly match", result.stdout + result.stderr)
        self.assertFalse((self.output / "train.jsonl").exists())

    def test_audio_directory_argument_is_required(self):
        write_pcm(self.audio / "example.wav")
        self.write_rows([raw_row()])
        result = self.run_cli(include_audio=False)
        self.assert_exit(result, 2)
        self.assertIn("--audio-dir", result.stderr)
        self.assertIn("required", result.stderr)
        self.assertFalse(self.output.exists())

    def test_extra_segment_fields_do_not_change_legacy_text_or_audio(self):
        path = self.audio / "example.wav"
        write_pcm(path)
        row = raw_row()
        row.update(audio="does-not-exist.wav", srcs=["NEW SOURCE MUST NOT WIN"],
                   chooses=["新格式参考不得覆盖旧字段"], voice_end=False)
        self.write_rows([row])
        self.assert_exit(self.run_cli(), 0)
        task = json.loads(self.read_jsonl()[0]["flex_task"])
        self.assertEqual(task["audio_path"], str(path.resolve()))
        self.assertEqual(task["source_text"], row["en"]["txt"])
        self.assertEqual(task["reference_text"], row["zh"]["txt"])
        self.assertEqual(task["input_format"], "legacy")

    def test_segment_only_row_is_not_auto_detected_by_legacy_cli(self):
        path = self.audio / "example.wav"
        write_pcm(path)
        self.write_rows([{"audio": str(path), "srcs": ["A source."], "chooses": ["参考。"]}])
        self.assert_exit(self.run_cli(), 2)
        self.assertFalse((self.output / "train.jsonl").exists())
        self.assertEqual(self.report()["rejected_rows"], 1)


if __name__ == "__main__":
    unittest.main()
