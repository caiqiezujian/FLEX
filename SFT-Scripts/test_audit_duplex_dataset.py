"""Exercise the audit against real WAV headers and the training reader."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_duplex_dataset as audit
from duplex_audio import chunk_wav_bytes, encode_chunk_ref


def source_file(tmp_path, frames=20000, rate=16000):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(frames, dtype=np.float32), rate, subtype="FLOAT")
    return path


def row_for(path, intervals=((0, 1, 1), (1, 1.25, 1))):
    refs = [encode_chunk_ref(path, *part) for part in intervals]
    messages = [{"role": "system", "content": "Translate incrementally."}]
    for _ in refs:
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": "<|DIRECT|>新增译文"}])
    return {"id": "case", "audios": refs, "messages": messages,
            "audio_chunk_durations": [p[2] for p in intervals]}


def check(row, **kwargs):
    return audit.audit_line(12, json.dumps(row).encode(), **kwargs)


def test_tail_padding_and_decode_pass(tmp_path):
    path = source_file(tmp_path)
    result = check(row_for(path), decode=True)
    assert not result["issue_counts"]
    assert result["chunks_checked"] == result["chunks_decoded"] == 2
    assert result["chunks_failed"] == 0


def test_rounding_suspect_still_fails_like_reader(tmp_path):
    path = source_file(tmp_path, frames=16001)
    actual = 16001 / 16000
    row = row_for(path, ((0, 1, 1), (1, round(actual, 6), 1)))
    result = check(row)
    assert dict(result["issue_counts"]) == {"bounds_seconds_only": 1}
    detail = result["errors"][0]
    assert detail["end_excess_seconds"] > 1e-8
    assert detail["end_excess_frames"] == 0
    with pytest.raises(ValueError, match="Chunk bounds"):
        chunk_wav_bytes(row["audios"][-1])


@pytest.mark.parametrize("interval", [(0, 1.5, 2), (0, 0.000001, 1)])
def test_real_overflow_or_empty_frame_range_matches_reader(tmp_path, interval):
    row = row_for(source_file(tmp_path), (interval,))
    result = check(row)
    assert result["issue_counts"]["bounds_frames"] == 1
    with pytest.raises(ValueError, match="Chunk bounds"):
        chunk_wav_bytes(row["audios"][0])


def test_many_bad_chunks_counted_despite_detail_limit(tmp_path):
    row = row_for(source_file(tmp_path), ((0, 2, 2), (2, 3, 1), (3, 4, 1)))
    result = check(row, max_details=1)
    assert result["chunks_checked"] == result["chunks_failed"] == 3
    assert result["issue_counts"]["bounds_frames"] == 3
    assert len(result["errors"]) == 1


def test_missing_audio_is_counted(tmp_path):
    result = check(row_for(tmp_path / "missing.wav"))
    assert result["issue_counts"]["missing_audio"] == 2
    assert result["chunks_failed"] == 2


def test_invalid_path_and_huge_duration_do_not_abort_audit(tmp_path):
    row = row_for(str(tmp_path / "bad") + "\x00.wav")
    assert check(row)["issue_counts"]["audio_stat_error"] == 2
    row = row_for(source_file(tmp_path))
    row["audio_chunk_durations"][0] = 10 ** 400
    assert check(row)["issue_counts"]["duration_mismatch"] == 1


def test_headers_and_real_pcm_integrity_have_distinct_scope(tmp_path):
    path = source_file(tmp_path)
    pcm = np.zeros(20000, dtype=np.float32)
    pcm[0] = np.nan
    sf.write(path, pcm, 16000, subtype="FLOAT")
    row = row_for(path)
    assert not check(row)["issue_counts"]
    assert check(row, decode=True)["issue_counts"]["decode_error"] == 1


def test_invalid_json_schema_ref_protocol_and_duration(tmp_path):
    assert audit.audit_line(1, b"{bad}")["issue_counts"] == {"invalid_json": 1}
    assert check([])["issue_counts"] == {"invalid_schema": 1}
    row = row_for(source_file(tmp_path))
    row["audios"][0] = "broken-reference"
    row["messages"][2]["content"] = "<|WAIT|>should not have text"
    row["audio_chunk_durations"][1] = float("nan")
    result = check(row)
    assert result["issue_counts"]["invalid_ref"] == 1
    assert result["issue_counts"]["invalid_protocol"] == 1
    assert result["issue_counts"]["duration_mismatch"] == 1


def test_typed_prepare_format(tmp_path):
    row = row_for(source_file(tmp_path))
    typed = {"id": "typed", "system": "Translate", "messages": [],
             "audio_chunk_durations": row["audio_chunk_durations"]}
    for ref in row["audios"]:
        typed["messages"].extend([
            {"role": "user", "content": [{"type": "audio", "audio": ref}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<|WAIT|>"}]},
        ])
    result = check(typed)
    assert not result["issue_counts"]
    assert result["row_format"] == "typed"


def test_report_counts_and_sources_unchanged(tmp_path, capsys):
    source = source_file(tmp_path)
    dataset = tmp_path / "train.jsonl"
    valid = row_for(source)
    invalid = row_for(source, ((0, 2, 2),))
    dataset.write_text(json.dumps(valid) + "\n\n{bad}\n" + json.dumps(invalid) + "\n", encoding="utf-8-sig")
    original, audio_bytes = dataset.read_bytes(), source.read_bytes()
    report = tmp_path / "audit"
    args = [str(dataset), "--output-dir", str(report), "--workers", "2", "--progress-every", "1"]
    assert audit.main(args) == 1
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["full_dataset_checked"]
    assert summary["rows_checked"] == 3
    assert summary["rows_passed"] == 1 and summary["rows_failed"] == 2
    assert summary["blank_lines"] == 1
    assert summary["reason_row_counts"] == {"invalid_json": 1, "bounds_frames": 1}
    issues = [json.loads(line) for line in (report / "invalid.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(issue["line"] for issue in issues) == [3, 4]
    assert dataset.read_bytes() == original and source.read_bytes() == audio_bytes
    assert "[audit] checked=" in capsys.readouterr().out
    summary_bytes = (report / "summary.json").read_bytes()
    assert audit.main(args) == 2  # Never overwrite an existing report.
    assert (report / "summary.json").read_bytes() == summary_bytes


def test_empty_dataset_is_not_a_success(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_bytes(b"\n")
    report = tmp_path / "empty-report"
    assert audit.main([str(dataset), "--output-dir", str(report)]) == 2
    assert not json.loads((report / "summary.json").read_text())["full_dataset_checked"]
