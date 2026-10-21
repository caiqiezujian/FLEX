"""Strict, CPU-only bridge from prepared tasks and saved SFT sampling groups.

The two projects already share task schema version 1. This module preserves
provenance rather than synthesizing labels or source-word timestamps. Saved SFT
token probabilities are audit data, NEVER permission for offline GSPO replay.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator

from .data import validate_task
from .protocol import parse_response


class ArtifactError(ValueError):
    """A saved artifact is inconsistent, incomplete, or from a mixed run."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ArtifactError(f"Non-finite JSON number: {value}")


def strict_json_loads(text: str):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    try:
        return strict_json_loads(Path(path).read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"Cannot read JSON {path}: {exc}") from exc


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = strict_json_loads(line)
                if not isinstance(row, dict):
                    raise ArtifactError("JSONL row must be an object")
            except ValueError as exc:
                raise ArtifactError(f"{path}:{number}: {exc}") from exc
            yield number, row


def normalize_task(raw: dict) -> dict:
    """Preserve the common task schema; do not silently upgrade raw metadata."""
    task = deepcopy(raw)
    try:
        validate_task(task)
    except (ValueError, TypeError, KeyError) as exc:
        raise ArtifactError(f"Invalid prepared task (run FLEX-GSPO-Data/01_prepare.sh first): {exc}") from exc
    # Both Windows inspection and Linux execution must preserve the server path.
    if not (PurePosixPath(task["audio_path"]).is_absolute()
            or PureWindowsPath(task["audio_path"]).is_absolute()):
        raise ArtifactError(f"{task['id']}: audio_path must be absolute")
    if task.get("key") is not None and task["key"] != task["id"]:
        raise ArtifactError(f"{task['id']}: metadata key disagrees with task id")
    if task.get("split") not in (None, "train"):
        raise ArtifactError(f"{task['id']}: only training task manifests are accepted")
    return task


def load_training_tasks(path: str | Path) -> list[dict]:
    tasks, seen = [], set()
    for number, raw in iter_jsonl(path):
        try:
            task = normalize_task(raw)
        except ValueError as exc:
            raise ArtifactError(f"{path}:{number}: {exc}") from exc
        if task["id"] in seen:
            raise ArtifactError(f"{path}:{number}: duplicate task id {task['id']!r}")
        seen.add(task["id"])
        tasks.append(task)
    if not tasks:
        raise ArtifactError(f"No prepared tasks in {path}")
    return tasks


def discover_trajectory_files(values: list[str | Path]) -> list[Path]:
    """Directories discover only exact trajectories.jsonl, never groups/logs."""
    result, seen = [], set()
    for value in values:
        path = Path(value).expanduser().resolve()
        files = sorted(path.rglob("trajectories.jsonl")) if path.is_dir() else [path]
        if not files:
            raise ArtifactError(f"No trajectories.jsonl files found beneath {path}")
        for candidate in files:
            if not candidate.is_file():
                raise ArtifactError(f"Trajectory input is not a file: {candidate}")
            resolved = candidate.resolve()
            if resolved in seen:
                raise ArtifactError(f"Trajectory file selected more than once: {resolved}")
            seen.add(resolved)
            result.append(resolved)
    if not result:
        raise ArtifactError("At least one trajectory file is required")
    return sorted(result)


def inspect_sampler_runs(paths: list[Path], manifest_digest: str) -> tuple[dict, str]:
    """Worker run IDs differ, but actor/config/data identities must agree."""
    runs, common_hash, shards = {}, None, set()
    for path in paths:
        run_path = path.parent / "run.json"
        run = read_json(run_path)
        if not isinstance(run, dict) or not isinstance(run.get("signature"), dict):
            raise ArtifactError(f"{run_path}: missing sampler signature")
        signature = run["signature"]
        # Matches the sampler's historical serialization, not our compact hash.
        expected = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
        if run.get("run_id") != expected:
            raise ArtifactError(f"{run_path}: sampler run_id does not match signature")
        if signature.get("manifest_sha256") != manifest_digest:
            raise ArtifactError(f"{run_path}: tasks manifest content differs from sampler input")
        if signature.get("backend") not in ("vllm", "transformers", "hf"):
            raise ArtifactError(f"{run_path}: real sampler backend required; mock runs cannot be scored")
        group_size = signature.get("group_size")
        if type(group_size) is not int or group_size < 2:
            raise ArtifactError(f"{run_path}: group_size must be an integer >= 2")
        if type(signature.get("seed")) is not int or not 0 <= signature["seed"] < 2**32:
            raise ArtifactError(f"{run_path}: seed must be an unsigned 32-bit integer")
        if signature.get("offline_gspo_replay_ready") is not False:
            raise ArtifactError(f"{run_path}: unsupported replay provenance")
        shard = signature.get("shard_index")
        shard_count = signature.get("num_shards")
        if type(shard_count) is not int or type(shard) is not int or not 0 <= shard < shard_count:
            raise ArtifactError(f"{run_path}: invalid shard metadata")
        if shard in shards:
            raise ArtifactError(f"Duplicate sampler shard {shard}; do not combine repeated runs")
        shards.add(shard)
        # A continuation source is per-worker provenance, not a sampling
        # distribution setting. Its full original signature remains protected
        # by run_id/run_file_sha256; actor/config hashes are still compared.
        comparable = {key: value for key, value in signature.items()
                      if key not in ("shard_index", "audio_inventory", "continue_from")}
        current_hash = fingerprint(comparable)
        if common_hash is not None and common_hash != current_hash:
            raise ArtifactError("Mixed sampler model/config/data signatures across trajectory files")
        common_hash = current_hash
        inventory = signature.get("audio_inventory")
        if not isinstance(inventory, list):
            raise ArtifactError(f"{run_path}: missing audio inventory")
        by_id = {}
        for item in inventory:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ArtifactError(f"{run_path}: malformed audio inventory")
            if item["id"] in by_id:
                raise ArtifactError(f"{run_path}: duplicate audio inventory id")
            by_id[item["id"]] = item
        runs[path] = {"run_id": run["run_id"], "group_size": group_size,
                      "signature": signature, "inventory": by_id,
                      "run_file": str(run_path), "run_file_sha256": file_digest(run_path)}
    return runs, common_hash or ""


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactError(f"{label} must be a finite number")
    return float(value)


def validate_sampling_group(record: dict, task: dict, run: dict) -> None:
    """Reject corruption; retain genuinely invalid policy outputs for scoring."""
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ArtifactError("Unsupported trajectory group schema")
    if record.get("id") != task["id"] or record.get("run_id") != run["run_id"]:
        raise ArtifactError("Trajectory group id/run_id mismatch")
    if record.get("offline_gspo_replay_ready") is not False:
        raise ArtifactError("Saved SFT groups must not be marked GSPO-replay-ready")
    if record.get("purpose") != "sft_rollout_audit_and_reward_calibration":
        raise ArtifactError("Unsupported saved trajectory purpose")
    embedded = normalize_task(record.get("task"))
    if canonical_json(embedded) != canonical_json(task):
        raise ArtifactError("Embedded task differs from the selected prepared manifest")
    inventory = run["inventory"].get(task["id"])
    if inventory is None or inventory.get("path") != task["audio_path"]:
        raise ArtifactError("Task audio path/id disagrees with sampler inventory")
    candidates = record.get("trajectories")
    if not isinstance(candidates, list) or len(candidates) != run["group_size"]:
        raise ArtifactError("Incomplete candidate group; all original candidates are required")
    tolerance = 1.1 / task["sample_rate"]  # Real decoder endpoints can differ by one sample.
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or candidate.get("id") != task["id"]:
            raise ArtifactError("Candidate id mismatch")
        if type(candidate.get("group_index")) is not int or candidate["group_index"] != index:
            raise ArtifactError("Candidate indices must retain original complete order")
        # Match sampling.py stable_seed; duplicated candidate seeds would defeat
        # the independent-candidate provenance even when group_index differs.
        seed_text = f"{run['signature']['seed']}\0{task['id']}\0{index}".encode("utf-8")
        expected_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], "big")
        if type(candidate.get("seed")) is not int or candidate["seed"] != expected_seed:
            raise ArtifactError("Candidate seed disagrees with original sampler configuration")
        if candidate.get("mock") is not False or candidate.get("offline_gspo_replay_ready") is not False:
            raise ArtifactError("Mock or replay-ready candidates cannot enter real calibration")
        for key in ("valid", "complete_input", "final_wait"):
            if type(candidate.get(key)) is not bool:
                raise ArtifactError(f"Candidate {key} must be boolean")
        turns = candidate.get("turns")
        if not isinstance(turns, list) or not turns or len(turns) > len(task["chunks"]):
            raise ArtifactError("Candidate turns must be a nonempty prefix of real audio chunks")
        if candidate["complete_input"] != (len(turns) == len(task["chunks"])):
            raise ArtifactError("complete_input disagrees with consumed audio turns")
        if candidate["valid"] and not candidate["complete_input"]:
            raise ArtifactError("A valid candidate cannot silently omit final audio chunks")
        deltas, previous_end, final_tag = [], 0.0, None
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or type(turn.get("turn_id")) is not int or turn["turn_id"] != turn_index:
                raise ArtifactError("Turn ids must be chronological contiguous integers")
            chunk = task["chunks"][turn_index]
            for field, expected in (("audio_start_seconds", chunk["start_seconds"]),
                                    ("audio_end_seconds", chunk["end_seconds"]),
                                    ("input_seconds", chunk["input_seconds"])):
                if not math.isclose(_finite(turn.get(field), field), expected, rel_tol=0, abs_tol=tolerance):
                    raise ArtifactError(f"Turn {turn_index} {field} disagrees with real audio schedule")
            if not math.isclose(turn["audio_start_seconds"], previous_end, rel_tol=0, abs_tol=1e-8):
                raise ArtifactError("Turn audio intervals have a gap or overlap")
            previous_end = turn["audio_end_seconds"]
            if type(turn.get("committed")) is not bool or not isinstance(turn.get("response"), str):
                raise ArtifactError("Turn must retain raw response and boolean committed status")
            if not turn["committed"]:
                if candidate["valid"] or turn_index != len(turns) - 1:
                    raise ArtifactError("Only the last turn of an invalid candidate may be uncommitted")
                continue
            try:
                tag, delta = parse_response(turn["response"])
            except ValueError as exc:
                raise ArtifactError(f"Committed turn has invalid protocol: {exc}") from exc
            if turn.get("tag") != tag or turn.get("delta") != delta:
                raise ArtifactError("Saved tag/delta differ from the actual raw model response")
            deltas.append(delta)
            final_tag = tag
        if candidate.get("translation") != "".join(deltas):
            raise ArtifactError("Full translation differs from append-only committed deltas")
        if candidate["final_wait"] != (final_tag == "WAIT"):
            raise ArtifactError("final_wait disagrees with final committed response")


def iter_validated_groups(paths: list[Path], tasks_by_id: dict, runs: dict):
    """Stream groups, retaining only an ID set rather than token-heavy episodes."""
    seen = set()
    for path in paths:
        for line, record in iter_jsonl(path):
            task_id = record.get("id")
            if not isinstance(task_id, str) or task_id not in tasks_by_id:
                raise ArtifactError(f"{path}:{line}: unknown task id")
            if task_id in seen:
                raise ArtifactError(f"{path}:{line}: duplicate trajectory group id {task_id!r}")
            seen.add(task_id)
            try:
                validate_sampling_group(record, tasks_by_id[task_id], runs[path])
            except (ValueError, TypeError) as exc:
                raise ArtifactError(f"{path}:{line} id={task_id}: {exc}") from exc
            yield path, line, record
