#!/usr/bin/env python3
"""Score saved grouped SFT candidates for reward calibration, NOT GSPO replay."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from flex_gspo.artifacts import (
    ArtifactError, canonical_json, discover_trajectory_files, file_digest, fingerprint,
    inspect_sampler_runs, iter_validated_groups, load_training_tasks, read_json,
)

LOG = logging.getLogger("flex.qalco.score")


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def heartbeat(label, seconds):
    done = threading.Event()
    started = time.monotonic()
    def report():
        while not done.wait(seconds):
            LOG.info("Still running: %s | elapsed=%.1fs", label, time.monotonic() - started)
    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ArtifactError("Scorer returned a non-finite/non-numeric metric")
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self):
        return {"count": self.n, "mean": self.mean if self.n else None,
                "std": math.sqrt(max(0, self.m2 / self.n)) if self.n else None,
                "min": self.minimum if self.n else None, "max": self.maximum if self.n else None}


def _check_scores(scores, count):
    if not isinstance(scores, list) or len(scores) != count:
        raise ArtifactError("score_group must return one score per original candidate")
    for score in scores:
        if not isinstance(score, dict) or not isinstance(score.get("components"), dict):
            raise ArtifactError("score_group returned malformed score/components")
        reward = score.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ArtifactError("score_group returned a non-finite reward")
        if type(score["components"].get("feasible")) is not bool:
            raise ArtifactError("score_group must report absolute boolean feasibility")
        canonical_json(score)


def scorer_identity(config):
    """Client code and declared-service provenance; no service requests."""
    from flex_gspo.metricx import validate_metricx_settings
    settings = validate_metricx_settings(config["reward"]["metricx"])
    package = Path(__file__).resolve().parent / "flex_gspo"
    code = {name: file_digest(package / name) for name in
            ("qalco.py", "metricx.py", "metricx_service.py", "rewards.py", "protocol.py", "artifacts.py", "runtime.py")}
    code["score_candidates.py"] = file_digest(__file__)
    identity = {
        "backend": "service", "service_url": settings["service_url"],
        "service_id": settings["service_id"], "mode": settings["mode"],
        "implementation_sha256": code,
        "weight_identity_basis": "declared_service_identity_not_verified_remote_weights",
        "metadata_and_tokenizer_assets_content_hashed": False,
    }
    return {"sha256": fingerprint(identity), "details": identity}


def run_scoring(tasks_path, trajectory_inputs, config_path, output_dir=None, *, limit=None,
                validate_only=False, check_audio=False, log_every=20, log_interval=5.0,
                scorer_factory=None):
    """A scorer_factory injection is for CPU tests only; the CLI has no fake scorer."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ArtifactError("--limit must be positive")
    if type(log_every) is not int or log_every < 1:
        raise ArtifactError("--log-every must be positive")
    if not math.isfinite(log_interval) or log_interval <= 0:
        raise ArtifactError("--log-interval must be finite and positive")
    started = time.monotonic()
    LOG.info("Preflight: validating prepared tasks and grouped SFT trajectory artifacts")
    with heartbeat("task/run/group validation (no model loaded)", log_interval):
        tasks = load_training_tasks(tasks_path)
        by_id = {task["id"]: task for task in tasks}
        files = discover_trajectory_files(trajectory_inputs)
        manifest_sha = file_digest(tasks_path)
        runs, sampler_fingerprint = inspect_sampler_runs(files, manifest_sha)
        input_hashes = {str(path): file_digest(path) for path in files}
        config = read_json(config_path)
        if not isinstance(config, dict) or not isinstance(config.get("reward"), dict):
            raise ArtifactError("config must contain the full reward configuration")
        if config["reward"].get("provider") != "qalco":
            raise ArtifactError("Offline calibration requires reward.provider=qalco")
        metricx_backend = "service"
        selected_ids, available_ids = [], []
        candidate_count = 0
        # Validate ALL supplied groups even if --limit selects fewer for scoring.
        for _, _, record in iter_validated_groups(files, by_id, runs):
            available_ids.append(record["id"])
            if len(available_ids) == 1 or len(available_ids) % log_every == 0:
                LOG.info("Validated groups=%d", len(available_ids))
        if not available_ids:
            raise ArtifactError("No complete saved candidate groups found")
        selected_ids = sorted(available_ids)[:limit] if limit else sorted(available_ids)
        selected = set(selected_ids)
        candidate_count = len(selected) * next(iter(runs.values()))["group_size"]
        if check_audio:
            for task_id in selected_ids:
                if not Path(by_id[task_id]["audio_path"]).is_file():
                    raise ArtifactError(f"{task_id}: audio file does not exist on this host")
        if scorer_factory is None:
            from flex_gspo.rewards import validate_reward_setup
            validate_reward_setup([by_id[key] for key in selected_ids], config)
            quality_identity = scorer_identity(config)
        else:
            quality_identity = {"test_injection": True, "real_model_identity": None}
    service_ready = None
    if scorer_factory is None:
        from flex_gspo.metricx import ensure_metricx_ready
        from flex_gspo.rewards import RewardError
        try:
            ensure_metricx_ready(config)
        except RewardError:
            service_ready = False
        else:
            service_ready = True
    provenance = {
        "tasks_path": str(Path(tasks_path).resolve()), "tasks_sha256": manifest_sha,
        "trajectory_files_sha256": input_hashes,
        "sampler_compatible_signature_sha256": sampler_fingerprint,
        "sampler_runs": [{"file": str(path), "run_id": runs[path]["run_id"],
                          "run_file": runs[path]["run_file"],
                          "run_file_sha256": runs[path]["run_file_sha256"],
                          "continue_from": runs[path]["signature"].get("continue_from")} for path in files],
        "reward_config": config["reward"], "reward_config_sha256": fingerprint(config["reward"]),
        "quality_scorer_identity": quality_identity,
        "selected_ids_sha256": fingerprint(selected_ids),
        "quality_scoring": ("injected_test_scorer" if scorer_factory is not None else "service_metricx"),
        "audio_files_existence_checked": check_audio,
        "audio_contents_decoded": False,
    }
    validated = {"valid": True, "manifest_tasks": len(tasks), "available_groups": len(available_ids),
                 "selected_groups": len(selected_ids), "selected_candidates": candidate_count,
                 "manifest_tasks_without_candidates": len(set(by_id) - set(available_ids)),
                 "model_loaded": False, "output_written": False,
                 "quality_service_ready": service_ready,
                 "offline_gspo_replay_ready": False, "provenance": provenance}
    if validate_only:
        LOG.info("Validation complete: selected=%d groups, %d candidates; no model loaded", len(selected), candidate_count)
        return validated
    if scorer_factory is None:
        from flex_gspo.metricx import ensure_metricx_ready
        # Check client configuration/dependency before creating output artifacts.
        ensure_metricx_ready(config)
    if output_dir is None:
        raise ArtifactError("--output-dir is required unless --validate-only")
    output = Path(output_dir).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError("Use a NEW output directory. Calibration resume/overwrite is intentionally unsupported.")
    # Do not permit placing generated files inside source run directories.
    resolved_output = output.resolve()
    for source_root in {path.parent for path in files}:
        if resolved_output == source_root or source_root in resolved_output.parents:
            raise ArtifactError("Calibration output must be outside sampler run directories")
    output.mkdir(parents=True, exist_ok=False)
    (output / "groups").mkdir()
    run_record = {"schema_version": 1, "purpose": "qalco_reward_calibration_not_training",
                  "offline_gspo_replay_ready": False, "status": "running", "provenance": provenance,
                  "resume_supported": False, "completed_groups": 0}
    atomic_json(output / "run.json", run_record)
    stats = {key: Moments() for key in ("reward", "quality", "laal_seconds", "raw_laal_seconds", "length_penalty")}
    tiers, translation_counts, action_counts, feasible_counts = Counter(), Counter(), Counter(), Counter()
    completed, total_candidates, feasible = 0, 0, 0
    try:
        if scorer_factory is None:
            from flex_gspo.rewards import build_scorer
            scorer_factory = build_scorer
        LOG.info("Initializing quality scorer backend=%s; no actor model is loaded", metricx_backend)
        with heartbeat("initializing quality scorer", log_interval):
            scorer = scorer_factory(config)
        LOG.info("Quality scorer ready; scoring %d groups / %d candidates", len(selected), candidate_count)
        # A second streaming pass avoids keeping token-heavy trajectories in RAM.
        for source_path, line, record in iter_validated_groups(files, by_id, runs):
            if record["id"] not in selected:
                continue
            task = by_id[record["id"]]
            candidates = record["trajectories"]
            with heartbeat(f"scoring group {completed + 1}/{len(selected)} id={task['id']}", log_interval):
                scores = scorer.score_group(task, candidates)
            _check_scores(scores, len(candidates))
            scored = dict(record)
            scored["purpose"] = "qalco_reward_calibration_not_training"
            scored["calibration"] = {"reward_config_sha256": provenance["reward_config_sha256"],
                                     "quality_scorer_identity": provenance["quality_scorer_identity"].get("sha256"),
                                     "source_file": str(source_path), "source_line": line,
                                     "source_record_sha256": fingerprint(record)}
            scored["trajectories"] = [dict(candidate, score=score) for candidate, score in zip(candidates, scores)]
            filename = fingerprint(task["id"]) + ".json"
            atomic_json(output / "groups" / filename, scored)
            group_feasible = 0
            for score in scores:
                components = score["components"]
                stats["reward"].add(score["reward"])
                for name in stats:
                    if name != "reward":
                        stats[name].add(components.get(name))
                tiers[components.get("tier", "unspecified")] += 1
                group_feasible += int(components["feasible"])
            feasible += group_feasible
            feasible_counts[str(group_feasible)] += 1
            translation_counts[str(len({c["translation"] for c in candidates}))] += 1
            # Label-only variations are recorded separately, not called timing diversity.
            schedules = {tuple((t["audio_end_seconds"], bool(t.get("delta", "").strip()))
                               for t in c["turns"]) for c in candidates}
            action_counts[str(len(schedules))] += 1
            completed += 1
            total_candidates += len(candidates)
            run_record["completed_groups"] = completed
            atomic_json(output / "run.json", run_record)
            if completed == 1 or completed % log_every == 0 or completed == len(selected):
                LOG.info("Scored groups=%d/%d candidates=%d/%d feasible=%d/%d elapsed=%.1fs",
                         completed, len(selected), total_candidates, candidate_count,
                         feasible, total_candidates, time.monotonic() - started)
        # Detect source artifacts changed during the run; never publish success for mixed inputs.
        for path, expected in input_hashes.items():
            if file_digest(path) != expected:
                raise ArtifactError(f"Input trajectory file changed during scoring: {path}")
        if file_digest(tasks_path) != manifest_sha:
            raise ArtifactError("Task manifest changed during scoring")
        for run in runs.values():
            if file_digest(run["run_file"]) != run["run_file_sha256"]:
                raise ArtifactError("Sampler run.json changed during scoring")
        if not quality_identity.get("test_injection") and scorer_identity(config) != quality_identity:
            raise ArtifactError("Quality scorer identity/code inventory changed during scoring")
        with (output / "scored.trajectories.jsonl.tmp").open("x", encoding="utf-8", newline="\n") as out:
            for task_id in selected_ids:
                group = read_json(output / "groups" / (fingerprint(task_id) + ".json"))
                out.write(canonical_json(group) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(output / "scored.trajectories.jsonl.tmp", output / "scored.trajectories.jsonl")
        summary = {
            **validated, "model_loaded": False, "output_written": True,
            "quality_scorer_initialized": True,
            "completed_groups": completed, "candidates": total_candidates,
            "feasible_candidates": feasible, "feasible_fraction": feasible / total_candidates,
            "all_infeasible_groups": feasible_counts.get("0", 0),
            "feasible_candidates_per_group": dict(feasible_counts), "tiers": dict(tiers),
            "metrics": {name: value.result() for name, value in stats.items()},
            "unique_translation_counts": dict(translation_counts),
            "unique_emission_schedule_counts": dict(action_counts),
            "elapsed_seconds": time.monotonic() - started,
            "notes": [
                "All original candidates retained, including invalid policy outputs; no best-only filtering.",
                "Scores are reward proxies, not human quality labels or causal WAIT values.",
                "Saved SFT sampling probabilities are never imported into the GSPO optimizer.",
                "No waveform decoding or prefix semantic support audit was performed.",
                "Length penalty is an over-budget proxy, not a semantic redundancy detector.",
            ],
        }
        atomic_json(output / "summary.json", summary)
        run_record.update(status="complete", summary="summary.json")
        atomic_json(output / "run.json", run_record)
        return summary
    except BaseException as exc:
        run_record.update(status="failed", completed_groups=completed,
                          failure_type=type(exc).__name__,
                          recovery="Completed groups preserved for inspection. Use a new output directory to rerun.")
        atomic_json(output / "run.json", run_record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Existing prepared train.tasks.jsonl")
    parser.add_argument("--trajectories", nargs="+", required=True,
                        help="Explicit JSONL files or sampler root directories (discovers only trajectories.jsonl)")
    parser.add_argument("--config", required=True, help="Full QALCO JSON config; only reward settings are loaded")
    parser.add_argument("--output-dir", help="NEW directory; no overwrite or resume")
    parser.add_argument("--limit", type=int, help="Score first N available task ids after sorting; validate all input groups")
    parser.add_argument("--validate-only", action="store_true", help="Validate artifacts/config without loading models or writing")
    parser.add_argument("--check-audio", action="store_true", help="Also check audio file existence on this host (no decoding)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--log-interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = run_scoring(args.tasks, args.trajectories, args.config, args.output_dir,
                              limit=args.limit, validate_only=args.validate_only, check_audio=args.check_audio,
                              log_every=args.log_every, log_interval=args.log_interval)
    except Exception:
        LOG.exception("Calibration failed; no training update was performed")
        return 1
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
