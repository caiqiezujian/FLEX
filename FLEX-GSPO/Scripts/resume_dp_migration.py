#!/usr/bin/env python3
"""Resume a V4 native checkpoint on four GPUs using its exact historical code.

This helper lives outside the training source fingerprint. It permits only the
explicit DP4 -> DP2 / vLLM replicas2 -> replicas1 transition; all other identity
fields still have to match. No optimizer reset, merged-weight warm start, reward
change, checkpoint rewrite, or edit to the main checkout is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


LAYOUT8 = dict(world_size=8, tensor_parallel_size=2, expert_parallel_size=2,
               pipeline_parallel_size=1, context_parallel_size=1,
               data_parallel_size=4, vllm_tensor_parallel_size=4, vllm_replicas=2)
LAYOUT4 = {**LAYOUT8, "world_size": 4, "data_parallel_size": 2, "vllm_replicas": 1}
IDENTITY_FIELDS = {"schema", "framework_commit", "model_path", "parallelism",
                   "model_metadata_sha256", "weight_file_metadata", "dataset_sha256",
                   "dataset_tasks", "reward_config_sha256", "settings", "code_sha256"}


def validate_dp_migration(original: dict, current: dict) -> None:
    for value in (original, current):
        if not isinstance(value, dict) or not IDENTITY_FIELDS.issubset(value):
            raise ValueError("Missing original/current run identity fields")
    if original["parallelism"] != LAYOUT8 or current["parallelism"] != LAYOUT4:
        raise ValueError("Only eight-to-four GPU DP migration with TP2/EP2/PP1/CP1 is supported")
    before, after = copy.deepcopy(original), copy.deepcopy(current)
    before.pop("parallelism")
    after.pop("parallelism")
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise ValueError("Migration must retain original model/data/reward/code/batch/LR horizon; changed: "
                         + ", ".join(changed))


def flag_value(argv, flag, default=None):
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError("Original launch argv must be a string array")
    positions = [i for i, value in enumerate(argv) if value == flag]
    if not positions:
        if default is not None:
            return default
        raise ValueError("Original launch missing " + flag)
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError("Ambiguous original launch argument " + flag)
    return argv[positions[0] + 1]


def build_resume_argv(document, checkpoint: Path, output: Path, gpu_ids: str,
                      reward_config: Path) -> list[str]:
    settings = document["run_identity"]["settings"]
    if not isinstance(settings, dict) or not isinstance(settings.get("model"), str):
        raise ValueError("Original launch has no model/settings")
    if document.get("reward_settings", {}).get("version") != 4:
        raise ValueError("This migration helper is for the original COMET22 V4 run only")
    if document.get("reward_settings", {}).get("quality_provider") != "comet22":
        raise ValueError("Original run must use COMET22")
    # Keep the original SFT base: using the merged policy here would change KL.
    argv = [settings["model"], flag_value(document["argv"], "--dataset"), str(output)]
    for key, value in settings.items():
        if key == "model" or value is None:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or isinstance(value, (dict, list, bool)):
            raise ValueError("Unsupported saved launcher setting: " + str(key))
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    argv.extend(["--gpu-ids", gpu_ids, "--resume-from", str(checkpoint),
                 "--reward-config", str(reward_config),
                 "--save-steps", flag_value(document["argv"], "--save_steps", "500"),
                 "--dataset-num-proc", flag_value(document["argv"], "--dataset_num_proc", "4"),
                 "--performance-log", document.get("performance", {}).get("phase_logging", "on"),
                 "--rollout-log-every", str(document.get("environment", {}).get("FLEX_ROLLOUT_LOG_EVERY", "5"))])
    return argv


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return result.stdout


def validate_source_hashes(expected):
    if not isinstance(expected, dict) or not {"train_megatron.py", "swift_plugin.py"}.issubset(expected):
        raise ValueError("Original launch lacks training source fingerprints")
    for name, digest in expected.items():
        path = PurePosixPath(name)
        allowed = (name in {"train_megatron.py", "swift_plugin.py"} or
                   (len(path.parts) == 2 and path.parts[0] == "flex_gspo" and path.suffix == ".py"))
        if not allowed or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid training source fingerprint")


def find_original_commit(repo: Path, expected: dict, explicit=None):
    validate_source_hashes(expected)
    if explicit:
        revisions = [git(repo, "rev-parse", "--verify", explicit + "^{commit}").decode().strip()]
    else:
        revisions = git(repo, "rev-list", "--all", "--max-count=300", "--", "FLEX-GSPO").decode().splitlines()
    blob_hashes = {}
    for index, revision in enumerate(revisions):
        if index % 20 == 0:
            print(f"[migration] Matching historical source: {index + 1}/{len(revisions)}", flush=True)
        entries = {}
        for entry in git(repo, "ls-tree", "-r", "-z", revision, "--", "FLEX-GSPO").split(b"\0"):
            if not entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            path = raw_path.decode("utf-8").removeprefix("FLEX-GSPO/")
            if path in ("train_megatron.py", "swift_plugin.py") or (
                    path.startswith("flex_gspo/") and path.count("/") == 1 and path.endswith(".py")):
                mode, kind, blob = header.decode().split()
                if kind != "blob" or mode not in ("100644", "100755"):
                    raise ValueError("Historical training source is not a regular file")
                entries[path] = blob
        if set(entries) != set(expected):
            continue
        matches = True
        for path, blob in entries.items():
            if blob not in blob_hashes:
                blob_hashes[blob] = hashlib.sha256(git(repo, "cat-file", "blob", blob)).hexdigest()
            if blob_hashes[blob] != expected[path]:
                matches = False
                break
        if matches:
            return revision
    raise ValueError("No historical Git version matches the saved training code. Preserve the original "
                     "launch JSON and recover that code; this helper does not bypass source differences.")


def historical_root(repo: Path, commit: str, expected: dict) -> Path:
    # A separate worktree preserves the user's current checkout and old outputs.
    target = repo.parent / (repo.name + "-resume-v4-" + commit[:12])
    if not target.exists():
        print(f"[migration] Creating historical worktree: {target}", flush=True)
        git(repo, "worktree", "add", "--detach", str(target), commit)
    if git(target, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("Historical worktree exists at a different commit")
    root = target / "FLEX-GSPO"
    paths = [root / "train_megatron.py", root / "swift_plugin.py", *sorted((root / "flex_gspo").glob("*.py"))]
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if actual != expected:
        raise ValueError("Historical worktree differs from original run source; no files overwritten")
    return root


def select_reward_config(document, root):
    original = document.get("environment", {}).get("FLEX_REWARD_CONFIG")
    candidates = [Path(original)] if original else []
    candidates.append(root / "configs" / "qalco_v4.json")
    expected = document["run_identity"]["reward_config_sha256"]
    for path in candidates:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path.resolve()
    raise ValueError("Cannot find the original reward config; restore its exact bytes, not a new reward")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-json", required=True, type=Path, help="Original eight-GPU launch_*.json")
    parser.add_argument("--resume-from", required=True, type=Path, help="Native checkpoint-500, never -merged")
    parser.add_argument("--output-dir", required=True, type=Path, help="New migration output directory")
    parser.add_argument("--gpu-ids", required=True, help="Four actually available GPU IDs, e.g. 0,2,4,6")
    parser.add_argument("--master-port", type=int, default=29583, help="Unused torchrun rendezvous port")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-commit", help="Optional exact historical commit; still checked against source hashes")
    parser.add_argument("--dry-run", action="store_true", help="Print migration plan without worktree/model/service changes")
    options = parser.parse_args(argv)
    try:
        document = json.loads(options.launch_json.read_text(encoding="utf-8-sig"))
        original = document["run_identity"]
        expected_new = {**original, "parallelism": LAYOUT4}
        validate_dp_migration(original, expected_new)
        if document.get("reward_settings", {}).get("version") != 4 or document["reward_settings"].get("quality_provider") != "comet22":
            raise ValueError("The selected launch must be the original COMET22 V4 training run")
        ids = options.gpu_ids.split(",")
        if len(ids) != 4 or len(set(ids)) != 4 or any(not x.isdigit() for x in ids):
            raise ValueError("Specify exactly four distinct available GPU IDs")
        checkpoint = options.resume_from.resolve()
        launch = options.launch_json.resolve()
        if launch.parent not in list(checkpoint.parents)[:3]:
            raise ValueError("Use the original launch JSON beside this checkpoint/run (within three parent levels)")
        tracker = checkpoint / "latest_checkpointed_iteration.txt"
        iteration = int(tracker.read_text().strip())
        if checkpoint.name.endswith("-merged") or iteration <= 0:
            raise ValueError("A complete native training checkpoint is required, not merged weights")
        payload = checkpoint / f"iter_{iteration:07d}"
        if not (payload / "common.pt").is_file() or not any(
                p.is_file() and p.name not in {"common.pt", "metadata.json", ".metadata"}
                and p.stat().st_size > 0 for p in payload.rglob("*")):
            raise ValueError("Native checkpoint lacks common.pt or model/optimizer shard payloads")
        saved_args = json.loads((checkpoint / "args.json").read_text(encoding="utf-8-sig"))
        if saved_args.get("no_save_optim", True) or saved_args.get("no_save_rng", True):
            raise ValueError("Checkpoint did not preserve optimizer/RNG; cannot perform native migration")
        for key in ("global_batch_size", "train_iters", "lora_rank"):
            if saved_args.get(key) != original["settings"].get(key):
                raise ValueError("Selected launch and checkpoint args disagree on " + key)
        output = options.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty output directory; the original run is preserved")
        repo = Path(git(options.repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        commit = find_original_commit(repo, original["code_sha256"], options.source_commit)
        if not 1024 <= options.master_port <= 65535:
            raise ValueError("master-port must be in 1024..65535")
        plan = {"migration": "native_v4_world8_to_world4_dp4_to_dp2", "original_launch": str(launch),
                "migration_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_commit": commit, "checkpoint": str(checkpoint), "restored_iteration": iteration,
                "original_parallelism": original["parallelism"], "new_parallelism": LAYOUT4,
                "global_batch_size": original["settings"].get("global_batch_size"),
                "total_train_iters": original["settings"].get("train_iters"),
                "gpu_ids": options.gpu_ids, "output": str(output),
                "optimizer_scheduler_rng": "load native state; no bitwise reproducibility promise",
                "kl_reference": "original SFT model retained"}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if options.dry_run:
            print("[migration] Plan only; checkpoint payloads/framework/GPU loading are not verified.", flush=True)
            return 0
        root = historical_root(repo, commit, original["code_sha256"])
        reward_config = select_reward_config(document, root)
        # torchrun and its `python -m` workers put cwd before PYTHONPATH.
        # Keep both the parent and workers anchored to the historical source,
        # otherwise a current flex_gspo package could shadow the checked files.
        for value in (original["settings"]["model"], flag_value(document["argv"], "--dataset")):
            if not Path(value).is_absolute():
                raise ValueError("Migration requires original absolute model/dataset paths")
        os.chdir(root)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("flex_original_launcher", root / "train_megatron.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        forwarded = build_resume_argv(document, checkpoint, output, options.gpu_ids, reward_config)
        forwarded.extend(["--master-port", str(options.master_port)])

        def verify_migration(args, identity):
            # Called by the ORIGINAL launcher after its native checkpoint,
            # optimizer/RNG, model, dataset, and prompt preflight succeeds.
            if Path(args.resume_from).resolve() != checkpoint or Path(args.output).resolve() != output:
                raise ValueError("Migration checkpoint/output mismatch")
            validate_dp_migration(original, identity)
            # Persistent evidence for the deliberate layout change. Native
            # launcher then writes its normal new run/session identity.
            output.mkdir(parents=True, exist_ok=True)
            with (output / "dp_migration.json").open("x", encoding="utf-8") as handle:
                json.dump({**plan, "original_launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                           "original_identity": original, "new_identity": identity}, handle, ensure_ascii=False, indent=2)
            print("[migration] Verified: only DP/world size and rollout replica count changed. "
                  "Loading native optimizer/scheduler/RNG; global batch and training horizon retained.", flush=True)

        # This explicit migration entry changes only the identity predicate,
        # never the trainer's model/optimizer algorithms or checkpoint contents.
        launcher.verify_resume_identity = verify_migration
        return launcher.main(forwarded)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"[migration] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
