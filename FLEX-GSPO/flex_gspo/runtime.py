"""Small deterministic runtime helpers; no model imports or network access."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import threading
import time

LOG = logging.getLogger("flex.gspo")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def directory_inventory(path):
    """Metadata hashes + weight stats, NOT a claim to hash 60 GB of weights."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"local model directory missing: {root}")
    return {
        "path": str(root),
        "metadata": {p.name: file_sha256(p) for p in sorted(root.iterdir())
                     if p.is_file() and (p.suffix in (".json", ".model", ".tiktoken")
                                         or p.name in ("merges.txt", "vocab.txt"))},
        "weights": [{"name": p.name, "bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns}
                    for p in sorted(root.iterdir())
                    if p.is_file() and p.suffix in (".safetensors", ".bin")],
    }


def run_identity(config, tasks, world_size):
    """Resume requires the same task selection, algorithm, reward and runtime code."""
    settings = json.loads(json.dumps(config))
    for name in ("output_dir", "resume_from", "allow_uncalibrated_reward"):
        settings.pop(name, None)
    training = settings.setdefault("train", {})
    for name in ("max_updates", "epochs", "save_every", "log_interval_seconds"):
        training.pop(name, None)
    package = Path(__file__).resolve().parent
    implementation = {p.name: file_sha256(p) for p in sorted(package.glob("*.py"))}
    implementation["train.py"] = file_sha256(package.parent / "train.py")
    models = {"actor": directory_inventory(config["model_path"])}
    if config.get("reward", {}).get("provider") == "qalco":
        from .metricx import validate_metricx_settings
        metric = validate_metricx_settings(config["reward"]["metricx"])
        # This identifies the configured service, not its remotely hosted weights.
        models["metricx_service"] = {
            "backend": "service", "service_url": metric["service_url"],
            "service_id": metric["service_id"], "mode": metric["mode"],
            "identity_basis": "declared_service_identity_not_verified_remote_weights",
        }
    identity = {"format_version": 2, "settings": settings, "tasks_sha256": canonical_digest(tasks),
                "world_size": world_size, "implementation": implementation, "models": models}
    identity["audio_inventory"] = []
    for task in tasks:
        if "audio_path" in task:
            path = Path(task["audio_path"]).resolve(strict=True)
            stat = path.stat()
            identity["audio_inventory"].append({"id": task["id"], "path": str(path),
                                                 "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"sha256": canonical_digest(identity), "details": identity}


def make_schedule(tasks, world_size, rank, *, epochs, seed, shuffle=True):
    if world_size < 1 or not 0 <= rank < world_size or epochs < 1:
        raise ValueError("invalid schedule configuration")
    usable = len(tasks) // world_size * world_size
    if not usable:
        raise ValueError("at least one complete task per rank is required")
    schedule = []
    for epoch in range(epochs):
        indices = list(range(len(tasks)))
        if shuffle:
            random.Random(seed + epoch).shuffle(indices)
        schedule.extend((epoch, tasks[index]) for index in indices[:usable][rank::world_size])
    return schedule


def rollout_seed(seed, step, rank, candidate, task_id):
    digest = canonical_digest([seed, step, rank, candidate, task_id])
    return int(digest[:15], 16)


class Progress:
    """Heartbeat remains visible during slow loading, reward inference and backward."""
    def __init__(self, interval=15.0, rank=0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("log_interval_seconds must be positive and finite")
        self.interval, self.rank = interval, rank
        self.started = self.stage_started = time.monotonic()
        self.message = "starting"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def status(self, message):
        with self.lock:
            self.message, self.stage_started = message, time.monotonic()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                LOG.info("rank=%s %s | stage=%.1fs run=%.1fs", self.rank, self.message,
                         time.monotonic() - self.stage_started, time.monotonic() - self.started)

    def __exit__(self, *unused):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)


def read_resume_metadata(directory):
    directory = Path(directory).resolve()
    metadata_path = directory / "checkpoint.json"
    if not metadata_path.is_file():
        raise ValueError("resume requires a completed new-format checkpoint with checkpoint.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2 or type(metadata.get("step")) is not int or metadata["step"] < 1:
        raise ValueError("invalid checkpoint metadata")
    if not (directory / "adapter" / "adapter_config.json").is_file():
        raise ValueError("checkpoint adapter is missing")
    inventory = metadata.get("files_sha256")
    if not isinstance(inventory, dict) or "training_state.pt" not in inventory:
        raise ValueError("checkpoint hash manifest missing")
    for relative, expected in inventory.items():
        target = (directory / relative).resolve()
        if directory not in target.parents or not target.is_file() or file_sha256(target) != expected:
            raise ValueError(f"checkpoint file integrity check failed: {relative}")
    return metadata
