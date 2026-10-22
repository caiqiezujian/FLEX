"""Read-only reward/performance log observer; never imported by training.

TensorBoard is optional and loaded only by EventWriter. No torch, CUDA, audio
decoding, remote scoring, distributed communication or model loading occurs.
"""
from __future__ import annotations

from collections import Counter
import csv
import html
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time

LOG = logging.getLogger("flex.monitor")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def action_label(response):
    text = response.strip() if isinstance(response, str) else ""
    return next((a for a in ("WAIT", "FLEX", "DIRECT") if text.startswith("<|" + a + "|>")), "UNRECOGNIZED")


def slim_response(response):
    """Keep action/exact WAIT status without retaining malformed raw prose."""
    if response.strip() == "<|WAIT|>":
        return "<|WAIT|>"
    action = action_label(response)
    return "<|" + action + "|>[text]" if action != "UNRECOGNIZED" else "[unrecognized]"


class EventWriter:
    """Minimal tensorboard-only writer, avoiding torch.utils.tensorboard."""

    def __init__(self, path):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.writer = EventFileWriter(str(path), max_queue_size=100, flush_secs=10)

    def scalar(self, tag, value, step, walltime):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.writer.add_event(Event(wall_time=walltime, step=step,
                                    summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])))

    def text(self, tag, value, step, walltime):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.compat.proto.types_pb2 import DT_STRING
        metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="text"))
        tensor = TensorProto(dtype=DT_STRING, string_val=[value.encode("utf-8")],
                             tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
        self.writer.add_event(Event(wall_time=walltime, step=step, summary=Summary(value=[
            Summary.Value(tag=tag, metadata=metadata, tensor=tensor)])))

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


def read_complete_lines(path, offset, limit=10000):
    """Do not advance past an unfinished line, including partial UTF-8 bytes."""
    rows = []
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        for _ in range(limit):
            start = stream.tell()
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n"):
                stream.seek(start)
                break
            offset = stream.tell()
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig" if start == 0 else "utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("not an object")
                rows.append(row)
            except (ValueError, UnicodeError) as exc:
                LOG.warning("Skipping malformed complete JSONL line: %s byte=%d (%s)", path, start, exc)
    return rows, offset


class SourceLookup:
    """Optional streaming SQLite index of prepared metadata, not audio files."""

    def __init__(self, dataset, output):
        dataset = Path(dataset).resolve()
        stat = dataset.stat()
        identity = json.dumps([str(dataset), stat.st_size, stat.st_mtime_ns])
        self.db = sqlite3.connect(str(Path(output) / "source_index.sqlite"))
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, source TEXT, reference TEXT)")
        prior = self.db.execute("SELECT identity FROM metadata LIMIT 1").fetchone()
        if prior and prior[0] == identity:
            return
        LOG.info("Building optional text lookup once: %s (no WAV reads)", dataset)
        with self.db:
            self.db.execute("DELETE FROM sources")
            self.db.execute("DELETE FROM metadata")
            with dataset.open(encoding="utf-8-sig") as stream:
                for index, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task = row.get("flex_task", row)
                    task = json.loads(task) if isinstance(task, str) else task
                    if not isinstance(task, dict) or not isinstance(task.get("id"), str):
                        continue
                    source, reference = task.get("source_text"), task.get("reference_text")
                    if isinstance(source, str) and isinstance(reference, str):
                        previous = self.db.execute("SELECT source, reference FROM sources WHERE id=?", (task["id"],)).fetchone()
                        if previous is not None and previous != (source, reference):
                            raise ValueError("Conflicting source/reference for duplicate task id: " + task["id"])
                        self.db.execute("INSERT OR IGNORE INTO sources VALUES (?, ?, ?)", (task["id"], source, reference))
                    if index % 25000 == 0:
                        LOG.info("Indexed text metadata lines=%d", index)
            self.db.execute("INSERT INTO metadata VALUES (?)", (identity,))

    def get(self, task_id):
        return self.db.execute("SELECT source, reference FROM sources WHERE id=?", (str(task_id),)).fetchone()

    def close(self):
        self.db.close()


def _summary(values):
    if not values:
        return {}
    mean = sum(values) / len(values)
    return {"mean": mean, "min": min(values), "max": max(values),
            "std_local": math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))}


def reward_summary(records):
    """Local callback summaries only; no inferred global G8 groups/advantages."""
    result = {"candidates": len(records)}
    scalar_fields = ("quality", "metricx_error", "comet22_score", "latency_seconds", "mean_lag_seconds", "length_penalty")
    for key in ("reward", *scalar_fields):
        values = [r.get("reward") if key == "reward" else r.get("components", {}).get(key) for r in records]
        for name, value in _summary([float(v) for v in values if number(v)]).items():
            result[key + "/" + name] = value
        result[key + "/scored_count"] = sum(number(v) for v in values)
    for key in ("first_output_seconds", "max_no_text_gap_seconds", "eof_text_fraction", "emitted_units", "reference_units",
                "eof_generation_seconds", "eof_delay_proxy_seconds"):
        values = [(r.get("components", {}).get("diagnostics") or {}).get(key) for r in records]
        for name, value in _summary([float(v) for v in values if number(v)]).items():
            result["diagnostics/" + key + "/" + name] = value
    for flag in ("protocol_valid", "empty_translation", "quality_pass", "latency_pass", "feasible", "severe_repetition"):
        result[flag + "_fraction"] = sum(r.get("components", {}).get(flag) is True for r in records) / len(records)
    for tier in ("feasible", "quality_fail", "latency_fail", "invalid", "degenerate"):
        result["tier/" + tier + "_fraction"] = sum(r.get("components", {}).get("tier") == tier for r in records) / len(records)
    actions = Counter()
    eof_actions = Counter()
    complete = complete_known = final_wait = 0
    for record in records:
        trajectory = record.get("trajectory") or {}
        complete += trajectory.get("complete_input") is True
        complete_known += type(trajectory.get("complete_input")) is bool
        turns = trajectory.get("turns", [])
        if turns and turns[-1].get("response", "").strip() == "<|WAIT|>":
            final_wait += 1
        for turn in turns:
            counter = eof_actions if turn.get("kind") == "eof" else actions
            counter[action_label(turn.get("response", ""))] += 1
    eof_count = sum(eof_actions.values())
    result["eof_completed_count"] = eof_count
    if eof_count:
        for action in ("WAIT", "FLEX", "DIRECT", "UNRECOGNIZED"):
            result["eof_actions/" + action + "_fraction"] = eof_actions[action] / eof_count
    result["complete_input_known_count"] = complete_known
    if complete_known:
        result["complete_input_fraction"] = complete / complete_known
    result["final_wait_fraction"] = final_wait / len(records)
    total = sum(actions.values())
    for action in ("WAIT", "FLEX", "DIRECT", "UNRECOGNIZED"):
        result["actions/" + action + "_count"] = actions[action]
        if total:
            result["actions/" + action + "_fraction"] = actions[action] / total
    return result


def sample_text(record, lookup=None):
    """Render raw model strings as escaped text, not executable HTML/Markdown."""
    lines = ["Task: " + str(record.get("task_id")), "Reward: " + str(record.get("reward")),
             "Components: " + json.dumps(record.get("components", {}), ensure_ascii=False)]
    pair = lookup.get(record.get("task_id")) if lookup else None
    if pair:
        lines.extend(["Source: " + pair[0], "Machine reference (not semantic authority): " + pair[1]])
    lines.append("Complete translation: " + str(record.get("translation")))
    for turn in (record.get("trajectory") or {}).get("turns", []):
        lines.append(str(turn.get("audio_end_seconds")) + "s | " + str(turn.get("response")))
    return "<pre>" + html.escape("\n".join(lines)) + "</pre>"


class Monitor:
    def __init__(self, run_dir, *, writer_factory=EventWriter, max_samples=2, sample_every=10, lookup=None):
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise ValueError("Training output directory does not exist: " + str(self.run_dir))
        self.output = self.run_dir / "monitor"
        self.output.mkdir(exist_ok=True)
        # Single observer per output root; the OS releases this advisory lock
        # on exit/crash. A stale filename does not block the next observer.
        self.lock_handle = (self.output / "observer.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.lock_handle.write(b"0")
                self.lock_handle.flush()
                self.lock_handle.seek(0)
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock_handle.close()
            raise ValueError("Another observer is active for this output directory") from exc
        self.state_path = self.output / "cursor.json"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {"schema": 1, "files": {}, "samples": 0}
            if self.state.get("schema") != 1:
                raise ValueError("Unknown monitor cursor schema")
        except BaseException:
            self.lock_handle.close()
            raise
        self.writers = {}
        self.writer_factory, self.max_samples, self.lookup = writer_factory, max_samples, lookup
        self.sample_every = sample_every
        self.last_progress = False

    def writer(self, name):
        if name not in self.writers:
            self.writers[name] = self.writer_factory(self.output / "tensorboard" / name)
        return self.writers[name]

    @staticmethod
    def axis(record):
        step = record.get("global_step")
        if type(step) is int and step >= 0:
            return "by_global_step", step
        call = record.get("call", 0)
        return "by_callback", int(call) if number(call) and call >= 0 else 0

    def _emit_rewards(self, name, rows):
        if not rows:
            return
        writer = self.writer(name)
        axis, step = self.axis(rows[0])
        stamp = rows[0].get("timestamp_unix")
        stamp = float(stamp) if number(stamp) else time.time()
        for key, value in reward_summary(rows).items():
            writer.scalar("qalco_" + axis + "/" + key, value, step, stamp)
        if number(rows[0].get("call")):
            writer.scalar("qalco_" + axis + "/local_callback", rows[0]["call"], step, stamp)

    def poll(self):
        """Import complete new records only; per-rank/PID series never combined."""
        imported = 0
        self.last_progress = False
        for folder, pattern, kind in (("rewards", "rewards-*.jsonl", "reward"),
                                      ("performance", "performance-*.jsonl", "performance")):
            for path in sorted((self.run_dir / folder).glob(pattern)):
                relative = str(path.relative_to(self.run_dir)).replace("\\", "/")
                entry = self.state["files"].setdefault(relative, {"offset": 0, "epoch": 0})
                stat = path.stat()
                identity = [stat.st_dev, stat.st_ino]
                if stat.st_size < entry["offset"] or (entry.get("identity") and entry["identity"] != identity):
                    LOG.warning("Source replaced/truncated; opening a separate series: %s", path)
                    entry.update(offset=0, epoch=entry["epoch"] + 1)
                    entry.pop("pending", None)
                entry["identity"] = identity
                rows, offset = read_complete_lines(path, entry["offset"])
                self.last_progress = self.last_progress or offset != entry["offset"]
                name = path.stem + ("-replacement" + str(entry["epoch"]) if entry["epoch"] else "")
                dirty = False
                for row in rows:
                    imported += 1
                    if kind == "reward":
                        if (not number(row.get("reward")) or not isinstance(row.get("components"), dict)
                                or type(row.get("call")) is not int or row["call"] < 1
                                or not isinstance(row.get("trajectory"), dict)
                                or not isinstance(row["trajectory"].get("turns"), list)
                                or any(not isinstance(t, dict) or not isinstance(t.get("response"), str)
                                       for t in row["trajectory"]["turns"])):
                            LOG.warning("Skipping malformed reward record in %s", path)
                            continue
                        # A writer can be mid-callback. Accumulate additions in
                        # cursor state and emit an updated local callback mean.
                        # These are partial snapshots, not finalized G8 group
                        # means. TensorBoard can display several points at the
                        # same x; it does NOT necessarily replace earlier ones.
                        group_key = [row.get("call"), row.get("global_step"), row.get("timestamp_unix")]
                        pending = entry.get("pending")
                        if pending and pending["key"] != group_key:
                            if dirty:
                                self._emit_rewards(name, pending["rows"])
                            pending = None
                        if pending is None:
                            pending = {"key": group_key, "rows": [], "text_samples": 0}
                            entry["pending"] = pending
                        # Keep only metrics and actions in resumable state, not
                        # long translations, source text or complete raw turns.
                        slim = {k: row.get(k) for k in ("reward", "components", "call", "global_step", "timestamp_unix")}
                        slim["trajectory"] = {"complete_input": row["trajectory"].get("complete_input"),
                                              "turns": [{"response": slim_response(t["response"]), "kind": t.get("kind", "audio")}
                                                               for t in (row.get("trajectory") or {}).get("turns", [])]}
                        pending["rows"].append(slim)
                        dirty = True
                        call = row.get("call", 1)
                        selected = type(call) is int and (call == 1 or call % self.sample_every == 0)
                        if selected and pending.get("text_samples", 0) < self.max_samples:
                            axis, step = self.axis(row)
                            text = sample_text(row, self.lookup)
                            # Bound TensorBoard display volume, not raw logs.
                            if len(text) > 24000:
                                text = text[:24000] + "\n[Display truncated; complete text remains in rewards JSONL.]</pre>"
                            self.writer(name).text("samples_" + axis + "/candidate_" + str(pending.get("text_samples", 0)),
                                                   text, step,
                                                   float(row["timestamp_unix"]) if number(row.get("timestamp_unix")) else time.time())
                            self.state["samples"] += 1
                            pending["text_samples"] = pending.get("text_samples", 0) + 1
                    else:
                        axis, step = self.axis(row)
                        stamp = float(row["timestamp_unix"]) if number(row.get("timestamp_unix")) else time.time()
                        for key, value in row.items():
                            if number(value) and (key.endswith("_seconds") or key.endswith("_count") or key == "local_trajectories"):
                                self.writer(name).scalar("performance_" + axis + "/" + key, value, step, stamp)
                        if row.get("status") in ("ok", "error"):
                            self.writer(name).scalar("performance_" + axis + "/generation_failed", row["status"] == "error", step, stamp)
                        entry["latest_performance"] = row
                if dirty:
                    self._emit_rewards(name, entry["pending"]["rows"])
                entry["offset"] = offset
        self.checkpoint()
        return imported

    def checkpoint(self):
        for writer in self.writers.values():
            writer.flush()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.state_path)
        summaries = {}
        for source, entry in self.state["files"].items():
            if entry.get("pending"):
                rows = entry["pending"]["rows"]
                summaries[source] = {"global_step": rows[0].get("global_step"), "call": rows[0].get("call"),
                                     "scope": "local callback, possibly still being appended", **reward_summary(rows)}
            elif entry.get("latest_performance"):
                summaries[source] = entry["latest_performance"]
        latest = self.output / "summary.latest.json"
        temporary = latest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"timestamp_unix": time.time(), "series": summaries}, ensure_ascii=False,
                                        indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, latest)

    def gpu_sample(self, ids):
        result = subprocess.run(["nvidia-smi", "--id=" + ids,
                                 "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        writer = self.writer("gpu_device_samples")
        stamp = time.time()
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 5:
                continue
            for name, raw in zip(("used_mib", "total_mib", "utilization_percent", "power_watts"), row[1:]):
                try:
                    value = float(raw.strip())
                except ValueError:
                    continue
                if math.isfinite(value):
                    writer.scalar("device_" + row[0].strip() + "/" + name, value, int(stamp), stamp)
        writer.flush()

    def close(self):
        try:
            for writer in self.writers.values():
                writer.close()
        finally:
            self.lock_handle.close()
