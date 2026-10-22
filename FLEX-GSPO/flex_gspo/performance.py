"""Optional host-side rollout timings, never a CUDA profiler or a collective.

These intervals include whatever asynchronous work the underlying call waits
for. They are not CUDA kernel durations and need not sum to generation time.
Logging failures are diagnostic only: they must not change training outcomes.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
from pathlib import Path
import time


LOG = logging.getLogger("flex.performance")
_ACTIVE = ContextVar("flex_generation_performance", default=None)
_SECONDS = ("weight_sync_seconds", "rollout_requests_seconds",
            "offload_to_cpu_seconds", "reload_to_gpu_seconds")


def _warn(exc):
    try:
        LOG.warning("Optional FLEX performance diagnostics failed: %s", type(exc).__name__)
    except Exception:
        pass


def start_generation(trainer):
    """Return an opaque session, or None when disabled/unavailable."""
    directory = os.environ.get("FLEX_PERF_LOG_DIR")
    if not directory:
        return None
    try:
        call = getattr(trainer, "_flex_performance_calls", 0) + 1
        trainer._flex_performance_calls = call
        step = getattr(getattr(trainer, "state", None), "global_step", None)
        source = "trainer.state.global_step"
        if not isinstance(step, int) or isinstance(step, bool):
            step, source = getattr(trainer, "_step", None), "trainer._step"
        if not isinstance(step, int) or isinstance(step, bool):
            step, source = None, "unavailable"
        rank = "".join(c for c in os.environ.get("RANK", "0") if c.isalnum() or c in "-_") or "0"
        row = {"schema_version": 1, "timestamp_unix": time.time(),
               "rank": rank, "pid": os.getpid(), "call": call,
               "global_step": step, "step_source": source,
               "timing_kind": "host_wall_clock_unsynchronized",
               "rollout_requests_count": 0,
               **{name: 0.0 for name in _SECONDS}}
        session = {"trainer": trainer, "directory": directory, "row": row,
                   "started": time.monotonic()}
        session["token"] = _ACTIVE.set(session)
        return session
    except Exception as exc:
        _warn(exc)
        return None


def finish_generation(session, *, status, local_trajectories=None):
    if session is None:
        return
    try:
        row = session["row"]
        row.update(status=status, local_trajectories=local_trajectories,
                   generation_seconds=max(0.0, time.monotonic() - session["started"]))
        directory = Path(session["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"performance-rank{row['rank']}-pid{row['pid']}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception as exc:
        _warn(exc)
    finally:
        _ACTIVE.reset(session["token"])


def _session_for(trainer):
    session = _ACTIVE.get()
    return session if session is not None and session["trainer"] is trainer else None


def _record(session, name, started, count=False):
    try:
        row = session["row"]
        row[name] += max(0.0, time.monotonic() - started)
        if count:
            row["rollout_requests_count"] += 1
    except Exception as exc:
        _warn(exc)


def _timed_method(original, name, count=False):
    @wraps(original)
    def wrapped(trainer, *args, **kwargs):
        session = _session_for(trainer)
        if session is None:
            return original(trainer, *args, **kwargs)
        started = time.monotonic()
        try:
            return original(trainer, *args, **kwargs)
        finally:
            _record(session, name, started, count)
    return wrapped


class _TimedOffload:
    """Proxy the original context manager, including exception suppression."""
    def __init__(self, original, session):
        self.original, self.session = original, session

    def __enter__(self):
        started = time.monotonic()
        try:
            return self.original.__enter__()
        finally:
            _record(self.session, "offload_to_cpu_seconds", started)

    def __exit__(self, *exc):
        started = time.monotonic()
        try:
            return self.original.__exit__(*exc)
        finally:
            _record(self.session, "reload_to_gpu_seconds", started)


def install_phase_timers(cls):
    """Wrap existing methods only when enabled; no upstream files are edited.

The class wrappers delegate untouched outside an active FLEX generation.
Missing methods leave the corresponding timing zero; generation still works.
"""
    if not os.environ.get("FLEX_PERF_LOG_DIR") or getattr(cls, "_flex_performance_timers_installed", False):
        return
    for method, field, count in (("_move_model_to_vllm", "weight_sync_seconds", False),
                                 ("_rollout_requests", "rollout_requests_seconds", True)):
        original = getattr(cls, method, None)
        if callable(original):
            setattr(cls, method, _timed_method(original, field, count))
    original_context = getattr(cls, "offload_context", None)
    if callable(original_context):
        @wraps(original_context)
        def offload_context(trainer, *args, **kwargs):
            context = original_context(trainer, *args, **kwargs)
            session = _session_for(trainer)
            return context if session is None else _TimedOffload(context, session)
        cls.offload_context = offload_context
    cls._flex_performance_timers_installed = True
