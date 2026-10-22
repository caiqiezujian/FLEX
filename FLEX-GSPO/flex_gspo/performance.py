"""Optional host-side rollout timings, never a CUDA profiler or a collective.

These intervals include whatever asynchronous work the underlying call waits
for. They are not CUDA kernel durations and need not sum to generation time.
Logging failures are diagnostic only: they must not change training outcomes.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import inspect
import json
import logging
import os
from pathlib import Path
import time


LOG = logging.getLogger("flex.performance")
_ACTIVE = ContextVar("flex_generation_performance", default=None)
_SECONDS = ("weight_sync_seconds", "rollout_requests_seconds",
            "offload_to_cpu_seconds", "reload_to_gpu_seconds", "vllm_sleep_seconds")


def _warn(exc):
    try:
        LOG.warning("Optional FLEX performance diagnostics failed: %s", type(exc).__name__)
    except Exception:
        pass


def _phase_event(session, phase, event, phase_id, *, status=None, elapsed=None):
    """Close each small append so a stalled call still leaves its begin event.

    Separate files keep existing completed-generation consumers unchanged.
    No audio, requests, model output, exception messages or credentials are logged.
    """
    try:
        metadata = session["row"]
        session["event_sequence"] = session.get("event_sequence", 0) + 1
        row = {name: metadata[name] for name in (
            "rank", "pid", "call", "global_step", "step_source", "timing_kind")}
        row.update(schema_version=1, timestamp_unix=time.time(), phase=phase,
                   event=event, phase_id=phase_id, event_sequence=session["event_sequence"],
                   generation_elapsed_seconds=max(0.0, time.monotonic() - session["started"]))
        if status is not None:
            row["status"] = status
        if elapsed is not None:
            row["elapsed_seconds"] = elapsed
        directory = Path(session["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"phases-rank{row['rank']}-pid{row['pid']}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception as exc:
        _warn(exc)


def _begin_phase(session, phase):
    started = time.monotonic()
    session["phase_sequence"] = session.get("phase_sequence", 0) + 1
    phase_id = session["phase_sequence"]
    _phase_event(session, phase, "begin", phase_id)
    return started, phase_id


def _end_phase(session, phase, span, status):
    started, phase_id = span
    _phase_event(session, phase, "end", phase_id, status=status,
                 elapsed=max(0.0, time.monotonic() - started))


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
        _phase_event(session, "generation", "begin", 0)
        return session
    except Exception as exc:
        _warn(exc)
        return None


def finish_generation(session, *, status, local_trajectories=None):
    if session is None:
        return
    try:
        _phase_event(session, "generation", "end", 0, status=status,
                     elapsed=max(0.0, time.monotonic() - session["started"]))
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
        try:
            _ACTIVE.reset(session["token"])
        except Exception as exc:
            _warn(exc)


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
        phase = name[:-len("_seconds")]
        span = _begin_phase(session, phase)
        status = "error"
        try:
            result = original(trainer, *args, **kwargs)
            status = "ok"
            return result
        finally:
            _record(session, name, span[0], count)
            _end_phase(session, phase, span, status)
    return wrapped


class _TimedOffload:
    """Proxy the original context manager, including exception suppression."""
    def __init__(self, original, session):
        self.original, self.session = original, session

    def __enter__(self):
        span = _begin_phase(self.session, "offload_to_cpu")
        status = "error"
        try:
            result = self.original.__enter__()
            status = "ok"
            return result
        finally:
            _record(self.session, "offload_to_cpu_seconds", span[0])
            _end_phase(self.session, "offload_to_cpu", span, status)

    def __exit__(self, *exc):
        span = _begin_phase(self.session, "reload_to_gpu")
        status = "error"
        try:
            result = self.original.__exit__(*exc)
            status = "ok"
            return result
        finally:
            _record(self.session, "reload_to_gpu_seconds", span[0])
            _end_phase(self.session, "reload_to_gpu", span, status)


def _install_sleep_timer(cls):
    """Instrument the imported call-site name, not the shared utility module.

    SWIFT decorates _generate_completions, so unwrap to find its real globals.
    Unknown call sites remain untouched. The active session and engine identity
    restrict timing to this FLEX generation even if other callers share globals.
    """
    try:
        method = getattr(cls, "_generate_completions", None)
        if method is None:
            return
        namespace = getattr(inspect.unwrap(method), "__globals__", {})
        original = namespace.get("sleep_vllm_engine")
        if not callable(original) or getattr(original, "_flex_sleep_timer", False):
            return

        @wraps(original)
        def sleep_vllm_engine(engine, *args, **kwargs):
            session = _ACTIVE.get()
            owner = getattr(session["trainer"], "engine", None) if session is not None else None
            if session is None or getattr(owner, "engine", None) is not engine:
                return original(engine, *args, **kwargs)
            span = _begin_phase(session, "vllm_sleep")
            status = "error"
            try:
                result = original(engine, *args, **kwargs)
                status = "ok"
                return result
            finally:
                _record(session, "vllm_sleep_seconds", span[0])
                _end_phase(session, "vllm_sleep", span, status)

        sleep_vllm_engine._flex_sleep_timer = True
        namespace["sleep_vllm_engine"] = sleep_vllm_engine
    except Exception as exc:
        _warn(exc)


def install_phase_timers(cls):
    """Wrap existing methods only when enabled; no upstream files are edited.

The class wrappers delegate untouched outside an active FLEX generation.
Missing methods leave the corresponding timing zero; generation still works.
"""
    if not os.environ.get("FLEX_PERF_LOG_DIR") or getattr(cls, "_flex_performance_timers_installed", False):
        return
    _install_sleep_timer(cls)
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
