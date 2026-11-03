"""FIFO, one-worker micro-batching for independent streaming conversations.

There is at most one outstanding turn per session. Only the pump submits model
work, so even an executor with multiple threads cannot call the engine in parallel.
Cancelling a caller does not release its in-flight audio/model state prematurely.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
import logging
import time

from .model import ModelBackendError


LOG = logging.getLogger(__name__)


class SchedulerCancelled(RuntimeError):
    """A queued turn or the scheduler itself was explicitly cancelled."""


class SchedulerUnavailable(RuntimeError):
    """Admission/result error with a stable service-health code."""

    def __init__(self, message, *, code='engine_unavailable'):
        super().__init__(message)
        if code not in ('engine_stalled', 'engine_failed', 'engine_unavailable'):
            raise ValueError('invalid scheduler availability code')
        self.code = code


@dataclass
class _BatchTiming:
    # Written by the one worker, read by the event loop. No model data is shared.
    started: float | None = None
    finished: float | None = None


@dataclass(frozen=True)
class ScheduledGeneration:
    result: object
    queue_wait_s: float
    service_s: float
    batch_size: int


@dataclass
class _Job:
    session_id: str
    session: object
    audio: object
    submitted_at: float
    future: asyncio.Future
    state: str = 'queued'


def _observe_exception(future):
    # A disconnected client may never resume its await. Retrieving the exception
    # prevents warnings; another await still receives the same exception normally.
    if not future.cancelled():
        future.exception()


class InferenceScheduler:
    def __init__(self, engine, executor, *, max_batch_size=4,
                 batch_wait_ms=10, max_sessions=4,
                 slow_inference_seconds=5, stalled_inference_seconds=60):
        for name, value in (('max_batch_size', max_batch_size),
                            ('max_sessions', max_sessions)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(batch_wait_ms, (float, int)) or isinstance(batch_wait_ms, bool):
            raise ValueError('batch_wait_ms must be a non-negative finite number')
        if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError('batch_wait_ms must be a non-negative finite number')
        for name, value in (('slow_inference_seconds', slow_inference_seconds),
                            ('stalled_inference_seconds', stalled_inference_seconds)):
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be a positive finite number')
        if stalled_inference_seconds <= slow_inference_seconds:
            raise ValueError('stalled_inference_seconds must exceed slow_inference_seconds')
        self.engine = engine
        self.executor = executor
        self.max_sessions = max_sessions
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.slow_inference_seconds = float(slow_inference_seconds)
        self.stalled_inference_seconds = float(stalled_inference_seconds)
        self._health_poll_seconds = min(0.25, max(0.001, self.slow_inference_seconds / 4))
        self._queue = deque()
        self._jobs = {}
        self._wake = asyncio.Event()
        self._task = None
        self._closing = False
        self._failed = None
        self._stalled = False
        self._active_clock = None
        self._active_worker = None
        self._active_jobs = []

    def start(self):
        if self._closing:
            raise SchedulerCancelled('inference scheduler is closed')
        if self._task is None:
            self._task = asyncio.create_task(self._pump(), name='flex-inference-scheduler')
            self._task.add_done_callback(self._pump_done)

    def _pump_done(self, task):
        # A task cancelled before its very first step never enters _pump's try
        # block. Its done callback still closes admission and wakes queued jobs.
        if task.cancelled():
            self._stop_accepting()
        elif task.exception() is not None:
            self._mark_failed(task.exception())
        elif not self._closing and self._failed is None:
            self._mark_failed(RuntimeError('inference scheduler stopped unexpectedly'))

    def state(self, session_id):
        job = self._jobs.get(session_id)
        return job.state if job is not None else 'idle'

    def health(self):
        """Read-only status; WAIT/empty translation and queue age are irrelevant."""
        clock = self._active_clock
        running_seconds = 0.0
        if clock is not None and clock.started is not None and clock.finished is None:
            running_seconds = max(0.0, time.perf_counter() - clock.started)
        if self._failed is not None:
            state = 'failed'
            message = 'Shared inference backend failed; restart the service.'
        elif self._stalled or running_seconds >= self.stalled_inference_seconds:
            state = 'stalled'
            message = 'Inference is taking unusually long; new work is paused until it finishes.'
        elif running_seconds >= self.slow_inference_seconds:
            state = 'slow'
            message = 'Inference is slower than usual; the current request is still running.'
        else:
            state = 'ready'
            message = 'Inference engine is ready.'
        if self._closing:
            message = 'Inference scheduler is shutting down.'
        return dict(state=state, accepting=not self._closing and state in ('ready', 'slow'),
                    running_seconds=running_seconds, queued_requests=len(self._queue),
                    message=message)

    async def generate(self, session_id, model_session, audio):
        if self._closing:
            raise SchedulerCancelled('inference scheduler is closed')
        health = self.health()
        if not health['accepting']:
            raise SchedulerUnavailable(health['message'], code=(
                'engine_failed' if health['state'] == 'failed' else 'engine_stalled'))
        if session_id in self._jobs:
            raise RuntimeError('session already has a queued or running inference turn')
        if len(self._jobs) >= self.max_sessions:
            raise RuntimeError('inference scheduler session capacity reached')
        self.start()
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_observe_exception)
        job = _Job(session_id, model_session, audio, time.perf_counter(), future)
        self._jobs[session_id] = job
        self._queue.append(job)
        self._wake.set()
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A completed turn may have already relinquished its slot before
            # this caller resumes. Never cancel a newer turn using the same ID.
            if self._jobs.get(session_id) is job:
                self.cancel_pending(session_id)
            # Thread work cannot be cancelled safely. Keep this coroutine alive
            # until the engine is done, including repeated cancellation requests.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if future.done() and not future.cancelled():
                future.exception()
            raise

    def cancel_pending(self, session_id):
        job = self._jobs.get(session_id)
        if job is None or job.state != 'queued':
            return False
        self._queue.remove(job)
        del self._jobs[session_id]
        job.state = 'cancelled'
        job.future.set_exception(SchedulerCancelled('queued inference turn was cancelled'))
        self._wake.set()
        return True

    async def close(self):
        self._stop_accepting()
        if self._task is not None:
            # A deliberately cancelled pump is a terminal shutdown state, not a
            # cancellation of this close caller. Shield still propagates an
            # actual cancellation request to the close caller itself.
            outcomes = await asyncio.shield(asyncio.gather(
                self._task, return_exceptions=True))
            if isinstance(outcomes[0], Exception):
                raise outcomes[0]

    def _stop_accepting(self):
        self._closing = True
        for job in list(self._queue):
            self.cancel_pending(job.session_id)
        self._wake.set()

    def _fail_queued(self, code, message):
        while self._queue:
            job = self._queue.popleft()
            del self._jobs[job.session_id]
            job.state = 'finished'
            job.future.set_exception(SchedulerUnavailable(message, code=code))
        self._wake.set()

    def _mark_failed(self, error):
        if self._failed is None:
            self._failed = error
            LOG.error('Shared inference service entered failed state',
                      exc_info=(type(error), error, error.__traceback__))
        self._fail_queued('engine_failed', 'Shared inference backend failed; restart the service.')

    def _clear_active(self):
        self._active_clock = None
        self._active_worker = None
        self._active_jobs = []
        self._stalled = False

    def _run_batch(self, jobs, clock):
        started = time.perf_counter()
        clock.started = started
        shared_error = None
        try:
            generate_batch = getattr(self.engine, 'generate_batch', None)
            if callable(generate_batch):
                results = list(generate_batch([(job.session, job.audio) for job in jobs]))
                if len(results) != len(jobs):
                    raise RuntimeError('model batch returned a different number of results')
            else:
                results = []
                for job in jobs:
                    try:
                        results.append(job.session.generate(job.audio))
                    except Exception as exc:
                        results.append(exc)
        except Exception as exc:
            results = [exc] * len(jobs)
            shared_error = exc
        if shared_error is None:
            shared_error = next((result for result in results
                                 if isinstance(result, ModelBackendError)), None)
        clock.finished = time.perf_counter()
        return started, clock.finished, results, shared_error

    async def _wait_worker(self, worker):
        # asyncio.wait does not cancel its member futures on timeout or caller
        # cancellation. Health checks therefore never interrupt real GPU work.
        while not worker.done():
            await asyncio.wait((worker,), timeout=self._health_poll_seconds)
            if worker.done():
                break
            if self.health()['state'] == 'stalled' and not self._stalled:
                self._stalled = True
                self._fail_queued('engine_stalled',
                                  'Inference is taking unusually long; try again after it finishes.')
        return worker.result()

    async def _drain_worker(self, worker):
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()

    async def _pump(self):
        try:
            await self._dispatch()
        except asyncio.CancelledError:
            # Also handles cancellation while idle or collecting a partial
            # batch. In-flight cleanup happens in _dispatch before reaching here.
            self._stop_accepting()
            raise
        except Exception as exc:
            # An unexpected pump error must not leave admission green or callers
            # waiting forever. Still keep in-flight ownership until thread exit.
            self._mark_failed(exc)
            if self._active_worker is not None:
                await self._drain_worker(self._active_worker)
            now = time.perf_counter()
            self._finish_batch(self._active_jobs, [SchedulerUnavailable(
                'Inference scheduler failed; restart the service.', code='engine_failed')
                for _ in self._active_jobs], now, now)
            self._clear_active()

    def _finish_batch(self, jobs, results, started, finished):
        for job, result in zip(jobs, results):
            del self._jobs[job.session_id]
            job.state = 'finished'
            if isinstance(result, Exception):
                job.future.set_exception(result)
            else:
                job.future.set_result(ScheduledGeneration(
                    result=result,
                    queue_wait_s=max(0.0, started - job.submitted_at),
                    service_s=max(0.0, finished - started),
                    batch_size=len(jobs),
                ))

    async def _dispatch(self):
        loop = asyncio.get_running_loop()
        while True:
            while not self._queue:
                if self._closing or self._failed is not None:
                    return
                self._wake.clear()
                await self._wake.wait()
            # The finite window gathers only turns already submitted by clients;
            # it never asks for or waits for future audio from any one client.
            deadline = loop.time() + self.batch_wait_s
            while len(self._queue) < self.max_batch_size and not self._closing:
                remaining = deadline - loop.time()
                if remaining <= 0 or not self._queue:
                    break
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), remaining)
                except asyncio.TimeoutError:
                    break
            if not self._queue:
                continue
            jobs = [self._queue.popleft()
                    for _ in range(min(self.max_batch_size, len(self._queue)))]
            for job in jobs:
                job.state = 'running'
            self._active_jobs = jobs
            self._active_clock = _BatchTiming()
            worker = None
            try:
                worker = loop.run_in_executor(
                    self.executor, self._run_batch, jobs, self._active_clock)
                self._active_worker = worker
                # Never let cancellation of the pump cancel the asyncio wrapper
                # around real thread work: it would lose the completion signal
                # even while the model is still using its session/audio state.
                started, finished, results, shared_error = await self._wait_worker(worker)
            except asyncio.CancelledError:
                self._stop_accepting()
                if worker is not None:
                    await self._drain_worker(worker)
                # Retrieve any executor error, then terminalize every in-flight
                # caller only after the actual worker has relinquished ownership.
                now = time.perf_counter()
                self._finish_batch(jobs, [SchedulerCancelled(
                    'inference scheduler stopped during generation') for _ in jobs], now, now)
                self._clear_active()
                raise
            except Exception as exc:
                # Executor submission/failure must not strand session ownership.
                # An unexpected polling failure can occur while real work is
                # still running; latch failure now but release ownership later.
                self._mark_failed(exc)
                if worker is not None:
                    await self._drain_worker(worker)
                started = finished = time.perf_counter()
                results = [exc] * len(jobs)
                shared_error = exc
            if shared_error is not None:
                self._mark_failed(shared_error)
                for index, result in enumerate(results):
                    if result is shared_error or isinstance(result, ModelBackendError):
                        unavailable = SchedulerUnavailable(
                            'Shared inference backend failed; restart the service.', code='engine_failed')
                        unavailable.__cause__ = result
                        results[index] = unavailable
            self._finish_batch(jobs, results, started, finished)
            self._clear_active()
