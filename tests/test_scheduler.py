"""CPU-only fairness, bounded admission, batching and cancellation tests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web.model import ModelBackendError
from flex_web.scheduler import InferenceScheduler, SchedulerCancelled, SchedulerUnavailable


class BatchEngine:
    def __init__(self, *, blocked=False, delay=0):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.delay = delay
        self.batches = []
        self.active = 0
        self.peak_active = 0
        self.fail_batch = False

    def generate_batch(self, jobs):
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.batches.append(list(jobs))
        self.entered.set()
        try:
            if not self.release.wait(5):
                raise RuntimeError('test forgot to release worker')
            time.sleep(self.delay)
            if self.fail_batch:
                self.fail_batch = False
                raise ValueError('shared backend error')
            return [ValueError('bad item') if audio == 'bad' else (session, audio)
                    for session, audio in jobs]
        finally:
            self.active -= 1


async def eventually(predicate):
    deadline = asyncio.get_running_loop().time() + 2
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError('condition did not become true')
        await asyncio.sleep(0.001)


def test_default_four_sessions_make_one_batch_and_full_batch_skips_wait():
    async def run(executor):
        engine = BatchEngine()
        scheduler = InferenceScheduler(engine, executor, batch_wait_ms=1000)
        scheduler.start()
        try:
            tasks = [asyncio.create_task(scheduler.generate(str(i), f's{i}', i))
                     for i in range(4)]
            results = await asyncio.wait_for(asyncio.gather(*tasks), 0.5)
            assert engine.batches == [[(f's{i}', i) for i in range(4)]]
            assert [result.result for result in results] == [(f's{i}', i) for i in range(4)]
            assert all(result.batch_size == 4 for result in results)
            assert all(scheduler.state(str(i)) == 'idle' for i in range(4))
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_four_clients_fifo_and_one_engine_call_even_with_multiworker_executor():
    async def run(executor):
        engine = BatchEngine(delay=0.01)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        try:
            async def client(index):
                for turn in range(3):
                    await scheduler.generate(str(index), index, turn)
            await asyncio.gather(*(client(i) for i in range(4)))
            assert engine.batches == [[(i, turn)] for turn in range(3) for i in range(4)]
            assert engine.peak_active == 1
            assert not scheduler._jobs and not scheduler._queue
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=4) as executor:
        asyncio.run(run(executor))


def test_fifth_job_capacity_and_duplicate_session_rejected_while_four_running():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor)
        tasks = [asyncio.create_task(scheduler.generate(str(i), i, i)) for i in range(4)]
        try:
            await eventually(engine.entered.is_set)
            assert all(scheduler.state(str(i)) == 'running' for i in range(4))
            with pytest.raises(RuntimeError, match='capacity'):
                await scheduler.generate('fifth', 5, 5)
            with pytest.raises(RuntimeError, match='already'):
                await scheduler.generate('0', 0, 100)
        finally:
            engine.release.set()
            await asyncio.gather(*tasks)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_cancel_pending_removes_queue_immediately_and_never_calls_engine_for_it():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            assert scheduler.cancel_pending('b') is True
            assert scheduler.cancel_pending('a') is False
            assert scheduler.cancel_pending('missing') is False
            assert scheduler.state('b') == 'idle'
            with pytest.raises(SchedulerCancelled):
                await second
            engine.release.set()
            await first
            assert engine.batches == [[('a', 1)]]
        finally:
            engine.release.set()
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_cancelled_queued_caller_is_removed_without_waiting_for_running_worker():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(second, 0.5)
            assert scheduler.state('b') == 'idle'
            assert not first.done()
        finally:
            engine.release.set()
            await first
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_running_caller_cancellation_keeps_ownership_until_worker_finishes():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        task = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(engine.entered.is_set)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert scheduler.state('a') == 'running'
            with pytest.raises(RuntimeError, match='already'):
                await scheduler.generate('a', 'replacement', 2)
            engine.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert scheduler.state('a') == 'idle'
        finally:
            engine.release.set()
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_private_errors_allow_progress_but_shared_errors_latch_failed():
    async def run(executor):
        engine = BatchEngine()
        scheduler = InferenceScheduler(engine, executor, max_batch_size=2)
        try:
            first, second = await asyncio.gather(
                scheduler.generate('a', 'a', 'bad'), scheduler.generate('b', 'b', 2),
                return_exceptions=True)
            assert isinstance(first, ValueError)
            assert second.result == ('b', 2)
            assert scheduler.health()['state'] == 'ready'
            engine.fail_batch = True
            shared = await asyncio.gather(
                scheduler.generate('a', 'a', 3), scheduler.generate('b', 'b', 4),
                return_exceptions=True)
            assert all(isinstance(item, SchedulerUnavailable) for item in shared)
            assert all(item.code == 'engine_failed' for item in shared)
            assert all(isinstance(item.__cause__, ValueError) for item in shared)
            assert scheduler.health()['state'] == 'failed'
            with pytest.raises(SchedulerUnavailable) as error:
                await scheduler.generate('c', 'c', 5)
            assert error.value.code == 'engine_failed'
            assert len(engine.batches) == 2
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_close_cancels_queued_work_waits_inflight_and_rejects_new_work():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            closing = asyncio.create_task(scheduler.close())
            with pytest.raises(SchedulerCancelled):
                await second
            assert not closing.done()
            with pytest.raises(SchedulerCancelled):
                await scheduler.generate('c', 'c', 3)
            engine.release.set()
            assert (await first).result == ('a', 1)
            await closing
            await scheduler.close()
            assert scheduler._task.done()
        finally:
            engine.release.set()
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_fallback_engine_is_serial_and_isolates_session_exception():
    class Session:
        def __init__(self, failing=False):
            self.failing = failing

        def generate(self, audio):
            if self.failing:
                raise ValueError('session failure')
            return audio

    async def run(executor):
        scheduler = InferenceScheduler(object(), executor, max_batch_size=2)
        try:
            results = await asyncio.gather(
                scheduler.generate('a', Session(True), 1),
                scheduler.generate('b', Session(), 2), return_exceptions=True)
            assert isinstance(results[0], ValueError)
            assert results[1].result == 2
            assert results[1].batch_size == 2
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_queue_and_service_timings_are_distinct_and_include_collection_window():
    async def run(executor):
        engine = BatchEngine(delay=0.025)
        scheduler = InferenceScheduler(engine, executor, batch_wait_ms=30)
        try:
            result = await scheduler.generate('a', 'a', 1)
            assert result.queue_wait_s >= 0.02
            assert result.service_s >= 0.02
            assert result.batch_size == 1
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_wrong_batch_cardinality_fails_every_item_and_latches_failed():
    class Engine:
        def generate_batch(self, jobs):
            return []

    async def run(executor):
        scheduler = InferenceScheduler(Engine(), executor, max_batch_size=2)
        try:
            results = await asyncio.gather(
                scheduler.generate('a', 'a', 1), scheduler.generate('b', 'b', 2),
                return_exceptions=True)
            assert all(isinstance(item, RuntimeError) for item in results)
            assert all(item.code == 'engine_failed' for item in results)
            assert scheduler.health()['state'] == 'failed'
            assert not scheduler.health()['accepting']
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize('kwargs', [
    {'max_sessions': 0}, {'max_sessions': True}, {'max_batch_size': 0},
    {'max_batch_size': 1.5}, {'batch_wait_ms': -1}, {'batch_wait_ms': float('nan')},
    {'batch_wait_ms': float('inf')}, {'batch_wait_ms': True},
    {'slow_inference_seconds': 0}, {'slow_inference_seconds': True},
    {'slow_inference_seconds': float('nan')}, {'stalled_inference_seconds': float('inf')},
    {'stalled_inference_seconds': 5}, {'stalled_inference_seconds': 4},
])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        InferenceScheduler(object(), None, **kwargs)


def test_close_without_start_is_safe_and_cannot_restart():
    async def run():
        scheduler = InferenceScheduler(object(), None)
        await scheduler.close()
        with pytest.raises(SchedulerCancelled):
            scheduler.start()
    asyncio.run(run())


@pytest.mark.parametrize('cancel_caller', [False, True])
@pytest.mark.parametrize('fail_worker', [False, True])
def test_forced_pump_cancellation_waits_real_worker_and_settles_every_job(
        cancel_caller, fail_worker):
    async def run(executor):
        engine = BatchEngine(blocked=True)
        engine.fail_batch = fail_worker
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        second = None
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            scheduler._task.cancel()
            await eventually(second.done)
            with pytest.raises(SchedulerCancelled):
                await second
            assert not first.done()
            assert engine.active == 1
            assert scheduler.state('a') == 'running'
            assert scheduler.state('b') == 'idle'
            with pytest.raises(SchedulerCancelled):
                await scheduler.generate('c', 'c', 3)
            # Repeated server cancellation must not sever the worker completion
            # signal, even when the client is being cancelled independently.
            scheduler._task.cancel()
            if cancel_caller:
                first.cancel()
            await asyncio.sleep(0.01)
            scheduler._task.cancel()
            if cancel_caller:
                first.cancel()
            await asyncio.sleep(0.01)
            assert not scheduler._task.done() and not first.done()
            assert scheduler.state('a') == 'running'
            engine.release.set()
            await eventually(lambda: scheduler._task.done() and first.done())
            expected = asyncio.CancelledError if cancel_caller else SchedulerCancelled
            with pytest.raises(expected):
                await first
            assert engine.active == 0
            assert scheduler.state('a') == 'idle'
            assert not scheduler._jobs and not scheduler._queue
            assert engine.batches == [[('a', 1)]]
            # A terminal cancelled pump must not cancel close's caller or skip
            # subsequent engine shutdown by propagating its CancelledError.
            await scheduler.close()
        finally:
            engine.release.set()
            await eventually(lambda: engine.active == 0)
            # Regression failures must not hang asyncio.run's own cancellation
            # cleanup. Only unblock futures after real thread work has finished.
            for job in list(scheduler._jobs.values()):
                if not job.future.done():
                    job.future.set_exception(SchedulerCancelled('test cleanup'))
            if scheduler._task is not None and not scheduler._task.done():
                scheduler._task.cancel()
            await asyncio.gather(first, *([second] if second is not None else []),
                                 return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_forced_pump_cancellation_during_batch_collection_cancels_queued_job():
    async def run(executor):
        engine = BatchEngine()
        scheduler = InferenceScheduler(engine, executor, batch_wait_ms=1000)
        task = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(lambda: scheduler.state('a') == 'queued')
            scheduler._task.cancel()
            await eventually(task.done)
            with pytest.raises(SchedulerCancelled):
                await task
            assert engine.batches == []
            assert scheduler.state('a') == 'idle'
            with pytest.raises(SchedulerCancelled):
                await scheduler.generate('b', 'b', 2)
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_pump_cancelled_before_first_step_still_closes_admission():
    async def run():
        scheduler = InferenceScheduler(object(), None)
        scheduler.start()
        scheduler._task.cancel()
        await eventually(lambda: scheduler._closing)
        with pytest.raises(SchedulerCancelled):
            await scheduler.generate('a', 'a', 1)
        await scheduler.close()
    asyncio.run(run())


def test_idle_and_queued_batch_collection_are_not_slow_or_stalled():
    async def run(executor):
        scheduler = InferenceScheduler(BatchEngine(), executor, batch_wait_ms=200,
                                       slow_inference_seconds=0.01,
                                       stalled_inference_seconds=0.03)
        scheduler.start()
        task = None
        try:
            await asyncio.sleep(0.04)
            assert scheduler.health() == dict(
                state='ready', accepting=True, running_seconds=0.0, queued_requests=0,
                message='Inference engine is ready.')
            task = asyncio.create_task(scheduler.generate('a', 'a', 1))
            await asyncio.sleep(0.04)
            assert scheduler.health()['state'] == 'ready'
            assert scheduler.health()['queued_requests'] == 1
            assert scheduler.health()['running_seconds'] == 0.0
            assert not task.done()
        finally:
            scheduler.cancel_pending('a')
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_executor_wait_is_not_counted_as_running_inference():
    async def run(executor):
        release = threading.Event()
        occupied = executor.submit(release.wait, 5)
        engine = BatchEngine()
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0,
                                       slow_inference_seconds=0.01,
                                       stalled_inference_seconds=0.03)
        task = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(lambda: scheduler.state('a') == 'running')
            await asyncio.sleep(0.04)
            assert scheduler.health()['state'] == 'ready'
            assert scheduler.health()['running_seconds'] == 0.0
            assert not engine.entered.is_set()
            release.set()
            result = await task
            assert result.queue_wait_s >= 0.01
            assert scheduler.health()['state'] == 'ready'
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await scheduler.close()
            occupied.result(timeout=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_slow_warns_but_keeps_queued_jobs_and_accepts_more_work():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0,
                                       slow_inference_seconds=0.01,
                                       stalled_inference_seconds=1)
        tasks = [asyncio.create_task(scheduler.generate('a', 'a', 1))]
        try:
            await eventually(lambda: scheduler.health()['state'] == 'slow')
            assert scheduler.health()['accepting']
            tasks.append(asyncio.create_task(scheduler.generate('b', 'b', 2)))
            await eventually(lambda: scheduler.state('b') == 'queued')
            assert not any(task.done() for task in tasks)
            assert scheduler.health()['queued_requests'] == 1
            engine.release.set()
            results = await asyncio.gather(*tasks)
            assert [item.result for item in results] == [('a', 1), ('b', 2)]
            assert scheduler.health()['state'] == 'ready'
        finally:
            engine.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_stalled_closes_only_queued_work_holds_inflight_and_recovers_after_success():
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0,
                                       slow_inference_seconds=0.01,
                                       stalled_inference_seconds=0.06)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        second = None
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            # Do not call health() to trigger mutation: the pump must wake this
            # queued client itself when the genuine running batch crosses 60ms.
            await eventually(second.done)
            with pytest.raises(SchedulerUnavailable) as error:
                await second
            assert error.value.code == 'engine_stalled'
            health = scheduler.health()
            assert health['state'] == 'stalled' and not health['accepting']
            assert health['running_seconds'] >= 0.06
            assert health['queued_requests'] == 0
            assert scheduler.state('a') == 'running' and engine.active == 1
            assert not first.done()
            with pytest.raises(SchedulerUnavailable) as new_error:
                await scheduler.generate('c', 'c', 3)
            assert new_error.value.code == 'engine_stalled'
            engine.release.set()
            assert (await first).result == ('a', 1)
            assert scheduler.health()['state'] == 'ready'
            assert scheduler.health()['accepting']
            assert (await scheduler.generate('c', 'c', 3)).result == ('c', 3)
            assert engine.batches == [[('a', 1)], [('c', 3)]]
        finally:
            engine.release.set()
            await asyncio.gather(first, *([second] if second is not None else []),
                                 return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_wait_reply_and_empty_translation_never_drive_health():
    class Engine:
        def generate_batch(self, jobs):
            return ['<|WAIT|>' for _ in jobs]

    async def run(executor):
        scheduler = InferenceScheduler(Engine(), executor, max_batch_size=1, batch_wait_ms=0,
                                       slow_inference_seconds=0.01,
                                       stalled_inference_seconds=0.03)
        try:
            for _ in range(10):
                assert (await scheduler.generate('a', 'a', 1)).result == '<|WAIT|>'
            await asyncio.sleep(0.04)
            assert scheduler.health()['state'] == 'ready'
            assert scheduler.health()['running_seconds'] == 0.0
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_typed_backend_error_latches_and_wakes_queued_with_one_diagnostic_log(caplog):
    class Engine(BatchEngine):
        def generate_batch(self, jobs):
            super().generate_batch(jobs)
            try:
                raise ValueError('original CUDA diagnostic')
            except ValueError as cause:
                error = ModelBackendError('batch unavailable')
                error.__cause__ = cause
            return [error for _ in jobs]

    async def run(executor):
        engine = Engine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)
        first = asyncio.create_task(scheduler.generate('a', 'a', 1))
        second = None
        try:
            await eventually(engine.entered.is_set)
            second = asyncio.create_task(scheduler.generate('b', 'b', 2))
            await eventually(lambda: scheduler.state('b') == 'queued')
            engine.release.set()
            outcomes = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(item, SchedulerUnavailable) for item in outcomes)
            assert all(item.code == 'engine_failed' for item in outcomes)
            assert isinstance(outcomes[0].__cause__, ModelBackendError)
            assert scheduler.health()['state'] == 'failed'
            assert not scheduler.health()['accepting']
            assert scheduler.health()['queued_requests'] == 0
            assert scheduler.state('a') == 'idle'
            assert scheduler.state('b') == 'idle'
            assert engine.batches == [[('a', 1)]]
            for _ in range(3):
                with pytest.raises(SchedulerUnavailable):
                    await scheduler.generate('c', 'c', 3)
            failures = [record for record in caplog.records
                        if record.message == 'Shared inference service entered failed state']
            assert len(failures) == 1
            assert failures[0].exc_info is not None
            assert 'original CUDA diagnostic' in caplog.text
        finally:
            engine.release.set()
            await asyncio.gather(first, *([second] if second is not None else []),
                                 return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_unexpected_pump_failure_latches_health_and_wakes_queued(monkeypatch):
    async def run(executor):
        scheduler = InferenceScheduler(BatchEngine(), executor)

        async def broken_dispatch():
            raise RuntimeError('unexpected scheduler failure')

        monkeypatch.setattr(scheduler, '_dispatch', broken_dispatch)
        try:
            with pytest.raises(SchedulerUnavailable) as error:
                await scheduler.generate('a', 'a', 1)
            assert error.value.code == 'engine_failed'
            assert scheduler.health()['state'] == 'failed'
            assert not scheduler.health()['accepting']
        finally:
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_health_poll_failure_does_not_release_real_worker_early(monkeypatch):
    async def run(executor):
        engine = BatchEngine(blocked=True)
        scheduler = InferenceScheduler(engine, executor, max_batch_size=1, batch_wait_ms=0)

        async def broken_wait(worker):
            await eventually(engine.entered.is_set)
            raise RuntimeError('health polling failure')

        monkeypatch.setattr(scheduler, '_wait_worker', broken_wait)
        task = asyncio.create_task(scheduler.generate('a', 'a', 1))
        try:
            await eventually(lambda: scheduler.health()['state'] == 'failed')
            assert scheduler.state('a') == 'running'
            assert engine.active == 1 and not task.done()
            scheduler._task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            engine.release.set()
            with pytest.raises(SchedulerUnavailable):
                await task
            assert scheduler.state('a') == 'idle'
            assert engine.active == 0
            assert scheduler.health()['state'] == 'failed'
        finally:
            engine.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await scheduler.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_normal_close_is_not_a_failed_engine():
    async def run():
        scheduler = InferenceScheduler(object(), None)
        scheduler.start()
        await scheduler.close()
        health = scheduler.health()
        assert health['state'] == 'ready'
        assert not health['accepting']
    asyncio.run(run())
