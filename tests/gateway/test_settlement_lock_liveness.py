"""A settlement waiting on a contended SQLite writer never stalls the owner event loop.

The settlement worker holds no stream lock across its write, so loop-side attach/replay/pending
publication keep running; the commit still looks atomic with its completion frame to them.
"""
import asyncio
import sqlite3
import threading
import time

import pytest

from gateway.session_contract import Principal, Submission
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority

_HOLD_S = 1.5
VIEWER = Principal('human', 'owned', frozenset({'session:read', 'session:submit'}), 'viewer')


async def _settling_behind_a_held_writer(tmp_path, monkeypatch):
    """Run a turn whose settlement write waits on a real competing SQLite writer."""
    from gateway import session_results
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames, entered, released = [], threading.Event(), threading.Event()
    authority.sessions['s'].event_stream.observers.add(frames.append)
    settle = session_results.settle_session_input

    def contended_settle(*args, **kwargs):
        entered.set()
        return settle(*args, **kwargs)
    monkeypatch.setattr(session_results, 'settle_session_input', contended_settle)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        # Another process holds the write lock while this turn settles.
        writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
        writer.execute('BEGIN IMMEDIATE')

        def release():
            time.sleep(_HOLD_S)
            released.set()
            writer.rollback()
            writer.close()
        threading.Thread(target=release, daemon=True).start()
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    await authority.submit(ACTOR, Submission('work', REF, {'text': 'work'}, 'queue'))
    drain = asyncio.create_task(authority._drain(REF))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    return db, authority, frames, released, drain


def _types(frames):
    return [(f['params']['type'], f['params']['payload'].get('running')) for f in frames]


@pytest.mark.asyncio
async def test_attach_during_a_contended_settlement_leaves_the_loop_free(tmp_path, monkeypatch):
    db, authority, frames, released, drain = await _settling_behind_a_held_writer(tmp_path, monkeypatch)
    with db:
        ticks = []
        asyncio.get_running_loop().call_later(0.05, lambda: ticks.append(released.is_set()))
        attach = asyncio.create_task(authority.attach(VIEWER, REF))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'an unrelated 50 ms callback waited for the held writer'
        snapshot = await asyncio.wait_for(attach, 10)
        await asyncio.wait_for(drain, 10)
        # The snapshot still never shows the terminal row without its completion frame.
        complete = next(f['params']['seq'] for f in frames if f['params']['type'] == 'message.complete')
        if snapshot.handle.execution_state == 'idle':
            assert snapshot.last_sequence >= complete
        else:
            assert snapshot.last_sequence < complete


@pytest.mark.asyncio
async def test_pending_publication_during_a_contended_settlement_keeps_idle_after_completion(
        tmp_path, monkeypatch):
    db, authority, frames, released, drain = await _settling_behind_a_held_writer(tmp_path, monkeypatch)
    with db:
        ticks = []
        loop = asyncio.get_running_loop()
        # A loop-side publisher (submit/cancel/automation call this) racing the settlement.
        loop.call_later(0.02, authority._publish_pending, REF)
        loop.call_later(0.05, lambda: ticks.append(released.is_set()))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'pending publication waited for the held writer'
        await asyncio.wait_for(drain, 10)
        kinds = _types(frames)
        start, complete = kinds.index(('message.start', None)), kinds.index(('message.complete', None))
        assert ('session.info', False) not in kinds[start:complete], kinds
        assert ('session.info', False) in kinds[complete:], kinds


@pytest.mark.asyncio
async def test_cancel_behind_a_held_writer_leaves_the_loop_free_and_still_settles_its_observer(
        tmp_path, monkeypatch):
    """``prompt.cancel`` / Delete on a queued card: its write waits on the writer, never on the loop."""
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        receipt = await authority.submit(ACTOR, Submission('queued', REF, {'text': 'queued'}, 'queue'))
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        released = threading.Event()
        writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
        writer.execute('BEGIN IMMEDIATE')

        def release():
            time.sleep(_HOLD_S)
            released.set()
            writer.rollback()
            writer.close()
        threading.Thread(target=release, daemon=True).start()
        ticks = []
        asyncio.get_running_loop().call_later(0.05, lambda: ticks.append(released.is_set()))
        cancel = asyncio.create_task(authority.cancel_queued(ACTOR, REF, receipt.admission_id))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'an unrelated 50 ms callback waited for the held writer'
        cancelled = await asyncio.wait_for(cancel, 10)
        assert (cancelled.status, cancelled.outcome) == ('terminal', 'cancelled')
        assert waiter.done() and receipt.admission_id not in authority.cancel_obligations


@pytest.mark.asyncio
async def test_claim_retried_behind_a_held_writer_leaves_the_loop_free(tmp_path, monkeypatch):
    """The drain's claim is a write too, retried in place while another process holds the writer.
    Each attempt waits out the write patience, so it must wait off the loop, or every retry
    freezes every session for that long."""
    from gateway import session_finite
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(db, '_WRITE_PATIENCE_S', 0.5)
    ran = []

    async def execute(owner, ref, row):
        ran.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        head = await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        waiter = authority.waiters.setdefault(head.admission_id, asyncio.get_running_loop().create_future())
        held, released = threading.Event(), threading.Event()

        def hold_writer():
            writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
            writer.execute('BEGIN IMMEDIATE')
            held.set()
            time.sleep(2 * _HOLD_S)  # several claim attempts at this patience
            released.set()
            writer.rollback()
            writer.close()
        threading.Thread(target=hold_writer, daemon=True).start()
        held.wait(5)
        schedule(REF)
        loop, late = asyncio.get_running_loop(), []
        while not released.is_set():
            due, fired = loop.time() + 0.05, loop.create_future()
            loop.call_at(due, fired.set_result, None)
            await fired
            late.append(loop.time() - due)
        assert await asyncio.wait_for(waiter, 10) == 'done'
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert ran == ['head']
    assert max(late) < 0.25, f'an unrelated 50 ms timer ran {max(late):.2f}s late while the claim waited'


@pytest.mark.asyncio
async def test_input_admitted_while_an_idle_claim_runs_is_not_stranded(tmp_path, monkeypatch):
    """An idle drain's claim found nothing; an input admitted before the drain ends finds the drain
    still running, so its own _schedule starts nothing. The drain must pick it up, not end. Native,
    automation, cron and API admission take no mutation_lock, so nothing else serializes them."""
    from gateway import session_authority, session_finite
    from hermes_state_runtime import admit_session_input, get_session_admission
    db, authority = _authority(tmp_path, monkeypatch)
    ran, claimed, admitted = [], threading.Event(), threading.Event()
    claim = session_authority.claim_session_input

    def claim_then_an_input_lands(*args, **kwargs):
        row = claim(*args, **kwargs)
        if row is None and not admitted.is_set():
            claimed.set()
            admitted.wait(1)
        return row
    monkeypatch.setattr(session_authority, 'claim_session_input', claim_then_an_input_lands)

    async def execute(owner, ref, row):
        ran.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        authority._schedule(REF)  # nothing is queued yet
        await asyncio.get_running_loop().run_in_executor(None, claimed.wait, 5)
        # What admit_native does: commit, publish, schedule (a no-op while the drain still runs).
        late = admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s',
                                   request_id='late', payload={'text': 'late'})
        authority._publish_pending(REF)
        authority._schedule(REF)
        admitted.set()
        async with asyncio.timeout(5):
            while get_session_admission(db, admission_id=late['admission_id'])['status'] != 'terminal':
                await asyncio.sleep(0.01)
    assert ran == ['late']


def _hold_writer_until_after(tmp_path, claim_started, seconds):
    """Another process's state.db writer: held from now, released ``seconds`` after the claim starts.
    Polls instead of a timed Event.wait, so a wall-clock step cannot cut the hold short."""
    held = threading.Event()

    def hold():
        writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
        writer.execute('BEGIN IMMEDIATE')
        held.set()
        while not claim_started.is_set():
            time.sleep(0.005)
        time.sleep(seconds)
        writer.rollback()
        writer.close()
    threading.Thread(target=hold, daemon=True).start()
    while not held.is_set():
        time.sleep(0.005)


def _claim_behind_a_held_writer(tmp_path, monkeypatch):
    """A queued head, a drain about to claim it, and the writer it will wait on."""
    from gateway import session_authority, session_finite
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(db, '_WRITE_PATIENCE_S', 3.0)
    started, finished, ran = threading.Event(), threading.Event(), []
    claim = session_authority.claim_session_input

    def recorded_claim(*args, **kwargs):
        started.set()
        try:
            return claim(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(session_authority, 'claim_session_input', recorded_claim)

    async def execute(owner, ref, row):
        ran.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    return db, authority, schedule, started, finished, ran


@pytest.mark.asyncio
async def test_profile_retirement_waits_for_a_claim_still_waiting_on_the_writer(tmp_path, monkeypatch):
    """Hot-unserve retires the profile while the drain's claim waits off the loop for another
    process's writer. The claim's thread commits whether or not its drain still exists, so retirement
    must not cancel the drain under it: this owner runs (or Stops) and settles the head it claimed,
    never leaving it started for the next owner to recover as unknown."""
    from gateway import run_runtime
    from hermes_state_runtime import list_session_admissions
    db, authority, schedule, started, finished, ran = _claim_behind_a_held_writer(tmp_path, monkeypatch)
    with db:
        await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        _hold_writer_until_after(tmp_path, started, 1.0)
        schedule(REF)
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert await asyncio.wait_for(run_runtime._retire_profile_authority(authority), 10) is True
        while not finished.is_set():  # wherever the claim ran, its write has committed or failed now
            await asyncio.sleep(0.01)
        row, = list_session_admissions(db, session_id='s', pending_only=False)
    assert (row['status'], ran) == ('terminal', ['head']), (
        f"retirement left the claimed head {row['status']!r} (ran {ran}); the next owner recovers it as unknown")


@pytest.mark.asyncio
async def test_a_claim_waiting_on_the_writer_is_live_authority_work(tmp_path, monkeypatch):
    """Shutdown's active-work count and its final join see a claim still waiting off the loop, from
    the write's start until the drain has consumed the result, like the turn it starts."""
    from gateway.session_runtime_workers import join_authority_work, uncounted_runtime_work
    db, authority, schedule, started, finished, ran = _claim_behind_a_held_writer(tmp_path, monkeypatch)
    authority.runner.session_authority = authority
    with db:
        await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        _hold_writer_until_after(tmp_path, started, 1.0)
        schedule(REF)
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert uncounted_runtime_work(authority.runner) >= 1, 'the head being claimed is invisible to shutdown'
        with pytest.raises(TimeoutError):
            await join_authority_work(authority, 0)
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert uncounted_runtime_work(authority.runner) == 0
        await join_authority_work(authority, 0)
    assert ran == ['head']


_CLAIM_POINTS = ('inside-the-transaction', 'after-the-commit')


def _wait_for(event, seconds=20.0):
    """A barrier wait that polls instead of trusting a timed Event.wait to the host clock."""
    deadline = time.monotonic() + seconds
    while not event.is_set():
        assert time.monotonic() < deadline, 'barrier never reached'
        time.sleep(0.002)


def _record_turns(monkeypatch):
    from gateway import session_finite
    ran = []

    async def execute(owner, ref, row):
        ran.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    return ran


def _park_the_claim(monkeypatch, point):
    """Hold the thread of the claim that takes the head at ``point`` until released: inside its
    transaction before COMMIT, or after the commit before the drain has the result."""
    from gateway import session_authority
    claim = session_authority.claim_session_input
    parked, release, finished = threading.Event(), threading.Event(), threading.Event()

    def hold(here):
        if here == point and not parked.is_set():
            parked.set()
            _wait_for(release)

    class _ParkingDB:
        def __init__(self, db):
            self._db = db

        def __getattr__(self, name):
            return getattr(self._db, name)

        def _execute_write(self, fn, patience_s=None):
            def parked_write(conn):
                row = fn(conn)
                if row:
                    hold('inside-the-transaction')
                return row
            return self._db._execute_write(parked_write, patience_s)

    def parked_claim(db, **kwargs):
        try:
            row = claim(_ParkingDB(db), **kwargs)
            if row:
                hold('after-the-commit')
            return row
        finally:
            if parked.is_set():
                finished.set()
    monkeypatch.setattr(session_authority, 'claim_session_input', parked_claim)
    return parked, release, finished


@pytest.mark.asyncio
@pytest.mark.parametrize('point', _CLAIM_POINTS)
async def test_profile_retirement_never_cancels_the_drain_under_its_claim_write(tmp_path, monkeypatch, point):
    """Retirement cancels a profile's session tasks only once join_authority_work finds no work, and
    a claim write is work from its start: with the thread parked inside the transaction or after
    the commit, the drain is not cancelled under it, and the head it claimed runs and settles."""
    from gateway import run_runtime, session_runtime_workers
    from hermes_state_runtime import list_session_admissions
    db, authority = _authority(tmp_path, monkeypatch)
    ran = _record_turns(monkeypatch)
    parked, release, _ = _park_the_claim(monkeypatch, point)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    stop, polls, polled = session_runtime_workers.stop_authority_work, [], asyncio.Event()

    def counted_stop(owner):  # once per join poll
        polls.append(owner)
        if len(polls) == 3:
            polled.set()
        return stop(owner)
    monkeypatch.setattr(session_runtime_workers, 'stop_authority_work', counted_stop)
    loop = asyncio.get_running_loop()
    with db:
        await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        schedule(REF)
        drain = authority.sessions['s'].task
        await loop.run_in_executor(None, _wait_for, parked)
        retirement = asyncio.create_task(run_runtime._retire_profile_authority(authority))
        third_poll = asyncio.create_task(polled.wait())
        await asyncio.wait({retirement, third_poll}, return_when=asyncio.FIRST_COMPLETED, timeout=20)
        third_poll.cancel()
        assert not retirement.done() and not drain.done(), 'retirement cancelled the drain under its claim write'
        release.set()
        assert await asyncio.wait_for(retirement, 20) is True
        row, = list_session_admissions(db, session_id='s', pending_only=False)
    assert ((row['status'], row['outcome']), ran) == (('terminal', 'completed'), ['head'])


@pytest.mark.parametrize('point', _CLAIM_POINTS)
def test_a_cancel_landing_mid_claim_write_settles_the_head_unrun(tmp_path, monkeypatch, point):
    """asyncio.run's final sweep cancels every task still alive when start_gateway returns, after
    settle_gateway_runtime stopped waiting for a claim still in flight: the one canceller that does
    not join claims first. The drain waits for the write and stays counted meanwhile; the head it
    claimed ends terminal unrun (interrupted, as a Stop before bootstrap leaves it) and its waiter is
    released. Never started for the next owner to recover as unknown, never requeued."""
    from gateway import run_runtime
    from hermes_state_runtime import list_session_admissions
    db, authority = _authority(tmp_path, monkeypatch)
    authority.runner.session_authority = authority
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 0.1)  # settle's wait; the barrier holds the claim
    ran = _record_turns(monkeypatch)
    parked, release, finished = _park_the_claim(monkeypatch, point)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    swept = {}

    async def start_gateway():
        loop = asyncio.get_running_loop()
        receipt = await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        swept['waiter'] = authority.waiters.setdefault(receipt.admission_id, loop.create_future())
        schedule(REF)
        live = authority.sessions['s']
        await loop.run_in_executor(None, _wait_for, parked)

        def after_the_sweep():
            swept.update(claiming=live.claiming, write_finished=finished.is_set())
            release.set()

        async def sentinel():
            # The sweep cancels every task before it runs any, so a callback this handler schedules
            # runs after the drain has handled its own cancel.
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop.call_soon(after_the_sweep)
                raise
        asyncio.create_task(sentinel())
        await run_runtime.settle_gateway_runtime(authority.runner)
    with db:
        asyncio.run(start_gateway())
        row, = list_session_admissions(db, session_id='s', pending_only=False)
    assert swept['claiming'] and not swept['write_finished'], 'the cancelled drain stopped counting a write in flight'
    assert ((row['status'], row['outcome']), ran) == (('terminal', 'interrupted'), [])
    assert swept['waiter'].result() == ''
