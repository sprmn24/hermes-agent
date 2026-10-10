"""Physical worker liveness and cooperative Stop across every served authority."""
import asyncio

from gateway.session_authorities import all_authorities


def executing_sessions(authority):
    for sid, live in getattr(authority, 'sessions', {}).items():
        execution = getattr(getattr(live, 'event_stream', None), 'execution', None) or {}
        task = live.task
        if execution.get('execution_generation') is not None and task is not None and not task.done():
            yield sid, live, execution['execution_generation']


def claiming_sessions(authority):
    """Sessions whose claim write is still in flight off the loop (``LiveSession.claiming``)."""
    return [sid for sid, live in getattr(authority, 'sessions', {}).items() if getattr(live, 'claiming', False)]


def authority_writes(runner):
    """Claim writes and tracked mutations still in flight. Both write state.db from the loop's
    default executor, which the shutdown executor quiesce never joins."""
    return sum(len(claiming_sessions(authority)) + len(mutation_tasks(authority))
               for authority in all_authorities(runner))


def uncounted_runtime_work(runner):
    """Count claims absent from the ordinary agent map, including pre-bootstrap managed turns and
    claim writes still in flight."""
    running = getattr(runner, '_running_agents', {})
    authorities = all_authorities(runner)
    managed = sum(not worker.closed.is_set() for authority in authorities
                  for worker in getattr(authority, '_managed_workers', {}).values())
    return (managed + sum(live.route not in running and sid not in getattr(authority, '_managed_workers', {})
                for authority in authorities for sid, live, _ in executing_sessions(authority))
            + authority_writes(runner))


def persist_work_count(authority, _task=None):
    """Re-publish the runner's in-flight count when owner work it counts (a drain, a tracked
    mutation, a mailbox operation) ends: the agent-slot release that last wrote
    ``gateway_state.json`` ran while that work was still live, so without this write the
    persisted count never returns to zero."""
    persist = getattr(getattr(authority, 'runner', None), '_persist_active_agents', None)
    if persist is not None:
        persist()


def mutation_tasks(authority):
    tasks = {*getattr(authority, '_mutation_tasks', ()), *getattr(authority, '_bot_mailbox_operations', ())}
    return [task for task in tasks if not task.done()]


def track_mutation(authority, operation):
    tasks = getattr(authority, '_mutation_tasks', None)
    if tasks is None:
        tasks = authority._mutation_tasks = set()
    task = asyncio.create_task(operation)
    tasks.add(task)
    def finished(done):
        tasks.discard(done)
        if not done.cancelled():
            done.exception()  # the observer may already have disconnected
        persist_work_count(authority)
    task.add_done_callback(finished)
    return task


def start_turn_worker(runner, worker, agent_holder, run_sync):
    authority = track_turn_worker(runner, worker, agent_holder)
    worker.executor_task = asyncio.ensure_future(runner._run_in_executor_with_context(run_sync))
    if authority is not None:
        # Normal settlement releases the retained agent promptly. A cancelled asyncio waiter is
        # not physical completion; leave that entry until worker_done proves the thread exited.
        worker.executor_task.add_done_callback(lambda _: list(physical_workers(authority)))


def track_turn_worker(runner, worker, agent_holder):
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is not None:
        workers = getattr(authority, '_turn_workers', None)
        if workers is None:
            workers = authority._turn_workers = {}
        list(physical_workers(authority))
        workers[id(worker)] = (worker, agent_holder)
    return authority


def physical_workers(authority):
    workers = getattr(authority, '_turn_workers', {})
    for key, (worker, holder) in list(workers.items()):
        if worker.worker_done.is_set():
            del workers[key]
        else:
            yield worker, holder


def stop_authority_work(authority):
    from gateway.run_runtime import stop_authority_turns
    from agent.interrupt_compat import request_hard_interrupt
    stop_authority_turns(authority, in_process=True)
    for _, holder in physical_workers(authority):
        if holder and holder[0] is not None:
            request_hard_interrupt(holder[0], 'Profile runtime is stopping')
    for cancel in getattr(authority, '_cron_cancellations', {}).values():
        cancel.set()


async def join_authority_work(authority, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        stop_authority_work(authority)
        if not (list(physical_workers(authority)) or list(executing_sessions(authority))
                or claiming_sessions(authority) or mutation_tasks(authority)):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError('Profile workers did not stop; ownership retained')
        await asyncio.sleep(.05)
