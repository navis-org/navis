#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""The submitit backend, for SLURM and friends.

Unlike every other backend, submitit's `Job` is not a `Future`: there is no
`concurrent.futures` shim, and results arrive via a polling loop rather than a
callback - a job may spend hours in a queue before it starts. Hence a backend
of its own.

The unit of work is correspondingly expensive. One neuron per job would mean
queueing 10,000 jobs; the chunking policy bundles them instead.

"""

from ... import config
from .base import ParallelBackend, apply_overrides

logger = config.get_logger(__name__)

__all__ = ['SubmititBackend']

#: Seconds between checks on the queue. submitit's own default, and about
#: right for a scheduler where a job's lifetime is minutes to hours.
POLL_FREQUENCY = 10

#: ... but submitit also has executors that run on this machine, where a job
#: lasts seconds and a 10s poll would be most of the runtime.
LOCAL_POLL_FREQUENCY = 0.25
LOCAL_EXECUTORS = ('DebugExecutor', 'LocalExecutor')

#: Why a bare `SubmititBackend` cannot run anything. Reported through
#: `unsupported()`, so `set_parallel_backend('submitit')` fails at the point
#: the mistake was made rather than at the next `parallel=True`.
NO_EXECUTOR = (
    "'submitit' needs an executor - it has no way to guess a log folder, "
    'partition or walltime. Build one and hand it over:\n'
    '    import submitit\n'
    "    ex = submitit.AutoExecutor(folder='logs')\n"
    "    ex.update_parameters(slurm_partition='cpu', timeout_min=60)\n"
    '    navis.set_parallel_backend(ex)'
)


def _underlying(executor):
    """Unwrap an `AutoExecutor` to the executor it actually dispatches to."""
    return getattr(executor, '_executor', executor)


class SubmititBackend(ParallelBackend):
    """Run each unit of work as a job on a cluster scheduler.

    Never selected automatically, and needs an executor: submitit cannot guess
    a log folder, let alone a partition or a walltime. Configure one with
    submitit's own API and hand it over::

        import submitit

        ex = submitit.AutoExecutor(folder='logs')
        ex.update_parameters(slurm_partition='cpu', timeout_min=60, mem_gb=8)

        with navis.set_parallel_backend(ex):
            navis.prune_twigs(nl, 5, parallel=True)

    Note that `n_cores` here only decides how the neurons are *split up* - how
    many of those jobs run at once is the scheduler's business, set on the
    executor (e.g. `slurm_array_parallelism`).

    Also works against submitit's local executors (`cluster='local'` runs jobs
    as subprocesses, `cluster='debug'` inline), which is the way to check a
    pipeline before putting it on the queue.
    """

    name = 'submitit'
    auto_select = False
    requires = 'submitit'
    adopts = ('submitit',)

    isolated = True
    pickles_by_value = True     # cloudpickle
    # What a job may use is the scheduler's business - `cpus_per_task` on the
    # executor, not the `n_cores` a caller passed on the submitting machine.
    shares_machine = False
    # submitit records a failed job as a *string* - the worker traceback - and
    # `job.result()` always raises `FailedJobError`, whatever went wrong. The
    # dispatcher works around it so callers still see their own exception.
    marshals_exceptions = False
    # A unit of work is a queued job, so units are far more expensive than on
    # dask. Two per worker: enough that one slow neuron doesn't strand a whole
    # job's worth of work, few enough to keep the array small.
    chunks_per_worker = 2

    def __init__(self, executor=None, *, poll_frequency=None, **overrides):
        self.executor = executor

        # `AutoExecutor` is a facade - what it dispatches to is what tells us
        # how the work actually runs.
        kind = type(_underlying(executor)).__name__

        if executor is not None:
            # `DebugExecutor` runs the job inline, in this interpreter, so
            # forcing `inplace=True` there would corrupt the caller's neurons.
            self.isolated = kind != 'DebugExecutor'

        if poll_frequency is None:
            poll_frequency = (LOCAL_POLL_FREQUENCY if kind in LOCAL_EXECUTORS
                              else POLL_FREQUENCY)
        self.poll_frequency = poll_frequency

        apply_overrides(self, **overrides)

    def _adopt(self, obj, **overrides):
        import submitit

        if isinstance(obj, submitit.Executor):
            return SubmititBackend(obj, **overrides)
        return None

    def unsupported(self, **requirements):
        reasons = super().unsupported(**requirements)
        if self.executor is None:
            reasons.append(NO_EXECUTOR)
        return reasons

    def map(self, func, payloads, *, n_workers, threads=None):
        import submitit

        executor = self.executor
        if executor is None:
            # `unsupported()` catches this on the way in, but a backend
            # constructed directly is handed straight back by `resolve_backend`
            raise ValueError(NO_EXECUTOR)

        # One array job for the lot: `map_array` batches the submission, which
        # matters because submitting N jobs individually is N scheduler round
        # trips and N times the queueing overhead.
        jobs = executor.map_array(func, list(payloads))
        logger.debug(f'Submitted {len(jobs)} job(s) via {type(executor).__name__}.')

        try:
            for job in submitit.helpers.as_completed(
                    jobs, poll_frequency=self.poll_frequency):
                # Re-raises whatever the job raised, with the worker traceback
                yield job.result()
        except BaseException:
            for job in jobs:
                try:
                    job.cancel()
                except Exception as e:
                    # Already finished, or the scheduler is unreachable -
                    # either way it must not mask the original error
                    logger.debug(f'Could not cancel job {job}: {e}')
            raise
