"""Helpers for memory-limit demos.

These functions do not toggle system swap. They cap this process's virtual
address space so allocations fail before the OS can back them with swap.

POSIX uses ``resource.setrlimit(RLIMIT_AS, ...)``.
Windows uses a Job Object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY``.
"""

from __future__ import annotations

import sys

import psutil

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import win32api  # type: ignore[import-not-found]
    import win32job  # type: ignore[import-not-found]

    _job = None
else:
    import resource # type: ignore[import-not-found]

    _original_soft_limit = None


def disable_swap(fraction: float = 0.9) -> int:
    """Cap this process's memory to ``fraction`` of physical RAM."""
    total_ram = psutil.virtual_memory().total
    limit_bytes = int(total_ram * fraction)

    if _IS_WINDOWS:
        global _job
        _job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            _job, win32job.JobObjectExtendedLimitInformation
        )
        info["ProcessMemoryLimit"] = limit_bytes
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        )
        win32job.SetInformationJobObject(
            _job, win32job.JobObjectExtendedLimitInformation, info
        )
        process_handle = win32api.GetCurrentProcess()
        win32job.AssignProcessToJobObject(_job, process_handle)
    else:
        global _original_soft_limit
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        _original_soft_limit = soft_limit
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard_limit))

    return limit_bytes


def enable_swap() -> None:
    """Remove the process memory cap applied by ``disable_swap()``."""
    if _IS_WINDOWS:
        global _job
        if _job is not None:
            info = win32job.QueryInformationJobObject(
                _job, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] &= ~(
                win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            )
            win32job.SetInformationJobObject(
                _job, win32job.JobObjectExtendedLimitInformation, info
            )
    else:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, hard_limit))
