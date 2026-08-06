"""Helpers for memory-limit demos.

These functions do not toggle system swap. They cap this process's virtual
address space so allocations fail before the OS can back them with swap.

POSIX uses ``resource.setrlimit(RLIMIT_AS, ...)``.
Windows uses a Job Object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY``.
"""

from __future__ import annotations

import sys

import psutil

try:
    import numpy as np
except ImportError:  # pragma: no cover - notebook dependency
    np = None

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import win32api  # type: ignore[import-not-found]
    import win32job  # type: ignore[import-not-found]

    _job = None
else:
    import resource # type: ignore[import-not-found]

    _original_soft_limit = None

_simulated_cap_bytes = None
_cap_mode = "none"


def get_cap_mode() -> str:
    """Return active cap mode: 'os', 'simulated', or 'none'."""
    return _cap_mode


def disable_swap(fraction: float = 0.9) -> int:
    """Cap this process's memory to ``fraction`` of physical RAM."""
    global _cap_mode
    global _simulated_cap_bytes

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the range (0, 1].")

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
        _cap_mode = "os"
        _simulated_cap_bytes = None
    else:
        global _original_soft_limit
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        _original_soft_limit = soft_limit
        if hard_limit == resource.RLIM_INFINITY:
            applied_limit = limit_bytes
        else:
            applied_limit = min(limit_bytes, hard_limit)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (applied_limit, hard_limit))
            _cap_mode = "os"
            _simulated_cap_bytes = None
        except ValueError as exc:
            if sys.platform == "darwin":
                _simulated_cap_bytes = limit_bytes
                _cap_mode = "simulated"
            else:
                raise RuntimeError(
                    "This runtime rejected RLIMIT_AS updates; process memory capping "
                    "is not available in this environment."
                ) from exc
        limit_bytes = applied_limit

    return limit_bytes


def enable_swap() -> None:
    """Remove the process memory cap applied by ``disable_swap()``."""
    global _cap_mode
    global _simulated_cap_bytes

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
        _cap_mode = "none"
        _simulated_cap_bytes = None
    else:
        global _original_soft_limit
        if _cap_mode == "simulated":
            _simulated_cap_bytes = None
            _cap_mode = "none"
            return

        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)

        target_soft = (
            _original_soft_limit
            if _original_soft_limit is not None
            else resource.RLIM_INFINITY
        )
        if hard_limit != resource.RLIM_INFINITY and target_soft > hard_limit:
            target_soft = hard_limit

        resource.setrlimit(resource.RLIMIT_AS, (target_soft, hard_limit))
        _original_soft_limit = None
        _cap_mode = "none"
        _simulated_cap_bytes = None


def random_array_float64(n_elements: int):
    """Allocate float64 random array with cap-aware behavior on all platforms."""
    if np is None:
        raise RuntimeError("NumPy is required for random array allocation demos.")

    if n_elements < 0:
        raise ValueError("n_elements must be >= 0")

    if _cap_mode == "simulated" and _simulated_cap_bytes is not None:
        requested_bytes = n_elements * np.dtype(np.float64).itemsize
        rss_bytes = psutil.Process().memory_info().rss
        if rss_bytes + requested_bytes > _simulated_cap_bytes:
            raise MemoryError(
                "Simulated cap hit on this platform: requested allocation would "
                "exceed process memory cap."
            )

    return np.random.random(n_elements)
