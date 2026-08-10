"""Helpers for memory-limit demos."""

from __future__ import annotations

import numpy as np
import psutil


class SimulateArrayLimits:
    def __init__(self):
        self.limit_bytes = None

    def enable_memory_limit(self, fraction: float):
        """Enable a simulated memory limit,
        and set it to ``fraction`` of physical RAM."""
        if fraction > 0 and not 0 < fraction <= 1:
            raise ValueError("fraction must be in the range (0, 1].")

        total_ram = psutil.virtual_memory().total
        self.limit_bytes = int(total_ram * fraction)
        return self.limit_bytes

    def disable_memory_limit(self):
        """Disable simulated memory limit."""
        self.limit_bytes = None
        return self.limit_bytes

    def memory_limit_gib(self):
        return self.limit_bytes / 1024**3

    def allocate_random_array(self, shape: tuple[int]) -> None:
        """Create an Float64 array of ``shape``
        if it fits into currently allowed limit.
        Raises ``MemoryError`` if limit exceeded."""

        if self.limit_bytes:
            requested_bytes = np.prod(shape) * np.dtype(np.float64).itemsize
            if requested_bytes > self.limit_bytes:
                raise MemoryError(
                    f"Not enough memory. Memory is capped to "
                    f"{self.limit_bytes / 1024**3:.2f} GiB, but "
                    f"you have requested {requested_bytes / 1024**3:.2f} GiB."
                )

        return np.random.random(shape)


simulated_array_limits = SimulateArrayLimits()
