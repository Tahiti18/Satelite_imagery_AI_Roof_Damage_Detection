"""
Performance profiling utilities.

These helpers are used to collect detailed timing metrics for the
zipcode-based roof damage detection pipeline. They are intentionally
lightweight and have no external dependencies beyond the standard
library and NumPy.

The goal is:
- Identify bottlenecks (e.g. image fetching vs. detection)
- Track end-to-end latency for different zipcodes
- Enable data-driven optimisation without changing core logic.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Iterator, Optional

import numpy as np


class PerformanceProfiler:
    """Profile performance of different operations.

    This class provides detailed timing information for different parts of
    the pipeline, helping identify bottlenecks and optimisation
    opportunities.

    Example:
        profiler = PerformanceProfiler()
        with profiler.profile("detection"):
            # Perform detection
            ...
        stats = profiler.get_stats()
    """

    def __init__(self) -> None:
        """Initialise the performance profiler with empty timing data."""
        self.timings: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def profile(self, operation: str) -> Iterator[None]:
        """Context manager to profile a specific operation.

        Args:
            operation: Name of the operation being profiled.

        Yields:
            None: Control is yielded back to the caller.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[operation].append(time.perf_counter() - start)

    def add(self, operation: str, duration_sec: float) -> None:
        """Manually record a timing for an operation."""
        self.timings[operation].append(duration_sec)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all profiled operations.

        Returns:
            Dict[str, Dict[str, float]]: Stats for each operation including
            mean, std, min, max and count in seconds.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for op, times in self.timings.items():
            if not times:
                continue
            arr = np.asarray(times, dtype=float)
            stats[op] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(arr.size),
            }
        return stats


@dataclass
class ProcessingMetrics:
    """Tracks processing performance metrics for a single analysis run.

    This class aggregates performance data throughout the zipcode
    processing pipeline, enabling detailed performance analysis and
    optimisation.

    Attributes:
        zipcode: Zipcode being processed.
        start_time: Timestamp when processing started.
        end_time: Timestamp when processing ended.
        stage_durations: Map of stage name -> duration in seconds.
        profiler_stats: Optional low-level stats from PerformanceProfiler.
    """

    zipcode: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stage_durations: Dict[str, float] = field(default_factory=dict)
    profiler_stats: Optional[Dict[str, Dict[str, float]]] = None

    def mark_stage(self, name: str, duration_sec: float) -> None:
        """Record the duration of a single logical stage."""
        self.stage_durations[name] = duration_sec

    @property
    def total_time_sec(self) -> float:
        """Total wall-clock time for the run."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, object]:
        """Serialise metrics to a dictionary for JSON/GeoJSON output."""
        return {
            "zipcode": self.zipcode,
            "total_time_sec": float(self.total_time_sec),
            "stages": {k: float(v) for k, v in self.stage_durations.items()},
            "profiler": self.profiler_stats or {},
        }


__all__ = ["PerformanceProfiler", "ProcessingMetrics"]
