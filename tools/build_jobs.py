"""Bound compiler parallelism; an explicit positive job count overrides the heuristic."""
import os
from pathlib import Path
import re


def build_jobs():
    value = os.environ.get('PS5_HIP_BUILD_JOBS')
    if value is not None:
        if not re.fullmatch(r'[1-9][0-9]*', value) or not 1 <= int(value) <= 64:
            raise ValueError('PS5_HIP_BUILD_JOBS (Make JOBS) must be an integer from 1 to 64')
        return int(value)
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    try:
        text = Path('/proc/meminfo').read_text()
        available = int(re.search(r'^MemAvailable:\s+(\d+) kB$', text, re.M).group(1)) * 1024
        # Leave 2 GiB for the system; reserve roughly 2 GiB per compiler worker.
        memory_jobs = max(1, (available - 2 * 1024**3) // (2 * 1024**3))
    except (OSError, AttributeError, ValueError):
        memory_jobs = 2
    return max(1, min(cpus, memory_jobs, 8))


if __name__ == '__main__':
    print(build_jobs())
