"""Host resource numbers for the header readout: what memory is in use, and
how much of it is this process.

Deliberately not part of `/health`. Health answers "can I serve a trace at
all", is polled once at start-up and is cheap; this is polled on a timer for
as long as a browser tab is open, and is allowed to fail without that meaning
anything about readiness — every field is optional for exactly that reason.

Unified memory is the reason this is not the usual "RAM bar, VRAM bar" pair.
On a GB10/Jetson-class part the CPU and the GPU address one physical pool, so
`torch.cuda.mem_get_info()[1]` and `psutil.virtual_memory().total` report the
same bytes. Drawing those as two independent gauges would claim headroom that
does not exist, so `unified` is measured and published and the interface draws
one pool when it is true.
"""

from __future__ import annotations

import subprocess
import time

import psutil

# Within 5% counts as the same pool. The two APIs disagree by a few MB on the
# same hardware (one reports what the driver sees, the other what the kernel
# has left after reserving its own regions), so an equality test would answer
# "not unified" on a machine that plainly is.
_UNIFIED_TOLERANCE = 0.05

# nvidia-smi returns in ~20ms, but it is a process spawn on a path the browser
# drives, so a result is reused for this long rather than forked per poll.
_SMI_TTL_SECONDS = 1.0

_smi_cache: tuple[float, int | None] = (0.0, None)


def _gpu_utilisation() -> int | None:
    """Percent of the last sampling period the GPU had work on it, or None if
    nvidia-smi is absent or slow. Unlike memory, there is no torch API for
    this: `torch.cuda.utilization()` needs nvidia-ml-py, which is not a
    dependency here and is not installed alongside torch on every platform."""
    global _smi_cache
    now = time.monotonic()
    cached_at, value = _smi_cache
    if now - cached_at < _SMI_TTL_SECONDS:
        return value

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        value = int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # Includes "[N/A]", which some parts report for fields they do not
        # expose — an unparseable number is a missing number, not a zero.
        value = None

    _smi_cache = (now, value)
    return value


def collect() -> dict[str, object]:
    """Every field beyond the two RAM figures is optional: a CPU-only box has
    no GPU numbers, and a missing number has to stay distinguishable from a
    zero one, or the readout invents headroom or invents pressure."""
    vm = psutil.virtual_memory()
    stats: dict[str, object] = {
        "ram_used_bytes": vm.total - vm.available,
        "ram_total_bytes": vm.total,
        "gpu_name": None,
        "gpu_free_bytes": None,
        "gpu_total_bytes": None,
        "torch_allocated_bytes": None,
        "torch_reserved_bytes": None,
        "gpu_util_pct": _gpu_utilisation(),
        "unified": False,
    }

    # Imported here, not at module scope: torch takes seconds to import and
    # this module is what the FastAPI app imports to define the route. The
    # service loads torch anyway, so by the time anything polls this the
    # import is a dict lookup.
    import torch

    if not torch.cuda.is_available():
        return stats

    try:
        free, total = torch.cuda.mem_get_info()
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["gpu_free_bytes"] = free
        stats["gpu_total_bytes"] = total
        stats["torch_allocated_bytes"] = torch.cuda.memory_allocated()
        stats["torch_reserved_bytes"] = torch.cuda.memory_reserved()
        stats["unified"] = abs(total - vm.total) / vm.total < _UNIFIED_TOLERANCE
    except RuntimeError:
        # A CUDA context that failed to initialise raises rather than
        # returning falsey. The readout degrades to RAM; the trace routes
        # will report the real problem.
        pass

    return stats
