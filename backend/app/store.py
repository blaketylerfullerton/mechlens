"""Reading and writing traces on disk.

A trace is two files that travel together:

    traces/<trace_id>.json            the Trace document (schema.py)
    traces/<trace_id>.residuals.npy   [n_tokens, n_layers, d_model] float32

The JSON is the contract; the .npy is the bulk. Splitting them keeps the JSON
small enough to read, diff, and ship to a browser, and lets the SAE/attribution
passes memory-map the tensor instead of parsing 12MB of numbers.

The sidecar path is stored *relative* to the JSON, so a trace directory can be
moved or copied around without rewriting anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from capture import CaptureResult
from schema import ResidualRef, Trace

DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[1] / "traces"

SIDECAR_SUFFIX = ".residuals.npy"


def save_trace(
    result: CaptureResult,
    out_dir: Path | str = DEFAULT_TRACE_DIR,
    name: str | None = None,
) -> Path:
    """Write both halves of `result`; returns the path of the JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = name or result.trace.trace_id
    json_path = out_dir / f"{stem}.json"
    npy_path = out_dir / f"{stem}{SIDECAR_SUFFIX}"

    np.save(npy_path, result.residuals)

    # The ref is written here rather than by the capture: only now is there a
    # filename to point at, and the description should match the bytes that
    # actually landed on disk.
    trace = result.trace
    trace.residuals = ResidualRef(
        path=npy_path.name,
        hook=result.hook,
        shape=tuple(result.residuals.shape),
        dtype=str(result.residuals.dtype),
    )

    json_path.write_text(trace.model_dump_json(indent=2))
    return json_path


def load_trace(json_path: Path | str) -> Trace:
    return Trace.model_validate_json(Path(json_path).read_text())


def load_residuals(
    trace: Trace,
    json_path: Path | str,
    mmap: bool = False,
) -> np.ndarray:
    """Load the residual tensor `trace` points at.

    `json_path` is where the trace was read from — the sidecar is resolved
    relative to it. Pass `mmap=True` to page the tensor in lazily, which is the
    right default for a pass that only touches a few layers.
    """
    if trace.residuals is None:
        raise ValueError(f"trace {trace.trace_id} has no residuals attached")

    npy_path = Path(json_path).resolve().parent / trace.residuals.path
    array = np.load(npy_path, mmap_mode="r" if mmap else None)

    expected = tuple(trace.residuals.shape)
    if array.shape != expected:
        raise ValueError(
            f"{npy_path.name}: expected {expected} from the trace, found {array.shape}"
        )
    return array


def load(json_path: Path | str, mmap: bool = False) -> tuple[Trace, np.ndarray]:
    """Convenience: both halves in one call."""
    trace = load_trace(json_path)
    return trace, load_residuals(trace, json_path, mmap=mmap)
