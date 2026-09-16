"""How much time and memory does an attribution actually cost on this box?

CIR-02 has to pick defaults and decide whether attribution can share a process
with the inference model. Both answers come from here, not from a guess.

Also probes the two operational limits the spec asks about explicitly:
progress reporting and cancellation. Neither has a first-class API in the pinned
version, so what gets recorded is the honest granularity, not a wish.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

import common

PROMPT = (
    "The Golden Gate Bridge is located in the city of San Francisco, California, "
    "which is a major port on the west coast of the United States of America and "
    "home to a large technology industry that employs many thousands of engineers, "
    "researchers and designers who moved there from other parts of the country over "
    "the last several decades in search of better weather and more interesting work"
)


def prefix_of(model, n_tokens: int) -> list[int]:
    ids = model.tokenizer(PROMPT).input_ids
    if n_tokens > len(ids):
        raise SystemExit(f"prompt only tokenizes to {len(ids)}; asked for {n_tokens}")
    return ids[:n_tokens]


def one_cell(model, prefix: list[int], max_feature_nodes: int, batch_size: int) -> dict:
    from circuit_tracer import attribute

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Target the model's own top prediction, so every cell measures the same
    # shape of work regardless of prefix length.
    with torch.no_grad():
        logits = model(torch.tensor([prefix], device=common.DEVICE))
    target = int(logits[0, -1].argmax())

    t0 = time.time()
    graph = attribute(prefix, model, attribution_targets=torch.tensor([target]),
                      max_feature_nodes=max_feature_nodes, batch_size=batch_size)
    elapsed = time.time() - t0

    tmp = common.OUT / "_bench_tmp.pt"
    graph.to_pt(str(tmp))
    size = tmp.stat().st_size
    tmp.unlink()

    return {
        "n_tokens": len(prefix),
        "max_feature_nodes": max_feature_nodes,
        "batch_size": batch_size,
        "attribution_s": elapsed,
        "peak_memory_mb": common.peak_memory_mb(),
        "n_active_features": int(graph.active_features.shape[0]),
        "n_selected_features": int(graph.selected_features.shape[0]),
        "adjacency_shape": list(graph.adjacency_matrix.shape),
        "graph_pt_bytes": size,
    }


def probe_operational(model, prefix: list[int]) -> dict:
    """What the pinned version can and cannot tell a job runner."""
    from circuit_tracer import attribute
    import inspect

    sig = inspect.signature(attribute).parameters
    return {
        "progress": {
            "callback_parameter": None,
            "available_knobs": [k for k in ("verbose", "update_interval") if k in sig],
            "mechanism": (
                "attribute() writes phase lines to the 'attribution' stdlib logger and "
                "renders a tqdm bar over feature nodes. There is no progress callback "
                "parameter. A job runner can attach a logging.Handler to that logger to "
                "get phase transitions, and would have to redirect tqdm to get "
                "within-phase counts."
            ),
            "phases": ["Phase 0: precompute", "Phase 1: forward pass",
                       "Phase 2: input vectors", "Phase 3: logit attributions",
                       "Phase 4: feature attributions"],
        },
        "cancellation": {
            "cancel_parameter": None,
            "mechanism": (
                "No cancellation hook of any kind. attribute() runs to completion once "
                "entered. The only boundaries a caller controls are before the call and "
                "after it returns. Cooperative cancellation would require either running "
                "attribution in a separate process that can be killed, or patching the "
                "library's batch loop."
            ),
            "honest_granularity": "whole-call; not interruptible in-process",
        },
        "teardown": {
            "note": (
                "TransformerLensReplacementModel.__del__ raises a TypeError during "
                "interpreter shutdown (reset_hooks -> isinstance on a torch Parameter "
                "shim). It is swallowed as 'Exception ignored in __del__', but it means "
                "hook cleanup must be done explicitly by the caller rather than left to "
                "garbage collection."
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32, 48])
    p.add_argument("--feature-nodes", type=int, nargs="+", default=[2000, 5000, 7500])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, load_s = common.load_replacement_model()
    load_peak = common.peak_memory_mb()
    print(f"model + transcoders loaded in {load_s:.1f}s, peak {load_peak:.0f}MB\n")

    cells = []
    for n in args.tokens:
        prefix = prefix_of(model, n)
        for k in args.feature_nodes:
            cell = one_cell(model, prefix, k, args.batch_size)
            cells.append(cell)
            print(f"  {n:>3} tok  {k:>5} nodes  "
                  f"{cell['attribution_s']:>6.2f}s  "
                  f"peak {cell['peak_memory_mb']:>8.0f}MB  "
                  f"selected {cell['n_selected_features']:>5}  "
                  f"pt {cell['graph_pt_bytes'] / 1e6:>7.1f}MB")

    record = common.provenance(
        load={"model_and_transcoders_s": load_s, "peak_memory_after_load_mb": load_peak},
        cells=cells,
        operational=probe_operational(model, prefix_of(model, 8)),
    )
    out = args.out or common.OUT / "bench.json"
    common.save_json(out, record)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
