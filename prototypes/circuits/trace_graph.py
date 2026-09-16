"""Run output-targeted attribution for an exact prefix/target pair.

The whole point of this script over the bundled `circuit-tracer attribute` CLI is
the target: the CLI takes a prompt string and lets the library pick salient logits
for you. CIR-01 has to explain *one specific token that mechlens already generated*,
so the prefix goes in as token ids and the target goes in as an explicit tensor.
No decoded text is ever re-tokenized.

Writes three things per run, all under out/:
  <name>.pt                raw Graph, for later analysis in Python
  graph_files/<name>*      pruned JSON the bundled viewer reads
  <name>.provenance.json   versions, artifact revisions, settings, timings
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common


def run(model, prefix_token_ids: list[int], target_token_id: int | None,
        *, max_feature_nodes: int | None, batch_size: int, verbose: bool = True):
    from circuit_tracer import attribute

    targets = None
    if target_token_id is not None:
        # A 1-element tensor of vocabulary indices: "explain exactly this logit".
        targets = torch.tensor([target_token_id])

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    graph = attribute(
        prefix_token_ids,
        model,
        attribution_targets=targets,
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
        verbose=verbose,
    )
    return graph, time.time() - t0


def summarize(graph) -> dict:
    """The shapes CIR-02 has to store. Node order in the adjacency matrix is
    [active_features, error_nodes, embed_nodes, logit_nodes], with
    n_layers * n_pos error nodes and one embed node per position."""
    n_pos = graph.n_pos
    n_layers = graph.cfg.n_layers
    return {
        "n_pos": n_pos,
        "n_layers": n_layers,
        "n_active_features": int(graph.active_features.shape[0]),
        "n_selected_features": int(graph.selected_features.shape[0]),
        "n_error_nodes": n_layers * n_pos,
        "n_embed_nodes": n_pos,
        "n_logit_nodes": len(graph.logit_targets),
        "adjacency_shape": list(graph.adjacency_matrix.shape),
        "adjacency_nonzero": int((graph.adjacency_matrix != 0).sum()),
        "scan_name": graph.scan_name,
        "input_tokens": graph.input_tokens.tolist(),
        "logit_targets": [
            {"token": getattr(t, "token", None), "probability": float(p)}
            for t, p in zip(graph.logit_targets, graph.logit_probabilities)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--export", type=Path, help="JSON from export_trace.py")
    src.add_argument("--prompt", help="a plain prompt; the library picks the targets")
    p.add_argument("--name", help="basename for the artifacts (default: derived)")
    p.add_argument("--max-feature-nodes", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--node-threshold", type=float, default=0.8)
    p.add_argument("--edge-threshold", type=float, default=0.98)
    p.add_argument("--no-graph-files", action="store_true", help="skip the viewer JSON")
    args = p.parse_args()

    model, load_s = common.load_replacement_model()
    print(f"model + transcoders loaded in {load_s:.1f}s")

    if args.export:
        rec = json.loads(args.export.read_text())
        prefix, target = rec["prefix_token_ids"], rec["target_token_id"]
        name = args.name or args.export.stem
        source = {"kind": "mechlens-trace", **{k: rec[k] for k in
                  ("trace_id", "prompt", "target_text", "target_position")}}
    else:
        prefix = model.tokenizer(args.prompt).input_ids
        target = None
        name = args.name or "prompt"
        source = {"kind": "prompt", "prompt": args.prompt}

    print(f"prefix: {len(prefix)} tokens; target: {target}")
    graph, attr_s = run(model, prefix, target,
                        max_feature_nodes=args.max_feature_nodes,
                        batch_size=args.batch_size)
    print(f"attribution took {attr_s:.1f}s")

    pt_path = common.OUT / f"{name}.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_pt(str(pt_path))

    graph_files_s = None
    if not args.no_graph_files:
        from circuit_tracer.utils.create_graph_files import create_graph_files

        t0 = time.time()
        create_graph_files(graph, slug=name, output_path=str(common.OUT / "graph_files"),
                           node_threshold=args.node_threshold,
                           edge_threshold=args.edge_threshold)
        graph_files_s = time.time() - t0

    record = common.provenance(
        source=source,
        prefix_token_ids=prefix,
        target_token_id=target,
        settings={
            "max_feature_nodes": args.max_feature_nodes,
            "batch_size": args.batch_size,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
        },
        graph=summarize(graph),
        timings={"model_load_s": load_s, "attribution_s": attr_s,
                 "graph_files_s": graph_files_s},
        peak_memory_mb=common.peak_memory_mb(),
        artifacts={"graph_pt": str(pt_path), "graph_pt_bytes": pt_path.stat().st_size},
    )
    out = common.save_json(common.OUT / f"{name}.provenance.json", record)
    print(f"wrote {pt_path}\nwrote {out}")
    print(json.dumps(record["graph"], indent=2, default=str)[:900])


if __name__ == "__main__":
    main()
