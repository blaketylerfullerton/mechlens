"""Does an intervention derived from the graph move the original model?

This mirrors the row structure of backend/app/experiments.py::measure_feature on
purpose, so the two are directly comparable:

    baseline      -> no intervention
    zero_control  -> intervene, but set the feature to the value it already has.
                     A no-op. It must reproduce baseline, or nothing downstream
                     can be interpreted.
    ablation      -> set the feature to 0.0

All rows run on identical token ids and are scored on the target token at the
last prefix position, in probability and log-probability.

The claim being tested is narrow and the spec is emphatic about it: ablating one
feature tests that one operation. It does not validate every edge on a path and
it does not establish mediation.

One subtlety worth knowing before reading the numbers: the activation cached at
the intervened site does *not* change. The intervention adds its delta to the
MLP output, downstream of the transcoder encoder that produced that activation,
and the encoder's input at that layer is unchanged. So the activation readback at
the site is the original value by construction, and the evidence that anything
happened is (a) the output scores and (b) the feature activations at later
layers. Reporting the site's readback as "after" would imply a write that never
occurred.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import common

# experiments.py's threshold for the residual-SAE path, which runs the same
# no-op-control idea in fp32. bfloat16 will almost certainly need looser; the
# point of recording it is to find out by how much, not to assume.
ZERO_CONTROL_TOL = 1e-5


def score(logits: torch.Tensor, target_token_id: int) -> dict:
    last = logits[0, -1].float() if logits.ndim == 3 else logits[-1].float()
    logprobs = torch.log_softmax(last, dim=-1)
    return {
        "target_logit": last[target_token_id].item(),
        "target_log_probability": logprobs[target_token_id].item(),
        "target_probability": logprobs[target_token_id].exp().item(),
        "argmax_token_id": int(last.argmax()),
    }


def top_contributors(graph, k: int) -> list[tuple[int, int, int]]:
    """The k selected feature nodes with the most influence on the target logit.

    Uses the library's own influence measure so the features we test are the ones
    a user would actually see at the top of the graph.
    """
    from circuit_tracer.graph import compute_node_influence

    n_logits = len(graph.logit_targets)
    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0])
    logit_weights[-n_logits:] = graph.logit_probabilities.cpu().float()
    influence = compute_node_influence(
        graph.adjacency_matrix.cpu().float(), logit_weights
    )

    n_features = int(graph.selected_features.shape[0])
    order = influence[:n_features].argsort(descending=True)[:k]
    out = []
    for i in order.tolist():
        layer, pos, feat = graph.active_features[graph.selected_features[i]].tolist()
        out.append((int(layer), int(pos), int(feat)))
    return out


def measure(model, prefix: list[int], target_token_id: int,
            layer: int, pos: int, feature_idx: int,
            *, constrained_layers=None, freeze_attention=True) -> dict:
    tokens = torch.tensor([prefix], device=common.DEVICE)

    with torch.no_grad():
        base_logits, acts = model.get_activations(tokens)
    observed = float(acts[layer, pos, feature_idx])
    baseline = score(base_logits, target_token_id)

    rows = [{"mode": "baseline", "requested_value": None,
             "activation_at_site": observed,
             "downstream_features_changed": 0,
             "max_downstream_activation_delta": 0.0,
             **baseline}]

    for mode, value in (("zero_control", observed), ("ablation", 0.0)):
        with torch.no_grad():
            logits, new_acts = model.feature_intervention(
                tokens, [(layer, pos, feature_idx, value)],
                constrained_layers=constrained_layers,
                freeze_attention=freeze_attention,
            )
        # Everything strictly after the intervened layer. This is where an
        # intervention becomes visible in activation space.
        delta = (new_acts[layer + 1:].float() - acts[layer + 1:].float()).abs()
        row = {"mode": mode, "requested_value": value,
               "activation_at_site": float(new_acts[layer, pos, feature_idx]),
               "downstream_features_changed": int((delta > 1e-3).sum()),
               "max_downstream_activation_delta": float(delta.max()) if delta.numel() else 0.0,
               **score(logits, target_token_id)}
        rows.append(row)

    for row in rows[1:]:
        row["delta_probability"] = row["target_probability"] - baseline["target_probability"]
        row["delta_log_probability"] = (
            row["target_log_probability"] - baseline["target_log_probability"])

    zero = rows[1]
    control_error = max(
        abs(zero[k] - baseline[k])
        for k in ("target_logit", "target_log_probability", "target_probability")
    )

    return {
        "feature": {"layer": layer, "position": pos, "feature_idx": feature_idx},
        "observed_activation": observed,
        "site_readback_note": (
            "activation_at_site is unchanged by design: the delta is added after the "
            "transcoder encodes this layer, so the encoder's own output is untouched. "
            "downstream_features_changed is the activation-space evidence."
        ),
        "settings": {"constrained_layers": str(constrained_layers),
                     "freeze_attention": freeze_attention},
        "zero_control_max_error": control_error,
        "zero_control_tolerance": ZERO_CONTROL_TOL,
        "zero_control_passed": control_error <= ZERO_CONTROL_TOL,
        "measurements": rows,
        "interpretation": (
            "Ablating this feature tests this one operation on the original model's "
            "forward pass, via the transcoder decoder direction. It does not validate "
            "other edges on any displayed path and does not establish mediation."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True, help=".pt from trace_graph.py")
    p.add_argument("--top", type=int, default=3, help="how many features to test")
    p.add_argument("--constrained", action="store_true",
                   help="also run with all layers constrained (direct effect only)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    from circuit_tracer import Graph

    rec = json.loads(args.export.read_text())
    prefix, target = rec["prefix_token_ids"], rec["target_token_id"]
    graph = Graph.from_pt(str(args.graph))

    model, load_s = common.load_replacement_model()
    print(f"model + transcoders loaded in {load_s:.1f}s")

    features = top_contributors(graph, args.top)
    print(f"testing {len(features)} top-influence features: {features}\n")

    reports = []
    for layer, pos, feat in features:
        reports.append(measure(model, prefix, target, layer, pos, feat))
        if args.constrained:
            reports.append(measure(model, prefix, target, layer, pos, feat,
                                   constrained_layers=range(model.cfg.n_layers)))

    for r in reports:
        f = r["feature"]
        print(f"L{f['layer']} pos{f['position']} feat{f['feature_idx']}  "
              f"constrained={r['settings']['constrained_layers']}")
        print(f"  zero-control max error {r['zero_control_max_error']:.3e} "
              f"({'ok' if r['zero_control_passed'] else 'ABOVE TOLERANCE'})")
        for row in r["measurements"]:
            d = row.get("delta_probability")
            print(f"    {row['mode']:<13} p={row['target_probability']:.6f}"
                  + (f"  delta={d:+.6f}" if d is not None else "           ")
                  + f"  downstream changed={row['downstream_features_changed']:>4}"
                    f" (max |d act| {row['max_downstream_activation_delta']:.2f})")
        print()

    failed = [r for r in reports if not r["zero_control_passed"]]
    record = common.provenance(
        export=str(args.export), graph=str(args.graph),
        source_trace=rec.get("trace_id"),
        target_token_id=target, target_text=rec.get("target_text"),
        reports=reports,
        zero_controls_failed=len(failed),
        execution_model=(
            "Interventions add (value - current_activation) * transcoder_decoder_vector "
            "into the original model's forward pass. This is an original-model test, "
            "not a replacement-model-only test."
        ),
        peak_memory_mb=common.peak_memory_mb(),
    )
    out = args.out or common.OUT / f"intervene-{args.export.stem}.json"
    common.save_json(out, record)
    print(f"wrote {out}")

    if failed:
        print(f"\n{len(failed)} zero control(s) above {ZERO_CONTROL_TOL}: "
              "the measured error is recorded; interpretation of those rows is blocked "
              "until the tolerance is justified.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
