"""mechlens CLI. Run from the `backend/` directory.

    python -m app.cli trace -p "The Golden Gate Bridge is in the city of" -n 20
    python -m app.cli trace -p "2 + 2 =" -n 8 --sae        # capture and encode
    python -m app.cli enrich traces/<id>.json --sae        # encode a saved trace
    python -m app.cli enrich traces/<id>.json --labels     # attach Neuronpedia labels
    python -m app.cli enrich traces/<id>.json --lens       # decode every layer
    python -m app.cli enrich traces/<id>.json --attribution # decompose every layer
    python -m app.cli show traces/<id>.json --layer 20
    python -m app.cli show traces/<id>.json --lens --token 12
    python -m app.cli show traces/<id>.json --attribution --token 12
    python -m app.cli view traces/<id>.json                 # the same thing, in a browser

`enrich` is the loop you actually iterate in: it reads the residual sidecar, so
re-running the SAE pass on a saved trace costs seconds. `--sae` and `--labels`
never touch gemma at all; `--lens` and `--attribution` are the exceptions —
`--lens` because W_U is 2304 x 256_000 and has to come from the model,
`--attribution` because it needs a fresh forward pass for the attention
pattern and per-head values, neither of which the residual sidecar holds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .capture import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, generate_trace
from .model_cache import MODEL_NAME, get_model
from .passes import apply
from .labels import DEFAULT_DB_PATH, feature_url
from .passes.attribution import DEFAULT_TOP_K as ATTRIBUTION_TOP_K
from .passes.attribution import AttributionPass
from .passes.labels import LabelsPass
from .passes.lens import DEFAULT_TOP_K as LENS_TOP_K
from .passes.lens import LogitLensPass
from .passes.sae import DEFAULT_TOP_K as SAE_TOP_K
from .passes.sae import SAEPass
from .sae_cache import DEFAULT_WIDTH
from .schema import Trace
from .store import DEFAULT_TRACE_DIR, load, save_trace, update_trace
from .viewer import DEFAULT_PORT

PROMPT = "The Golden Gate Bridge is located in the city of"


def parse_layers(spec: str | None, n_layers: int) -> list[int] | None:
    """"0-5,20,25" -> [0,1,2,3,4,5,20,25]; None means every layer."""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise SystemExit(f"layers out of range for this model: {bad}")
    return sorted(set(out))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_capture(trace: Trace, json_path: Path) -> None:
    print(f"\nprompt:     {trace.prompt!r}")
    print(f"completion: {trace.completion!r}")
    print(
        f"{trace.n_prompt_tokens} prompt + {trace.n_generated_tokens} generated tokens "
        f"in {trace.elapsed_s:.1f}s ({trace.stop_reason})"
    )
    top = trace.steps[-1].logits.top_k[:5]
    print("next-token top 5: " + ", ".join(f"{t.text!r} {t.prob:.1%}" for t in top))
    print(f"\nwrote {json_path}")


def print_sae_summary(trace: Trace, layer: int | None = None, position: int | None = None) -> None:
    """The phase-2 sanity check, printed rather than guessed at."""
    record = trace.pass_record("sae")
    if record is None:
        print("no SAE pass has been run on this trace")
        return

    l0 = record.stats["l0_mean"]
    ev = record.stats["explained_variance_mean"]
    print(
        f"\nSAE {record.params['release']} {record.params['width']} "
        f"| mean L0 {l0:.1f} | mean explained variance {ev:.3f} "
        f"| {record.elapsed_s:.1f}s"
    )
    healthy = 30 <= l0 <= 150 and ev >= 0.7
    print("  numbers look sane" if healthy else "  ⚠ outside the expected range — see phase2.md")

    # A "handful of strong activations, long tail of weak ones" is the shape to
    # look for. Default to the last token — the one the model just produced.
    # (Picking by resid_norm instead always lands on position 1, since norms
    # grow with depth but shrink with position in Gemma 2.)
    pos = position if position is not None else len(trace.steps) - 1
    step = trace.steps[pos]

    # Default to the middle of the layers that were actually *encoded*, not the
    # middle of the model: `--layers 12` leaves every other layer featureless,
    # and defaulting blindly to n_layers//2 reports an empty l0=None state as
    # though the pass had done nothing.
    if layer is None:
        encoded = [s.layer for s in step.layers if s.features]
        layer = encoded[len(encoded) // 2] if encoded else trace.n_layers // 2
    state = step.layers[layer]
    print(f"\nlayer {layer}, token {pos} {step.token.text!r}  (l0={state.l0})")

    labelled = bool(trace.labels)
    for f in state.features[:8]:
        bar = "█" * max(1, round(f.activation / state.features[0].activation * 12))
        line = f"  #{f.index:<6} {f.activation:7.2f}  {bar:<12}"
        if labelled:
            label = trace.label(layer, f.index)
            line += f"  {label.text[:58] if label else '—'}"
        print(line)

    if labelled:
        # One worked link, so the trace is verifiable by hand rather than
        # taken on faith — click through and the activating examples should
        # look like the label says they will.
        top = state.features[0]
        print(f"\n  {feature_url(layer, top.index)}")


def print_lens_summary(trace: Trace, position: int | None = None) -> None:
    """The phase-4 payoff: one token's answer forming as you go down the stack.

    Printed as a full column rather than a summary because that shape is the
    whole point — a mean over layers hides the crystallisation it is meant to
    show.
    """
    record = trace.pass_record("lens")
    if record is None:
        print("no logit lens pass has been run on this trace")
        return

    agreement = record.stats.get("final_layer_agreement")
    print(
        f"\nlogit lens | top-1 agreement with the final answer "
        f"{record.stats['agreement_first_layer']:.0%} -> "
        f"{record.stats['agreement_last_layer']:.0%} "
        f"| crossover at layer {record.stats['crossover_layer']:.0f} "
        f"| {record.elapsed_s:.1f}s"
    )
    last = trace.n_layers - 1
    if agreement is None:
        print(f"  layer {last} was not decoded — no correctness check on this trace")
    elif agreement == 1.0:
        # The claim that makes the rest of the column trustworthy: the last
        # layer is not an approximation of the model's output, it is the
        # model's output, and it came back identical.
        ties = int(record.stats["final_layer_argmax_ties"])
        # Called out rather than hidden: a tie means two logits landed on the
        # same bf16 value and topk picked between them arbitrarily. The
        # distribution matched; only the coin flip differed.
        note = f" ({ties} resolved by an argmax tie)" if ties else ""
        print(
            f"  layer {last} reproduces the model's own output on all "
            f"{len(trace.steps)} positions{note} "
            f"— max prob delta {record.stats['final_layer_max_prob_delta']:.1e}, "
            f"max entropy delta {record.stats['final_layer_max_entropy_delta']:.1e}"
        )
    else:
        print(
            f"  ⚠ layer {last} reproduces the model on only {agreement:.1%} of "
            f"positions — softcap or norm is wrong, see phase4.md"
        )

    # Default to the last position: the one the model was actually answering.
    pos = position if position is not None else len(trace.steps) - 1
    step = trace.steps[pos]
    answer = step.logits.top_k[0]
    print(
        f"\n  token {pos} {step.token.text!r} — the model answers "
        f"{answer.text!r} {answer.prob:.1%}"
    )

    print("   ◀ = the model's final answer   · = an echo of the current token\n")

    for state in step.layers:
        lens = state.logit_lens
        if lens is None:
            continue
        top = lens.top_k[0]
        # Two markers, because the naive reading of this column is wrong.
        # "◀" is the crystallisation: the depth where it starts and never stops
        # is where the answer was decided. "·" is the early-layer artifact —
        # residuals near the embedding decode back to the token already sitting
        # here, so a confident-looking L4 is often just reading itself.
        if top.token_id == answer.token_id:
            mark = "◀"
        elif top.token_id == step.token.token_id:
            mark = "·"
        else:
            mark = " "
        print(
            f"   L{state.layer:<3} {top.text[:14]!r:<16} {top.prob:6.2%}  "
            f"H {lens.entropy:5.2f}  {mark} {'█' * round(top.prob * 20)}"
        )


def print_label_summary(trace: Trace) -> None:
    record = trace.pass_record("labels")
    if record is None:
        return
    print(
        f"\nlabels {record.stats['features_labelled']:.0f}/"
        f"{record.stats['features_wanted']:.0f} distinct features "
        f"({record.stats['coverage']:.1%}) | {record.params['explainers']}"
    )


def print_attribution_summary(trace: Trace, position: int | None = None) -> None:
    """The phase-5 payoff: which upstream edges produced one token's residual
    at every layer, printed as a column the same way the lens is."""
    record = trace.pass_record("attribution")
    if record is None:
        print("no attribution pass has been run on this trace")
        return

    gap = record.stats["reconstruction_max_rel_gap"]
    coverage = record.stats["attn_topk_coverage"]
    print(
        f"\nattribution | reconstruction gap {gap:.1e} | "
        f"attn top-{record.params['top_k']} coverage {coverage:.1%} | "
        f"{record.elapsed_s:.1f}s"
    )
    # This is a max over every (layer, position) — hundreds of samples on a
    # 26-layer gemma-2-2b trace — so it is the tail of bf16 rounding, not its
    # typical size: measured on golden-gate.json, the per-layer mean sits at
    # ~0.004-0.005 with no growth across depth, while the max alone ranges up
    # to ~0.013. A real bug (missing term, wrong hook) shows up as the mean
    # itself drifting, not as an occasional spike in the max.
    if gap >= 5e-2:
        print("  ⚠ reconstruction gap is far from zero — a hook or a sign is wrong, see design.md")

    pos = position if position is not None else len(trace.steps) - 1
    step = trace.steps[pos]
    print(f"\n  token {pos} {step.token.text!r}\n")

    for state in step.layers:
        by_kind: dict[str, list] = {}
        for edge in state.edges:
            by_kind.setdefault(edge.kind, []).append(edge)
        parts = []
        if "resid" in by_kind:
            parts.append(f"resid {by_kind['resid'][0].weight:5.1f}")
        if "mlp" in by_kind:
            parts.append(f"mlp {by_kind['mlp'][0].weight:5.1f}")
        attn = sorted(by_kind.get("attn", []), key=lambda e: e.weight, reverse=True)
        if attn:
            top = ", ".join(f"pos{e.source.position}={e.weight:.1f}" for e in attn[:3])
            parts.append(f"attn [{top}]")
        print(f"   L{state.layer:<3} " + "  ".join(parts))


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_trace(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    result = generate_trace(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        stop_at_eos=not args.no_stop_at_eos,
    )
    json_path = save_trace(result, args.out_dir, name=args.name)
    print_capture(result.trace, json_path)

    if args.sae:
        layers = parse_layers(args.layers, result.trace.n_layers)
        apply(
            SAEPass(width=args.width, top_k=args.sae_top_k, layers=layers, device=args.device),
            result.trace,
            result.residuals,
        )
    if args.lens:
        # The model is already resident here, so the lens costs no extra load.
        layers = parse_layers(args.layers, result.trace.n_layers)
        apply(
            LogitLensPass(top_k=args.lens_top_k, layers=layers, model=model, verbose=False),
            result.trace,
            result.residuals,
        )
    if args.labels:
        apply(_labels_pass(args), result.trace, result.residuals)

    if args.attribution:
        # Same reasoning as --lens: the model is already resident, so this
        # costs one extra forward pass, not an extra load.
        layers = parse_layers(args.layers, result.trace.n_layers)
        apply(
            AttributionPass(
                top_k=args.attribution_top_k, layers=layers, model=model, verbose=False
            ),
            result.trace,
            result.residuals,
        )

    if args.sae or args.labels or args.lens or args.attribution:
        update_trace(result.trace, json_path)
        # Same as `enrich`: report once every pass has run, so the feature list
        # is printed with its labels rather than as bare integers.
        if args.sae:
            print_sae_summary(result.trace)
        print_label_summary(result.trace)
        if args.lens:
            print_lens_summary(result.trace)
        if args.attribution:
            print_attribution_summary(result.trace)


def cmd_enrich(args: argparse.Namespace) -> None:
    trace, residuals = load(args.trace, mmap=True)
    print(
        f"{args.trace}: {len(trace.steps)} tokens x {trace.n_layers} layers "
        f"({trace.model}, schema {trace.schema_version})"
    )

    if not (args.sae or args.labels or args.lens or args.attribution):
        raise SystemExit("nothing to do — pass --sae, --labels, --lens and/or --attribution")

    if args.sae:
        layers = parse_layers(args.layers, trace.n_layers)
        apply(
            SAEPass(width=args.width, top_k=args.sae_top_k, layers=layers, device=args.device),
            trace,
            residuals,
        )

    if args.lens:
        layers = parse_layers(args.layers, trace.n_layers)
        # Unlike the other two, this one loads gemma — W_U has to come from
        # somewhere. Left to the pass so `enrich --sae` stays model-free.
        apply(
            LogitLensPass(top_k=args.lens_top_k, layers=layers, verbose=False),
            trace,
            residuals,
        )

    if args.attribution:
        layers = parse_layers(args.layers, trace.n_layers)
        # Also loads gemma — attention pattern and per-head values are not on
        # the residual sidecar, so this pays its own forward pass.
        apply(
            AttributionPass(top_k=args.attribution_top_k, layers=layers, verbose=False),
            trace,
            residuals,
        )

    if args.labels:
        apply(_labels_pass(args), trace, residuals)

    # Reported only once every pass has run: the feature list is worth far more
    # with the labels beside it, and printing it mid-way shows bare integers.
    if args.sae:
        print_sae_summary(trace)
    print_label_summary(trace)
    if args.lens:
        print_lens_summary(trace)
    if args.attribution:
        print_attribution_summary(trace)

    update_trace(trace, args.trace)
    print(f"\nupdated {args.trace}")


def _labels_pass(args: argparse.Namespace) -> LabelsPass:
    return LabelsPass(
        width=args.width,
        db_path=args.labels_db,
        fetch_missing=args.fetch_missing,
    )


def cmd_show(args: argparse.Namespace) -> None:
    trace, residuals = load(args.trace, mmap=True)
    print(f"{trace.trace_id}  {trace.model}  schema {trace.schema_version}")
    print(f"  {trace.prompt!r} -> {trace.completion!r}")
    print(f"  residuals {tuple(trace.residuals.shape)} {trace.residuals.dtype} ({trace.residuals.hook})")
    print(f"  passes: {[p.name for p in trace.passes] or 'none'}")

    if args.lens or (trace.pass_record("lens") and not trace.pass_record("sae")):
        print_lens_summary(trace, args.token)
        return

    if args.attribution:
        print_attribution_summary(trace, args.token)
        return

    if trace.pass_record("sae"):
        print_sae_summary(trace, args.layer, args.token)
        print_label_summary(trace)
    else:
        norms = np.array([[l.resid_norm for l in s.layers] for s in trace.steps])
        print(f"  resid norms: L0 {norms[:, 0].mean():.0f} -> L{trace.n_layers - 1} {norms[:, -1].mean():.0f}")


def cmd_view(args: argparse.Namespace) -> None:
    # Imported here rather than at module scope: `trace` and `enrich` have no
    # business pulling in an HTTP server.
    from .viewer import serve

    serve(
        port=args.port,
        trace=str(args.trace) if args.trace else None,
        trace_dir=args.out_dir,
        open_browser=not args.no_browser,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.cli", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add_sae_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--width", default=DEFAULT_WIDTH, help="SAE width: 16k, 65k, 262k")
        sp.add_argument("--sae-top-k", type=int, default=SAE_TOP_K, help="features kept per layer")
        sp.add_argument("--layers", help="subset to encode, e.g. '0-5,20' (default: all)")
        sp.add_argument("--device", help="cuda / cpu (default: cuda when available)")

    def add_lens_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--lens",
            action="store_true",
            help="decode every layer through ln_final + W_U (loads the model)",
        )
        sp.add_argument("--lens-top-k", type=int, default=LENS_TOP_K)

    def add_label_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--labels", action="store_true", help="attach Neuronpedia labels")
        sp.add_argument("--labels-db", type=Path, default=DEFAULT_DB_PATH)
        sp.add_argument(
            "--fetch-missing",
            action="store_true",
            help="ask neuronpedia.org about features the local DB has never seen",
        )

    def add_attribution_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--attribution",
            action="store_true",
            help="decompose every layer's residual into resid/attn/mlp edges (loads the model)",
        )
        sp.add_argument(
            "--attribution-top-k", type=int, default=ATTRIBUTION_TOP_K, help="attn edges kept per layer/position"
        )

    t = sub.add_parser("trace", help="generate and capture a new trace")
    t.add_argument("-p", "--prompt", default=PROMPT)
    t.add_argument("-n", "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    t.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="next-token candidates kept")
    t.add_argument("--model", default=MODEL_NAME)
    t.add_argument("--out-dir", type=Path, default=DEFAULT_TRACE_DIR)
    t.add_argument("--name", help="filename stem; defaults to the generated trace id")
    t.add_argument("--no-stop-at-eos", action="store_true")
    t.add_argument("--sae", action="store_true", help="also run the SAE pass")
    add_sae_flags(t)
    add_lens_flags(t)
    add_label_flags(t)
    add_attribution_flags(t)
    t.set_defaults(func=cmd_trace)

    e = sub.add_parser("enrich", help="run passes over a saved trace")
    e.add_argument("trace", type=Path)
    e.add_argument("--sae", action="store_true", help="run the SAE pass")
    add_sae_flags(e)
    add_lens_flags(e)
    add_label_flags(e)
    add_attribution_flags(e)
    e.set_defaults(func=cmd_enrich)

    s = sub.add_parser("show", help="summarise a saved trace")
    s.add_argument("trace", type=Path)
    s.add_argument("--layer", type=int, help="layer to show features for (default: middle)")
    s.add_argument("--token", type=int, help="token position to show (default: the last one)")
    s.add_argument("--lens", action="store_true", help="show the per-layer logit lens instead")
    s.add_argument("--attribution", action="store_true", help="show the per-layer attribution edges instead")
    s.set_defaults(func=cmd_show)

    v = sub.add_parser("view", help="open the trace viewer in a browser")
    v.add_argument("trace", type=Path, nargs="?", help="trace to open (default: the picker)")
    v.add_argument("--port", type=int, default=DEFAULT_PORT)
    v.add_argument("--out-dir", type=Path, default=DEFAULT_TRACE_DIR, help="where traces live")
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(func=cmd_view)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
