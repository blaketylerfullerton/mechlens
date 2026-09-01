"""mechlens CLI. Run from the `backend/` directory.

    python -m app.cli trace -p "The Golden Gate Bridge is in the city of" -n 20
    python -m app.cli trace -p "2 + 2 =" -n 8 --sae        # capture and encode
    python -m app.cli enrich traces/<id>.json --sae        # encode a saved trace
    python -m app.cli enrich traces/<id>.json --labels     # attach Neuronpedia labels
    python -m app.cli show traces/<id>.json --layer 20

`enrich` is the loop you actually iterate in: it reads the residual sidecar and
never loads gemma, so re-running the SAE pass on a saved trace costs seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .capture import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, generate_trace
from .model_cache import MODEL_NAME, get_model
from .passes import apply
from .labels import DEFAULT_DB_PATH, feature_url
from .passes.labels import LabelsPass
from .passes.sae import DEFAULT_TOP_K as SAE_TOP_K
from .passes.sae import SAEPass
from .sae_cache import DEFAULT_WIDTH
from .schema import Trace
from .store import DEFAULT_TRACE_DIR, load, save_trace, update_trace

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
    layer = layer if layer is not None else trace.n_layers // 2
    pos = position if position is not None else len(trace.steps) - 1
    step = trace.steps[pos]
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


def print_label_summary(trace: Trace) -> None:
    record = trace.pass_record("labels")
    if record is None:
        return
    print(
        f"\nlabels {record.stats['features_labelled']:.0f}/"
        f"{record.stats['features_wanted']:.0f} distinct features "
        f"({record.stats['coverage']:.1%}) | {record.params['explainers']}"
    )


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
        update_trace(result.trace, json_path)
        print_sae_summary(result.trace)

    if args.labels:
        apply(_labels_pass(args), result.trace, result.residuals)
        update_trace(result.trace, json_path)
        print_label_summary(result.trace)


def cmd_enrich(args: argparse.Namespace) -> None:
    trace, residuals = load(args.trace, mmap=True)
    print(
        f"{args.trace}: {len(trace.steps)} tokens x {trace.n_layers} layers "
        f"({trace.model}, schema {trace.schema_version})"
    )

    if not (args.sae or args.labels):
        raise SystemExit("nothing to do — pass --sae and/or --labels")

    if args.sae:
        layers = parse_layers(args.layers, trace.n_layers)
        apply(
            SAEPass(width=args.width, top_k=args.sae_top_k, layers=layers, device=args.device),
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
    if trace.pass_record("sae"):
        print_sae_summary(trace, args.layer, args.token)
        print_label_summary(trace)
    else:
        norms = np.array([[l.resid_norm for l in s.layers] for s in trace.steps])
        print(f"  resid norms: L0 {norms[:, 0].mean():.0f} -> L{trace.n_layers - 1} {norms[:, -1].mean():.0f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.cli", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add_sae_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--width", default=DEFAULT_WIDTH, help="SAE width: 16k, 65k, 262k")
        sp.add_argument("--sae-top-k", type=int, default=SAE_TOP_K, help="features kept per layer")
        sp.add_argument("--layers", help="subset to encode, e.g. '0-5,20' (default: all)")
        sp.add_argument("--device", help="cuda / cpu (default: cuda when available)")

    def add_label_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--labels", action="store_true", help="attach Neuronpedia labels")
        sp.add_argument("--labels-db", type=Path, default=DEFAULT_DB_PATH)
        sp.add_argument(
            "--fetch-missing",
            action="store_true",
            help="ask neuronpedia.org about features the local DB has never seen",
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
    add_label_flags(t)
    t.set_defaults(func=cmd_trace)

    e = sub.add_parser("enrich", help="run passes over a saved trace (no model load)")
    e.add_argument("trace", type=Path)
    e.add_argument("--sae", action="store_true", help="run the SAE pass")
    add_sae_flags(e)
    add_label_flags(e)
    e.set_defaults(func=cmd_enrich)

    s = sub.add_parser("show", help="summarise a saved trace")
    s.add_argument("trace", type=Path)
    s.add_argument("--layer", type=int, help="layer to show features for (default: middle)")
    s.add_argument("--token", type=int, help="token position to show (default: the last one)")
    s.set_defaults(func=cmd_show)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
