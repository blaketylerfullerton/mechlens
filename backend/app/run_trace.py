"""Phase 1 entrypoint: run a prompt, save a trace.

    python run_trace.py --prompt "The Golden Gate Bridge is in the city of"
    python run_trace.py -p "2 + 2 =" -n 8 --out-dir ../traces

Run it from backend/app (the modules import each other flat, as in phase 0).
Under ipython the model stays loaded between runs:

    %load_ext autoreload; %autoreload 2
    %run run_trace.py -p "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capture import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, generate_trace
from model_cache import MODEL_NAME, get_model
from store import DEFAULT_TRACE_DIR, save_trace

PROMPT = "The Golden Gate Bridge is located in the city of"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-p", "--prompt", default=PROMPT)
    p.add_argument("-n", "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_TRACE_DIR)
    p.add_argument("--name", help="filename stem; defaults to the generated trace id")
    p.add_argument("--no-stop-at-eos", action="store_true")
    return p.parse_args()


def summarize(trace, residuals, json_path: Path) -> None:
    print(f"\nprompt:     {trace.prompt!r}")
    print(f"completion: {trace.completion!r}")
    print(
        f"{trace.n_prompt_tokens} prompt + {trace.n_generated_tokens} generated tokens "
        f"in {trace.elapsed_s:.1f}s ({trace.stop_reason})"
    )

    # Residual norms grow steeply with depth in Gemma 2 — a flat or exploding
    # profile here is the first sign something is wrong with the capture.
    norms = [trace.steps[-1].layers[l].resid_norm for l in range(trace.n_layers)]
    print(f"\nfinal-token resid norms: L0={norms[0]:.1f} ... L{trace.n_layers - 1}={norms[-1]:.1f}")

    top = trace.steps[-1].logits.top_k[:5]
    print("next-token top 5: " + ", ".join(f"{t.text!r} {t.prob:.1%}" for t in top))

    npy_path = json_path.with_name(trace.residuals.path)
    print(
        f"\nwrote {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)"
        f"\n      {npy_path}  ({npy_path.stat().st_size / 1e6:.1f} MB, "
        f"{tuple(trace.residuals.shape)} {trace.residuals.dtype})"
    )


def main() -> None:
    args = parse_args()
    model = get_model(args.model)

    result = generate_trace(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        stop_at_eos=not args.no_stop_at_eos,
    )
    json_path = save_trace(result, args.out_dir, name=args.name)
    summarize(result.trace, result.residuals, json_path)


if __name__ == "__main__":
    main()
