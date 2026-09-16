"""Turn a saved mechlens trace into an exact prefix/target pair for attribution.

Deliberately imports nothing from `backend.app`: this script runs inside the
prototype venv, which pins transformers 4.57 against mechlens's 5.16, so pulling
in mechlens would drag the wrong library into the process. The Trace JSON is the
contract (see backend/app/schema.py) and plain `json` is enough to read it.

For a generated token at step j, the explanatory prefix is steps[0..j-1] and the
target is steps[j]. The target is never part of its own prefix -- that is the one
invariant this file exists to enforce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO / "backend" / "traces"


def load(trace_id: str) -> dict:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        raise SystemExit(f"no such trace: {path}")
    return json.loads(path.read_text())


def generated_steps(trace: dict) -> list[dict]:
    return [s for s in trace["steps"] if s["token"]["source"] == "generated"]


def export(trace: dict, step_index: int) -> dict:
    steps = trace["steps"]
    if not 0 <= step_index < len(steps):
        raise SystemExit(f"step {step_index} out of range (0..{len(steps) - 1})")

    target = steps[step_index]["token"]
    if target["source"] != "generated":
        raise SystemExit(
            f"step {step_index} is a {target['source']} token; only generated tokens "
            "have a prediction to explain"
        )

    prefix = [s["token"]["token_id"] for s in steps[:step_index]]
    if not prefix:
        raise SystemExit("empty prefix")
    if prefix[0] != steps[0]["token"]["token_id"]:
        raise SystemExit("prefix does not start at position 0")

    # The invariant. Positions are contiguous from 0, so the target's position is
    # exactly len(prefix) and it cannot also be an element of the prefix.
    assert len(prefix) == target["position"], (len(prefix), target["position"])

    return {
        "trace_id": trace["trace_id"],
        "model": trace["model"],
        # The Trace schema carries no model revision, so it is genuinely unknown
        # rather than merely omitted here.
        "model_revision": trace.get("model_revision"),
        "dtype": trace["dtype"],
        "device": trace["device"],
        "prompt": trace["prompt"],
        "completion": trace["completion"],
        "n_prompt_tokens": trace["n_prompt_tokens"],
        "prefix_token_ids": prefix,
        "prefix_texts": [s["token"]["text"] for s in steps[:step_index]],
        "target_token_id": target["token_id"],
        "target_text": target["text"],
        "target_position": target["position"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_id")
    p.add_argument("--step", type=int, help="step index of the target token")
    p.add_argument("--list", action="store_true", help="show the generated tokens and exit")
    p.add_argument("--out", type=Path, help="write the export here (default: out/<id>-<step>.json)")
    args = p.parse_args()

    trace = load(args.trace_id)

    if args.list or args.step is None:
        print(f"{trace['trace_id']}  {trace['model']}  {trace['prompt']!r}")
        print(f"  completion: {trace['completion']!r}")
        for s in generated_steps(trace):
            t = s["token"]
            print(f"  step {s['step']:>3}  pos {t['position']:>3}  id {t['token_id']:>6}  {t['text']!r}")
        if args.step is None:
            raise SystemExit("\npick one with --step")

    record = export(trace, args.step)
    out = args.out or Path(__file__).parent / "out" / f"{args.trace_id}-{args.step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))

    print(f"\nwrote {out}")
    print(f"  prefix : {len(record['prefix_token_ids'])} tokens, "
          f"first={record['prefix_token_ids'][0]} last={record['prefix_token_ids'][-1]}")
    print(f"  target : id {record['target_token_id']} {record['target_text']!r} "
          f"at position {record['target_position']}")


if __name__ == "__main__":
    main()
