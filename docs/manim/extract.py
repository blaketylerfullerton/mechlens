"""Pull the real numbers the step videos draw out of a saved trace.

The videos must not contain a quantity nobody computed, so every figure in them
comes from here. Traces are gitignored; the extracted JSON is small and is
committed, so `manim` renders for someone who has never run the model.

    python docs/manim/extract.py backend/traces/<id>.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# The cell the step-2 video walks through. Layer 20 carries the Golden Gate
# features this prompt exists to show; position 4 is the ' Bridge' token.
LAYER, POSITION = 20, 4
OUT = Path(__file__).parent / "data" / "step2.json"


def main(path: Path) -> None:
    trace = json.loads(path.read_text())
    residuals = np.load(path.with_suffix(".residuals.npy"))
    sae = next(p for p in trace["passes"] if p["name"] == "sae")
    params, stats = sae["params"], sae["stats"]

    step = trace["steps"][POSITION]
    state = step["layers"][LAYER]
    labels = trace.get("labels") or {}

    def label_of(index: int):
        entry = labels.get(f"{LAYER}/{index}")
        if isinstance(entry, dict):
            return entry.get("text") or entry.get("description")
        return entry

    vector = residuals[POSITION, LAYER]
    payload = dict(
        source=dict(trace=path.name, prompt=trace.get("prompt"), model=trace["model"],
                    release=params["release"], width=params["width"], hook=params["hook"],
                    top_k=params["top_k"], n_layers=params["n_layers"]),
        cell=dict(layer=LAYER, position=POSITION, token=step["token"]["text"],
                  l0=state["l0"], d_model=int(vector.shape[0]),
                  d_sae={"16k": 16384, "65k": 65536, "262k": 262144}[params["width"]]),
        # The real residual, for a strip drawn to scale rather than sketched.
        residual=dict(head=[round(float(v), 2) for v in vector[:48]],
                      norm=round(float(np.linalg.norm(vector)), 1),
                      min=round(float(vector.min()), 2), max=round(float(vector.max()), 2)),
        # Only the top k are saved in a trace; l0 above says how many really fired.
        features=[dict(index=f["index"], activation=round(f["activation"], 2), label=label_of(f["index"]))
                  for f in state["features"]],
        diagnostics=dict(l0_mean=round(stats["l0_mean"], 1),
                         explained_variance_mean=round(stats["explained_variance_mean"], 4),
                         explained_variance_min=round(stats["explained_variance_min"], 4)),
    )
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} from {path.name}: layer {LAYER}, token {payload['cell']['token']!r}, l0={payload['cell']['l0']}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "backend/traces/0405a85811f4.json"))
