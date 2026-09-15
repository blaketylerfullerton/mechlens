# The step videos

Five short explainers, one per step of the pipeline, in the order someone meets
them:

| | | file |
| --- | --- | --- |
| 1 | get activations | — |
| 2 | **sparsify with an SAE** | `step2_sparse_features.py` |
| 3 | check it didn't lie to you | — |
| 4 | label the features | — |
| 5 | place them in space (UMAP) | — |

`architecture.py` predates the series. It is the single conceptual overview
(`sae-concept.mp4`) and overlaps steps 2 and 4; leave it or retire it once the
five exist.

## Rendering

```bash
./venv/bin/manim -pql docs/manim/step2_sparse_features.py SparseFeatures   # draft, 480p15
./venv/bin/manim -pqh docs/manim/step2_sparse_features.py SparseFeatures   # 1080p60
```

`manim` is not in `backend/requirements.txt` — it is a docs tool, not a runtime
dependency, and it pulls in cairo, pango and ffmpeg. Install it into the venv
when you need it:

```bash
./venv/bin/pip install manim
```

Renders land in `media/` (gitignored). Copy the one you want to keep into
`docs/manim/`.

## The rule these videos follow

**Every number on screen was computed, not chosen.** `extract.py` reads a real
saved trace and writes `data/step2.json`; the scene reads that file and draws
nothing else. Bar widths come from `value / max`, the sparse marks sit at their
true feature indices, and the labels are Neuronpedia's actual text — including
its artifacts, like the `<ctrl63>` in feature #6492. An invented proportion or a
plausible-looking activation would make these videos exactly the thing the
project exists to argue against.

Traces are gitignored, so the extracted JSON is committed. Regenerate it against
a different trace with:

```bash
python docs/manim/extract.py backend/traces/<id>.json
```

It expects a trace that has had the `sae` and `labels` passes run.

## Style

`theme.py` holds the palette and the two text helpers, mirroring
`frontend/src/index.css` — which is the source of truth. If they disagree, the
CSS wins.

- `human()` for anything a person wrote, `machine()` for anything the system
  emitted: tokens, indices, hooks, hook names, numbers.
- `fn` (#82AAFF) is the lead accent and marks the one row that is the point.
  `num`/`str`/`const` appear only in the diagnostics panel, which reads as a
  legend.
- No gradient, no glow, no shadow, nothing loops.
- **The caveat never hides.** Each video ends on what its step cost and how you
  would know it went wrong — for step 2 that is `16 of 83`, the explained
  variance, and the mean L0.

Inter and JetBrains Mono are not installed on this machine, so `theme.py`
falls back to DejaVu. Install the real faces for a final render.
