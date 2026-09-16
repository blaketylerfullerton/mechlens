# CIR-01 prototype — output-targeted attribution

Feasibility work for the Circuits workspace (`docs/circuits-feature-spec.md`).
The findings and the go/no-go call live in `docs/circuits-cir01-report.md`.

This directory has **its own venv on purpose**. `circuit-tracer` pins
`transformers>=4.56,<=4.57.3`; the mechlens venv runs 5.16.1. Nothing here is
importable from `backend/`, and `backend/requirements.txt` is untouched.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock      # exact, reproducible
```

`requirements.in` records what was asked for; `requirements.lock` is what
resolved and was measured. Model and transcoder weights come from the shared
HuggingFace cache, so nothing re-downloads if mechlens has already pulled
`gemma-2-2b`. The transcoder set is ~4GB on first use.

## Scripts

| | |
| --- | --- |
| `common.py` | model loading, provenance, atomic JSON writes |
| `export_trace.py` | a saved mechlens trace → exact prefix ids + target id |
| `trace_graph.py` | run `attribute()` on that pair, save graph + viewer files |
| `verify_identity.py` | BOS/token/target checks and baseline agreement. Exits non-zero on failure |
| `intervene.py` | baseline / zero-control / ablation on the graph's top features |
| `feature_evidence.py` | activation examples and logit effects for a feature |
| `bench.py` | runtime, peak memory, graph size; probes progress and cancellation |

## A full pass

```bash
python3 export_trace.py 0405a85811f4 --list          # pick a generated token
python3 export_trace.py 0405a85811f4 --step 12

.venv/bin/python trace_graph.py     --export out/0405a85811f4-12.json --name ggb-francisco
.venv/bin/python verify_identity.py --export out/0405a85811f4-12.json
.venv/bin/python intervene.py       --export out/0405a85811f4-12.json \
                                    --graph  out/ggb-francisco.pt --top 3 --constrained
.venv/bin/python feature_evidence.py 21:6271 24:2455
.venv/bin/python bench.py --tokens 8 16 32 64
```

`export_trace.py` deliberately runs under plain `python3`: it only reads JSON, and
keeping it out of the prototype venv is a reminder that it must never import
mechlens code.

View the saved graphs in the bundled upstream viewer:

```bash
.venv/bin/circuit-tracer start-server --graph_file_dir out/graph_files
```

Everything lands in `out/`, which is gitignored — the artifacts are large
(a 12-token graph is ~114MB as `.pt`, ~22MB as viewer JSON).
