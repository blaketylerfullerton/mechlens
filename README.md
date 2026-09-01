# mechlens

Mechanistic-interpretability tracing for `gemma-2-2b` under TransformerLens.

Generate token by token, capture the residual stream at every layer, then run
enrichment passes over the saved trace — SAE features and Neuronpedia labels
now, logit lens and attribution next.

## Status

Phases 0–3 are done; the trace schema is at **1.2**.

| phase | what | state |
| --- | --- | --- |
| 0 | model loading — `gemma-2-2b` under TransformerLens, bf16 on CUDA | done |
| 1 | residual capture — token-by-token, all 26 layers | done |
| 2 | SAE encoding — Gemma Scope 16k, top-k features per (token, layer) | done |
| 3 | Neuronpedia labels — human-readable text and links for those features | done |
| — | logit lens — `LayerState.logit_lens` defined and empty | unclaimed |
| 4 | attribution — `LayerState.edges` defined and empty | next |

Measured on the traces in `backend/traces/`:

| | |
| --- | --- |
| SAE health | mean L0 **78.3**, mean explained variance **0.880** (per-layer 0.83–0.96) |
| SAE encode | **0.3s** for a 31-token trace, 26 layers, once the SAEs are resident |
| label coverage | **6234/6234** distinct features on golden-gate, in **1.5s**, no network |
| label table | **425,679** explanations, 26 layers, imported in **51s** |
| mapping check | **10/10** features matched Neuronpedia's own activations (corr ≥ 0.998) |
| tests | **52 passed, 1 skipped** in ~7s |

Two known gaps, both deliberate. Explanation embeddings are not imported by
default (`--embeddings`, ~2GB) and nothing consumes them yet. And position 0
(BOS) produces meaningless SAE activations — a known Gemma Scope artifact, kept
per-token but excluded from every summary statistic.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # gemma is gated: accept the license at hf.co/google/gemma-2-2b
```

## Use

Everything runs from `backend/`:

```bash
python -m app.cli trace -p "The Golden Gate Bridge is located in the city of" -n 20
python -m app.cli enrich traces/<id>.json --sae     # SAE features, no model load
python -m app.cli enrich traces/<id>.json --labels  # Neuronpedia labels, no network
python -m app.cli show traces/<id>.json --layer 20
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels   # or all three at once
```

`--labels` reads a local SQLite table built once from Neuronpedia's public export:

```bash
python scripts/import_neuronpedia.py               # 425,679 labels, ~50MB, ~50s
python scripts/verify_neuronpedia_mapping.py       # confirm our features are their features
```

```
layer 20, token 30 ' The'  (l0=80)
  #6631     84.75  ████████████  The Future
  #1370     81.21  ███████████   bridge and crossing
  #3124     29.10  ████          San Francisco, Oakland, Bay Area
  https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/6631
```

A trace is two files that travel together:

| file | what |
| --- | --- |
| `<id>.json` | the `Trace` document — tokens, next-token distributions, per-layer state |
| `<id>.residuals.npy` | `[n_tokens, n_layers, d_model]` float32 residual stream (`hook_resid_post`) |

Splitting them is what makes `enrich` cheap: a pass reads the tensor off disk and
never touches gemma, so iterating on pass code costs seconds instead of a
generation run.

Neither is committed. `backend/traces/` and `backend/data/` are gitignored — a
trace is a few MB of JSON plus a multi-MB `.npy`, and the label DB (66MB) is
rebuildable from the export in under a minute.

```python
from app.store import load
trace, residuals = load("traces/golden-gate.json")   # mmap=True to page it in lazily
residuals[7, 11]                                     # token 7, layer 11 -> [d_model]
trace.steps[7].layers[11].features                   # its top SAE features
trace.label(11, 4023)                                # what feature 4023 means
```

Labels live in `Trace.labels` keyed `"layer/index"`, not on each `Feature`: a
feature recurs ~2x per trace, so copying the text onto every occurrence would
roughly double the JSON for no added information.

## Layout

| module | role |
| --- | --- |
| `app/schema.py` | the trace schema — the JSON contract everything writes into |
| `app/capture.py` | phase 1: token-by-token loop, residual stream cached at every layer |
| `app/passes/sae.py` | phase 2: Gemma Scope SAE features per (token, layer) |
| `app/passes/labels.py` | phase 3: Neuronpedia labels for those features |
| `app/labels.py` | the label store — SQLite lookup, offline, with a capped API fallback |
| `app/passes/__init__.py` | the `Pass` protocol — take a trace + residuals, fill fields |
| `app/store.py` | save/load: JSON document plus its `.npy` sidecar |
| `app/model_cache.py` | one model per process (loading gemma costs ~6s) |
| `app/sae_cache.py` | one SAE per layer per process (~302MB each at 16k) |
| `app/cli.py` | `trace` / `enrich` / `show` |
| `scripts/import_neuronpedia.py` | one-time load of the explanation export into SQLite |
| `scripts/verify_neuronpedia_mapping.py` | proves our SAE features are the ones Neuronpedia labelled |

`LayerState.logit_lens` and `.edges` are defined and empty — the logit lens and
the attribution pass fill them on traces that already exist.

One caveat worth knowing before comparing labels across layers: Neuronpedia's
export does not use a single explainer. For `gemma-2-2b` at 16k, layers 16, 18,
20, 22 and 24 carry `gemini-2.5-flash-lite` explanations and the other 21 carry
`gpt-4o-mini`, and the two write in visibly different styles. Every label
records its `explainer` for that reason. See `phase3.md`.

## Tests

```bash
pytest backend/tests -q            # ~7s: a 1M-param model on CPU, and a stand-in SAE
MECHLENS_SLOW=1 pytest backend/tests -q   # adds a real Gemma Scope SAE (302MB download)
```
