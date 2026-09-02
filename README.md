# mechlens

![Dashboard](/docs/dashboard.png)
Mechanistic-interpretability tracing for `gemma-2-2b` under TransformerLens.

Generate token by token, capture the residual stream at every layer, then run
enrichment passes over the saved trace — SAE features, Neuronpedia labels, the
logit lens, and now attribution. A FastAPI service now sits in front of all
of it, so tracing and feature steering are also reachable over HTTP.

## Status

Phases 0–6 are done; the trace schema is at **1.3**.

| phase | what | state |
| --- | --- | --- |
| 0 | model loading — `gemma-2-2b` under TransformerLens, bf16 on CUDA | done |
| 1 | residual capture — token-by-token, all 26 layers | done |
| 2 | SAE encoding — Gemma Scope 16k, top-k features per (token, layer) | done |
| 3 | Neuronpedia labels — human-readable text and links for those features | done |
| 4 | logit lens — every layer decoded through `ln_final` + `W_U` | done |
| 5 | attribution — exact resid/attn/mlp decomposition of every layer | done |
| 6 | API service — FastAPI `/trace`, `/steer`, `/feature`, job queue, GPU lock | done |
| 7 | feature-level attribution — `kind="sae"` edges, deferred from phase 5 | next |

Measured on the traces in `backend/traces/`:

| | |
| --- | --- |
| SAE health | mean L0 **78.3**, mean explained variance **0.880** (per-layer 0.83–0.96) |
| SAE encode | **0.3s** for a 31-token trace, 26 layers, once the SAEs are resident |
| label coverage | **6234/6234** distinct features on golden-gate, in **1.5s**, no network |
| label table | **425,679** explanations, 26 layers, imported in **51s** |
| mapping check | **10/10** features matched Neuronpedia's own activations (corr ≥ 0.998) |
| lens | 26 layers x 31 tokens in **2.3s**; on all five saved traces layer 25 reproduces the model's own output at **every** position, max prob and entropy delta **0.0** |
| attribution | 26 layers x 31 tokens in **1.2s**; reconstruction gap **≤1.3e-2** across all five traces (bf16 tail, per-layer mean ~0.004–0.005, flat with depth); attn top-8 coverage **90–98%** |
| tests | **118 passed, 1 skipped** in ~17s |

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
python -m app.cli enrich traces/<id>.json --lens    # logit lens, loads the model
python -m app.cli enrich traces/<id>.json --attribution  # resid/attn/mlp edges, loads the model
python -m app.cli show traces/<id>.json --layer 20
python -m app.cli show traces/<id>.json --lens --token 8
python -m app.cli show traces/<id>.json --attribution --token 8
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels --lens --attribution   # or all at once
```

`--sae` and `--labels` never load gemma. `--lens` and `--attribution` are the
exceptions: `W_U` is 2304 x 256,000, too big to sit in a sidecar beside every
trace, and attention patterns and per-head values aren't in the sidecar at
all — so both need a fresh pass through the model. When you run them as part
of `trace` the model is already resident and it costs nothing extra to load.

## The logit lens

Ask the model's own output head what it would answer at layer L instead of at
layer 25, at every depth, and you can watch a fact resolve:

```
  token 8 ' of' — the model answers ' Paris' 92.6%
   ◀ = the model's final answer   · = an echo of the current token

   L3   ' of'            71.81%  H  1.76  · ██████████████
   L7   'بوابة'          37.97%  H  3.94    ████████
   L17  ' city'          40.87%  H  2.60    ████████
   L19  ' Paris'         22.47%  H  3.15  ◀ ████
   L20  ' Paris'         92.66%  H  0.46  ◀ ███████████████████
   L25  ' Paris'         92.59%  H  0.60  ◀ ███████████████████
```

Three things about that output are worth knowing before you read one:

**It is checkable, and it is checked.** Layer 25 is `resid_post` of the last
block, which is precisely what `forward` hands to `ln_final` — so the last
layer's lens is not an approximation of the model's output, it *is* the model's
output, recomputed. `final_layer_agreement` on the pass record must be 1.0; on
gemma the probabilities come back bit-identical. Anything less means the softcap
or the norm is wrong.

One wrinkle that looks like a bug and is not: gemma's late-layer logits sit
around 28, where bf16 steps by 0.125, so the top two candidates sometimes land
on the *same* value and `topk` breaks the tie arbitrarily. Same distribution,
same entropy, different argmax. Those positions count as agreement and are
reported as `final_layer_argmax_ties` rather than absorbed;
`final_layer_max_entropy_delta` is the check that is blind to tie-breaking
altogether. See `phase4.md`.

**The softcap is easy to miss.** Gemma 2 caps logits at 30 with
`cap * tanh(x / cap)`, and TransformerLens applies it in `forward`, *after*
`unembed` — not inside it. Skip it and top-1 still looks about right while every
probability and entropy is quietly wrong. The pass calls the model's own
`apply_softcap` so the two cannot drift.

**Early layers echo, they do not predict.** Residuals near the embedding decode
back to the token already sitting at that position — on gemma, layers 0–9 read
the current token back at 60–85% confidence. That is why entropy is *not* the
headline number: it starts low for a reason that has nothing to do with the
model being sure of an answer. `top1_agreement_by_layer` is the honest
crystallisation curve, `echo_by_layer` says how much of its early portion to
discount, and `crossover_layer` — the first depth where the answer is already
in place for half the positions — is the one number to read.

## Attribution

For every layer and position, `resid_post = resid_pre + attn_out + mlp_out` —
not an approximation but the literal structure of the residual stream, so
decomposing it into edges needs no gradients or patching, just the model's
own intermediate tensors:

```
   L20  resid 340.1  mlp 117.1  attn [pos0=28.7, pos4=14.2, pos20=11.7]
   L21  resid 387.7  mlp 129.8  attn [pos29=27.5, pos0=26.8, pos30=17.1]
   L25  resid 624.1  mlp 236.3  attn [pos30=149.3, pos0=27.8, pos4=26.6]
```

**It is checkable, and it is checked.** `resid_pre + attn_out + mlp_out` should
equal the captured `resid_post` exactly, the same class of claim as the lens's
final-layer identity — `reconstruction_max_rel_gap` on the pass record is that
check. On gemma it reads **≤1.3e-2**, which looks less clean than the lens's
**0.0** until you notice it is a *max* over 26 layers x 31 positions of bf16
noise (mean **~0.004–0.005** per layer, flat with depth) rather than a single
number — a real bug shows up as the mean drifting, not the max's tail.

**Attention needs a per-source decomposition, not a per-head one.** Gemma 2's
attention output is normalised again before it joins the residual stream (the
"sandwich norm"), and that normalisation's scale and gain are computed from
the whole summed output, not per source — so it distributes over a per-source
split exactly, computed once from the total and applied identically to every
term. Grouped-query attention adds a second wrinkle: the cached value vectors
sit at 4 KV-head granularity, not gemma's 8 query heads, and have to be
expanded to line up with the attention pattern before they mean anything.

Edges are truncated to the top 8 source positions by weight, same as the SAE
pass truncates features — `attn_topk_coverage` says how much of the real
total that top-8 keeps, honestly: **90–98%** across the five saved traces,
lower on longer ones where attention has more positions to spread across.

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

Splitting them is what makes `enrich` cheap: a pass reads the tensor off disk
instead of regenerating, so iterating on pass code costs seconds. The SAE and
label passes never touch gemma at all; the lens needs `W_U` and attribution
needs the attention pattern and per-head values, so both pay the ~10s load.

Neither is committed. `backend/traces/` and `backend/data/` are gitignored — a
trace is a few MB of JSON plus a multi-MB `.npy`, and the label DB (66MB) is
rebuildable from the export in under a minute.

```python
from app.store import load
trace, residuals = load("traces/golden-gate.json")   # mmap=True to page it in lazily
residuals[7, 11]                                     # token 7, layer 11 -> [d_model]
trace.steps[7].layers[11].features                   # its top SAE features
trace.steps[7].layers[11].logit_lens.top_k[0]        # what it would answer here
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
| `app/passes/lens.py` | phase 4: every layer decoded through `ln_final` + `W_U` |
| `app/passes/attribution.py` | phase 5: resid/attn/mlp edges decomposing every layer's residual |
| `app/labels.py` | the label store — SQLite lookup, offline, with a capped API fallback |
| `app/passes/__init__.py` | the `Pass` protocol — take a trace + residuals, fill fields |
| `app/store.py` | save/load: JSON document plus its `.npy` sidecar |
| `app/model_cache.py` | one model per process (loading gemma costs ~6s) |
| `app/sae_cache.py` | one SAE per layer per process (~302MB each at 16k) |
| `app/cli.py` | `trace` / `enrich` / `show` |
| `scripts/import_neuronpedia.py` | one-time load of the explanation export into SQLite |
| `scripts/verify_neuronpedia_mapping.py` | proves our SAE features are the ones Neuronpedia labelled |

`LayerState.edges` holds `resid`/`attn`/`mlp` contributions once the
attribution pass has run. `kind="sae"` edges — attributing a feature's own
activation to upstream contributions, rather than the residual stream as a
whole — are deferred to a later phase: it needs the SAE encoder's own
reconstruction error in the loop, a different correctness story than the
exact decomposition here.

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
