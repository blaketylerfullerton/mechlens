# mechlens

<img width="1704" height="964" alt="Screenshot 2026-09-08 at 10 21 00 PM" src="https://github.com/user-attachments/assets/e79d946a-34db-4acb-bd81-14529f7dc4a8" />

Mechanistic-interpretability tracing for `gemma-2-2b` under TransformerLens.

Generate token by token, capture the residual stream at every layer, then run
enrichment passes over the saved trace — SAE features, Neuronpedia labels, the
logit lens, and attribution. A FastAPI service sits in front of all of it, so
tracing and feature steering are also reachable over HTTP. Phase 7 is under way:
every SAE feature gets a fixed place in a *feature atlas*, so a trace can be
drawn as the features it actually lit up.

## Status

Phases 0–6 are done, phase 7 is partly landed; the trace schema is at **1.4**.

| phase | what | state |
| --- | --- | --- |
| 0 | model loading — `gemma-2-2b` under TransformerLens, bf16 on CUDA | done |
| 1 | residual capture — token-by-token, all 26 layers | done |
| 2 | SAE encoding — Gemma Scope 16k, top-k features per (token, layer) | done |
| 3 | Neuronpedia labels — human-readable text and links for those features | done |
| 4 | logit lens — every layer decoded through `ln_final` + `W_U` | done |
| 5 | attribution — exact resid/attn/mlp decomposition of every layer | done |
| 6 | API service — FastAPI `/trace`, `/steer`, `/feature`, job queue, GPU lock | done |
| 7 | feature atlas — a fixed position per feature, and the brain drawn from it | in progress |
| 8 | feature-level attribution — `kind="sae"` edges, deferred from phase 5 | next |

Phase 7, in more detail — the parts that are in and the parts that are not:

| | state |
| --- | --- |
| `POST /trace` runs the `sae` and `labels` passes; the job reports an `sae` phase | done |
| the atlas build, its five diagnostics, and two source representations | done |
| schema 1.4 — `Trace.layout`, a side table keyed `"layer/index"` | done |
| serving the layout with a trace, and the atlas endpoint | not started |
| the brain renders atlas nodes, and says what the layout does and does not claim | done |
| lighting nodes from a trace's activations, areas, the transport, label search | not started |
| retiring the layer bands — the brain still draws both | not started |

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
| atlas (label source) | 98,210 features over 6 layers; kNN preservation **0.292**, 32 clusters, **27** earning a name at coherence **0.465** against a random baseline of **0.233**; explainer influence **0.026** |
| atlas (decoder source) | 98,304 features over the same 6 layers; kNN preservation **0.143**, 12 clusters, **0** earning a name (margin **+0.049**, needs +0.15) |
| tests | **286 passed, 1 skipped** in ~53s |

The atlas numbers above are a **6-layer pilot** (0, 5, 10, 16, 20, 25), not the
full 26. Those six were picked to span the depth range and to include two of the
five layers Neuronpedia explained with a different model, so the explainer
confound could be measured rather than assumed.

One known gap, deliberate: position 0 (BOS) produces meaningless SAE
activations — a known Gemma Scope artifact, kept per-token but excluded from
every summary statistic. The other long-standing gap is now closed: the
explanation embeddings (`--embeddings`, ~2GB) have a consumer, and it turned out
to be a bigger one than expected — see the atlas section below.

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

Over HTTP, the same passes are opt-in per request. `POST /trace` takes
`passes: ["sae", "labels", "lens"]` and reports an `sae` phase while it runs —
that pass really does walk layers one at a time, unlike generation, so it is one
of the two phases that can honestly report a depth sweep. Asking for `labels`
without `sae` is a 422: the label pass labels the features the SAE pass records,
so the dependency is caught at the boundary rather than on the worker thread.
`sae_layers` restricts the pass to a subset, because 26 resident 16k SAEs come
to ~7.9GB — fine on a unified-memory box, not fine on a 16GB discrete GPU.

The atlas is built separately, once, and is not per-trace:

```bash
python scripts/import_neuronpedia.py --embeddings      # +explanation vectors, ~2GB
python scripts/build_feature_atlas.py                  # all 26 layers
python scripts/build_feature_atlas.py --dry-run        # measure and print, write nothing
```

`--dry-run` exists because an atlas nobody looked at the numbers for is not
worth building on: it reports kNN preservation, cluster coherence against its
baseline, and explainer influence without writing anything.

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

## The feature atlas

A trace says which features fired. It does not say where to *draw* them, and a
position has to come from somewhere that means something. So every feature gets
one, once, offline:

```bash
python scripts/build_feature_atlas.py --layers 0,5,10,16,20,25   # ~3 min, reads the label DB
python scripts/build_feature_atlas.py --source decoder           # the other representation, needs the SAEs
```

```
atlas 16k-s0-731e35f436  (98,210 features, source=labels 256d)
  kNN preservation    0.292 (k=20, 4,000 sampled, ±0.180)
  clusters            32 (27 named, 5 unnamed)
  cluster coherence   0.465 vs baseline 0.233 (margin 0.232, need 0.150)
  explainer influence 0.026 (0 = areas are meaning, 1 = areas are the labeller)

  largest named areas:
     8,320  coh 0.441 (+0.208)  L5#8356    programming-related constructs and elements in code
     2,296  coh 0.552 (+0.319)  L25#12372  mathematical notations and expressions
     2,136  coh 0.498 (+0.266)  L0#14707   references to legal terminology and concepts
     1,518  coh 0.517 (+0.283)  L25#12912  numeric values and statistical references
```

**It is checkable, and it is checked.** Flattening to three dimensions throws
away a great deal — from 256 for the label source, from 2,304 for the decoder
one — and the whole value of the picture rests on how much survived, so that is
a measurement rather than an assumption. `knn_preservation` is the fraction of
a feature's 20 nearest neighbours in the source space still among its 20
nearest in the layout, and it is this pass's peer to the SAE's
`explained_variance` and the lens's `final_layer_agreement`. It reads **0.292**
on the shipping atlas, against **0.0002** for a random layout of the same
98,210 features. It is published in the interface, not buried in a build log.

**Near means similar; far means nothing.** UMAP preserves local neighbourhoods
and distorts everything larger, so "these two nodes are close" is a real
statement and "these two areas are far apart" is not one. The view therefore
ships with no axes, no coordinate readout and no distance scale: there is no
honest way to answer a question about distance here, so the question is not
offered.

**Two sources, because they answer different questions.** An atlas is projected
from one of two representations and records which:

| | what "near" means | preservation | areas |
| --- | --- | --- | --- |
| `labels` | an explainer *described* two features similarly | **0.292** | 32, 27 named |
| `decoder` | the model *writes* the two features similarly | 0.143 | 12, none named |

The label source is the default, and that reverses this project's original
decision. Positions were first taken from `W_dec` alone, on the grounds that a
layout over label text would inherit Neuronpedia's explainer split — layers 16,
18, 20, 22 and 24 carry `gemini-2.5-flash-lite` explanations and the other 21
`gpt-4o-mini` — and separate those five layers for a reason about the labeller
rather than about gemma. Sound reasoning; false premise. Measured on a pilot
chosen to contain both explainers, the influence is **0.026** — no meaningful
effect. Meanwhile the decoder layout cannot produce areas at all: HDBSCAN in the
full 2,304-dimensional space calls **99.1%** of features noise, and a forced
k-means partition earns **zero** names at k=24 or k=60. It is a continuum, not
an archipelago. Both atlases are built and kept; `app/atlas.py` and
`design.md` carry the numbers.

**A name has to be earned, against the right baseline.** An area is named after
its most central member's own label — a real feature says "URLs and web links",
and the interface can show which one — but only when the members' label
agreement beats a *random group of the same size* by a recorded margin. That
relative test replaced an absolute threshold, and the replacement was not
cosmetic: the first pilot measured a mean coherence of 0.281 against a baseline
of 0.232 and duly named 14 clusters that beat chance by 0.049. The names read
plausibly, which is what made it dangerous. Explanation embeddings from one
explainer share a similarity floor well above zero, so an absolute threshold
measures the floor and not any agreement. Under the relative gate the decoder
atlas names none of its 12 clusters, which is the correct answer.

**What the nodes leave out is stated.** A trace records the top 16 features per
layer out of an `l0` of ~78 that actually fired, so the interface reports both
rather than letting the lit nodes read as the complete set — the same habit as
`attn_topk_coverage`. BOS is excluded. And a node is a feature, not a place: no
named anatomical region computes any of this, and the shell is silhouette.

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
| `app/atlas.py` | phase 7: the feature atlas — the shell warp, its diagnostics, the naming gate |
| `app/labels.py` | the label store — SQLite lookup, the atlas tables, a capped API fallback |
| `app/passes/__init__.py` | the `Pass` protocol — take a trace + residuals, fill fields |
| `app/store.py` | save/load: JSON document plus its `.npy` sidecar |
| `app/model_cache.py` | one model per process (loading gemma costs ~6s) |
| `app/sae_cache.py` | one SAE per layer per process (~302MB each at 16k) |
| `app/cli.py` | `trace` / `enrich` / `show` |
| `scripts/import_neuronpedia.py` | one-time load of the explanation export into SQLite |
| `scripts/verify_neuronpedia_mapping.py` | proves our SAE features are the ones Neuronpedia labelled |
| `scripts/build_feature_atlas.py` | one-time UMAP layout of every feature; the only module importing umap |

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
pytest backend/tests -q            # ~53s: a 1M-param model on CPU, and stand-in SAEs
MECHLENS_SLOW=1 pytest backend/tests -q   # adds a real Gemma Scope SAE (302MB download)
```

The atlas tests deserve a note on why they are the slow ones.
`umap.UMAP(random_state=...)` is single-threaded — that is the price of an
artifact whose whole value is being the same map every time — so
`test_atlas_build.py` exercises the real pipeline end to end on 192 synthetic
features with planted structure rather than on 425,984 real ones. The geometry,
the measurements and the naming gate are tested separately in `test_atlas.py`,
on a dozen hand-written vectors.

Two of those tests are worth knowing about because they check things a comment
cannot. `test_shape_ellipsoid_matches_the_typescript_it_ports` lifts
`shapeEllipsoid` straight out of `Brain.tsx`, runs it under `node`, and compares
against the Python port to 1e-12 — the mesh shapes its surface with one and the
atlas places nodes inside it with the other, so the two agreeing is not
optional. And `test_atlas_deps.py` imports each module in a fresh interpreter to
confirm `umap` never reaches the service: the atlas is a precomputed table, so
serving it needs no projection library, and `app/schema.py` still imports
nothing but pydantic.
