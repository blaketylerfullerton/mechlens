## Why

The brain on screen today shows a continuous field: seven depth bands, each a
range of transformer layers, tinted by the logit-lens classification at the
selected token. It answers "how did the answer crystallise with depth" — a
layer x token question, blurred into seven bands and shown one token at a
time, which the trace grid's layer x token surface can answer exactly and all
at once. So the brain spends every spatial dimension it has on a signal that
is better read as a grid, and shows nothing the grid could not.

Meanwhile the one thing in a trace that is genuinely *discrete and nameable* —
the SAE features that fired, with an activation value and a human-readable
Neuronpedia label — has no visual representation at all, and is not even on the
wire: `TracePass` is `Literal["lens"]`, so every trace the frontend has ever
received carries `features: []`. Phases 2 and 3 exist, are measured, and are
invisible.

This change makes the brain a map of the model's features and a trace the light
on it: which features fired, how strongly, in which clusters, and when. It also
closes one of the two gaps the README calls out — the Neuronpedia explanation
embeddings that are importable but "nothing consumes them yet" — and it
implements the UMAP layout `backend/app/labels.py` already anticipates by name.

## What Changes

**A feature atlas, built once and shipped.** One node per (layer, feature) —
26 x 16,384 = 425,984 for `gemma-2-2b` at 16k. **This change ships the pilot
six layers of that — 0, 5, 10, 16, 20 and 25, some 98,210 features — and defers
the remaining twenty to a follow-on change.** The six span the depth range and
straddle Neuronpedia's explainer split, and the label store covers exactly the
same set, so a trace pinned to them via `sae_layers` gets a label and a position
for every feature it records. All 26 layers remain the goal; see design decision
2b for why they are not the POC. Positions come from a UMAP
projection of the SAE decoder rows `W_dec[layer][f]` into 3D, then warped
through the brain shell's existing `shapeEllipsoid` so the cloud fills the
rendered brain. The projection is deterministic (fixed seed) and versioned; a
trace records which atlas version it was drawn against.

**Positions from the model, names from the labels.** All 26 layers' decoder
rows live in one shared residual-stream space, so pooling them for a layout is
legitimate. Neuronpedia's explanation embeddings are *not* used for position:
five of the 26 layers carry a different explainer, which would cluster those
layers together for reasons about the explainer rather than about gemma — the
confound `labels.py` documents. They are used instead to name and to validate
clusters, which is what they can honestly support.

**Areas, named only when the naming is earned.** Clusters in the layout become
"areas". An area is named from a representative member label only when its
members' measured label-embedding coherence clears a threshold; otherwise it
stays explicitly unnamed rather than carrying an invented summary.

**The atlas is checked, not asserted.** The build records k-nearest-neighbour
preservation (how much of the 2304-dimensional neighbourhood structure survives
in 3D), per-cluster label coherence against a shuffled baseline, cluster
agreement between the decoder-direction layout and a label-embedding layout,
and a layout hash. These are the atlas's equivalents of `explained_variance`,
`final_layer_agreement` and `reconstruction_max_rel_gap`, and they are surfaced
rather than kept in a build log.

**BREAKING: layer stops being a spatial axis on the brain.** Depth is handed
over to the layout, so a concept appears once as a region instead of 26 times as
a thin disc, and "which layer" becomes "when it lights up" — driven by the
existing shared (layer, token) selection, which already makes the trace grid a
token x layer transport control. The seven layer bands, their rings, the
crossover ring and the band hover breakdown are removed from the brain; the
logit-lens classification they carried moves to the trace grid, which gains a
colour mode for it and a crossover marker on its layer axis, so the reading
changes surface rather than disappearing.

**BREAKING: `Trace` schema 1.3 -> 1.4.** Adds `Trace.layout`, a side table
keyed `"layer/index"` in the same shape and for the same reason as
`Trace.labels`, plus the atlas reference the layout was taken from.

**The SAE and label passes reach the service.** `POST /trace` accepts `"sae"`
and `"labels"` alongside `"lens"`, and the job reports an `sae` phase. Unlike
`generating`, that phase genuinely walks layers, so it can drive a real
per-layer sweep while a run is in flight.

**Feature search.** Filtering the lit nodes by label text turns the atlas into
a query over the model's own concepts.

## Capabilities

### New Capabilities
- `feature-atlas`: the layout build and its correctness story — UMAP over SAE
  decoder directions, clustering, conditional area naming from label
  embeddings, the four diagnostics, determinism and versioning, and how a
  trace refers to the atlas it was drawn against.

### Modified Capabilities
- `frontend-brain-view`: replaces layer bands with the feature atlas — nodes
  positioned by the layout and lit by measured activation, named areas,
  level-of-detail, the shared selection as a transport, and a rewritten set of
  honesty requirements covering UMAP's distorted global distances, the top-k
  activation coverage, the BOS artifact, and unnamed clusters.
- `api-service`: `POST /trace` accepts the `sae` and `labels` passes, the job
  reports an `sae` phase, and a returned trace carries the layout for the
  features it reports.

### Unchanged Capabilities
- `steering` and `attribution` are untouched. Attribution edges are not drawn
  on the atlas: an attention edge runs between token positions, and token
  position has no spatial axis here.

## Impact

**Backend**
- `app/schema.py` — schema 1.4: `Trace.layout`, atlas reference.
- `app/service/models.py` — `TracePass` gains `"sae"` and `"labels"`.
- `app/service/jobs.py` — `JobPhase` and `_PHASE_ORDER` gain `"sae"`.
- `app/service/app.py` — runs the two passes, attaches the layout.
- `app/labels.py` — a `layout` table beside `labels`; `embeddings()` gains its
  first caller.
- `scripts/build_feature_atlas.py` — new. Defaults to all 26 layers, which
  needs every SAE resident (~7.9GB at 16k) once and the label DB with
  `--embeddings` (~2GB). The POC builds the pilot six instead: ~1.8GB of SAEs
  and the ~458MB label DB already on disk.
- New dependencies: `umap-learn` and a density clusterer, build-time only —
  neither is imported by the service or by `schema.py`.

**Frontend**
- `components/Brain.tsx` — bands, rings and their hover panel removed; a point
  cloud, area markers, level-of-detail, and picking added.
- `lib/lens.ts` — band helpers (`bandLayers`, `blendBand`, `bandOfLayer`) lose
  their only consumer and are deleted; the per-cell classification stays, and
  gains its first renderer in the grid.
- `components/TraceViewer.tsx` — the residual map gains a colour mode for the
  `answer` / `echo` / `other` classification and a `crossover_layer` marker on
  its layer axis, so the reading the bands carried has somewhere to live.
- `lib/api-types.ts` — mirrors the schema and `TracePass` changes.
- A static idle-atlas subsample so the brain still has structure with no trace
  loaded.

**Not affected**
- The CLI, `store.py`, the residual sidecar format, and the attribution and
  steering paths.
