## Context

See proposal.md — Why. The constraints that shape the approach:

- **The features are already computed and already invisible.** `SAEPass` fills
  `LayerState.features` with the top 16 activations per (token, layer) plus
  `l0`, the true count that fired (mean **78.3** on the saved traces). The
  label pass resolves those to Neuronpedia text. Neither reaches the browser:
  `TracePass` is `Literal["lens"]`.
- **Decoder rows across layers are comparable.** Every layer's SAE decodes into
  the same residual-stream basis (`d_model = 2304`), so pooling `W_dec` rows
  from all 26 layers into one layout is a legitimate operation on one vector
  space, not a splice of 26 unrelated ones.
- **The label text is not uniform.** Layers 16, 18, 20, 22 and 24 carry
  `gemini-2.5-flash-lite` explanations; the other 21 carry `gpt-4o-mini`.
  `app/labels.py` names this and names the consequence: any layout built from
  pooled label text will show those five layers as their own region for reasons
  that have nothing to do with gemma.
- **The pieces for this were built and left unwired.**
  `LabelStore.embeddings(layer, features)` is written, documented *"for laying
  features out"*, and has no caller; `import_neuronpedia.py --embeddings` is
  likewise unused. The `labels` table already has an `embedding BLOB` column.
- **Scale.** 26 x 16,384 = **425,984** features; **425,679** have explanations.
  A 31-token trace touches ~**6,234** distinct features across all layers, 416
  per token position, 16 per cell.
- **Hardware.** A 16k SAE is ~302MB; all 26 come to ~**7.9GB**. On this box
  (GB10, one 128GB unified pool) that sits beside gemma's 5GB without shuttling.
  On a 16GB discrete GPU it does not.
- **The brain's existing shell is reusable.** `shapeEllipsoid()` in
  `Brain.tsx` maps a unit direction to the rendered brain's outer form, and is
  already factored out and called from two places.

## Goals / Non-Goals

**Goals**

- One position per feature, fixed across traces, so familiarity accumulates.
- The layout's fidelity is a recorded number, not a claim.
- Areas that carry a name only when the name is earned, and say so otherwise.
- No new render mode: the atlas replaces the bands on the surface that already
  exists.
- Reuse the shared (layer, token) selection as the transport rather than
  inventing a second control.

**Non-Goals**

- **Not** attribution on the brain. An attention edge runs between token
  positions and token position has no spatial axis here; edges stay in the grid.
- **Not** feature-level attribution (`kind="sae"` edges), still deferred.
- **Not** a corpus. Co-activation statistics across many traces would be a
  better clustering signal and are out of scope; a single trace cannot supply
  them.
- **Not** interactive re-layout. The atlas is a build artifact; nothing
  recomputes it at request time.
- **Not** steering changes. `POST /steer` is untouched.

## Decisions

### 1. Layer becomes time, not depth

Depth is handed to the layout; layer is expressed through the selection.

```
   z = layer  (today)                    all 3 dims = layout  (chosen)

   L4   L9   L14  L20  L25               one blob per concept,
    .    .    *    *    *                members spanning many layers
    .    *    *    .    .
    *    *    .    .    .                        * * *
                                                * * A * *      .:. .
   one concept, 26 thin discs                    * * *        .:::::.
```

*Why:* with a layer axis, a cluster of similar features is 26 slices ~0.09
units apart on a 1.4-radius shell. Nothing reads as a region, which is the
thing being asked for. Freeing all three dimensions makes a concept one object
whose members happen to span depths, and "which layer" becomes "when it
lights" — which the shared selection already expresses.

*Alternatives:* (a) keep `z = layer`, 2D layout per slab — preserves the
existing convention and cross-layer columns are visible, but no region ever
forms; (b) hybrid, layer as a soft prior on one axis — muddies both readings
and makes the honesty copy incoherent ("this axis is partly layer"). Rejected.

*Cost:* most of `frontend-brain-view` is a rewrite, and the layer bands, rings,
crossover ring and band hover panel are deleted. That deletion takes the lens
classification off the screen with them: `lens.ts` is the shared source, but
the brain was its only renderer — the grid colours its cells by residual L2
norm, and nothing drew `crossover_layer` at all. Keeping the reading therefore
costs a renderer in the grid, not just a deletion on the brain (see task 5.5).

### 2. Two atlases: label embeddings for the default view, `W_dec` for the geometry

**This decision was reversed by measurement.** The original reasoning is kept
below because it was sound reasoning from a false premise, and the premise is
what the pilot falsified.

The original call was: positions from `W_dec` only, because a layout built from
explanation text would import Neuronpedia's explainer split (layers 16, 18, 20,
22 and 24 are `gemini-2.5-flash-lite`, the other 21 `gpt-4o-mini`) straight
into the geometry, and those five layers would form their own region for a
reason about the labeller rather than about gemma.

The pilot layers were chosen to include two gemini layers precisely so that
risk could be measured. It is not there:

| | decoder directions | explanation embeddings |
| --- | --- | --- |
| kNN preservation, 3d | 0.154 | **0.303** |
| clusters found | 2 (one holding 99.8%) | **32** |
| clusters earning a name | **0** of 84 attempted | **27 of 32** |
| coherence margin over baseline | +0.02 median | **+0.223** median, +0.429 max |
| explainer influence (AMI) | n/a | **0.026** |

An AMI of 0.026 says the clustering is essentially independent of which model
wrote the labels. The feared confound does not materialise, and the
decoder-direction layout cannot produce areas at all: it is a continuum, not an
archipelago — HDBSCAN in the full 2304-d space calls 99.1% of features noise,
and a forced k-means partition earns zero names at either k=24 or k=60.

So both are built, and the atlas records which source it came from:

- **explanation embeddings** drive the default view. It is the only layout that
  yields areas, and it yields good ones — "URLs and web links", "punctuation
  marks", "opening and closing braces and parentheses in code".
- **decoder directions** remain as a second atlas, because they answer the
  question the interface cannot otherwise answer: how the model itself groups
  what it writes. They also make the cross-source agreement measure a real
  comparison between two shipped artifacts rather than an internal diagnostic.

**What the default view must therefore say.** A label-embedding layout maps
*descriptions of* features, not the model's computation. "Near" means an
explainer described them similarly. That is a weaker claim than the original
design made and it has to be stated in the legend, not left to be inferred —
which is why the spec now requires the source to be recorded and a text-derived
atlas to be labelled as such.

The original reasoning, retained:

### 2a. Why positions were first taken from `W_dec` alone

```
   W_dec[layer][f]  (2304-d)            explanation embedding
   the model's own direction            what an LLM wrote about it
            |                                     |
            | UMAP -> 3D                          | names clusters,
            v                                     v measures coherence
      +-------------+                       +--------------+
      |  GEOMETRY   |                       |   MEANING    |
      +-------------+                       +--------------+
   pooling is legitimate:                 confound: 5 of 26 layers
   one shared residual basis              use a different explainer
```

*Why:* the layout should be a map of gemma, not of the labeller. Using
explanation embeddings for position would import the explainer split directly
into the geometry, producing five layers as a visible region for a reason
`labels.py` already documents as an artifact. Splitting the sources puts each
where it is trustworthy, and still gives the embeddings their first consumer.

*Alternative:* label embeddings for position. Tempting — clusters become
trivially nameable, and the data is a `--embeddings` flag away. Rejected on the
confound. If it is ever wanted, it belongs as a *second*, separately labelled
atlas, not as the default.

*Consequence:* a cluster of decoder-similar features is not guaranteed to be
label-coherent, which is why naming is gated on a measurement (decision 5)
rather than assumed.

### 2b. The pilot six ship; all 26 layers are a follow-on

The atlas this change ships covers layers 0, 5, 10, 16, 20 and 25 rather than
all 26. That is a scope decision, not an unfinished build.

Those six were picked to span the depth range and to straddle Neuronpedia's
explainer split, so the confound decision 2 turns on could be *measured* rather
than assumed. Having served that purpose, they turn out to be a coherent
shipping unit, because the label store and the atlas cover exactly the same
features:

```
   layers 0, 5, 10, 16, 20, 25

     label store    98,210 features   100% with embeddings
     atlas layout   98,210 features   placed
                    ^^^^^^
     the same set -- the label-source atlas places precisely
     the features that have labels, so coverage cannot diverge
```

`sae_layers` (already on `TraceRequest`, added for SAE residency: 26 resident
16k SAEs come to ~7.9GB) pins a trace to those six. Every feature such a trace
records then has a label and a position, and coverage is 100% rather than the
~23% an all-26 request would get against these stores.

*Why not build all 26 now.* The label store currently holds only the pilot six;
the other twenty need a fresh `--embeddings` import of roughly 1.5GB, after
which UMAP runs single-threaded — the price of a fixed seed, and of an artifact
whose whole value is being the same map every time — over 4.3x the points, at
an unmeasured cost. Neither is hard; both are a separate piece of work from
getting the view onto the screen, and neither changes a line of section 3 or 4,
which are indifferent to how many layers the atlas holds.

*The open question that build has to answer.* UMAP's local preservation
typically degrades as n grows at fixed `n_neighbors`, so 0.292 over 98,210
features is not a promise about 425,984. If the full atlas comes back near the
decoder source's 0.143, the honest outcome is a smaller, well-measured atlas
rather than a larger vague one — which is why the gate is re-run rather than
assumed to pass.

*Consequence:* the interface has to state its own scope — six of twenty-six
layers — in the same register as `attn_topk_coverage` and the shown-versus-fired
count. Task 4.10 already requires naming the layers without data and refusing to
render a missing layer as one in which nothing fired, so this needs no new
requirement, only that the existing one is read as covering deliberate scope and
not just missing data.

### 3. UMAP, with the reduction's cost measured

*Why UMAP over PCA:* 2304 -> 3 by PCA keeps global variance and destroys the
local neighbourhood structure that makes clusters legible; the top 3 principal
components of a residual basis are dominated by a few high-variance directions.
UMAP optimises exactly the local structure the view depends on. Over t-SNE:
UMAP scales to 426k points, supports a reusable transform, and is the standard
in this corner of the field (Neuronpedia's own feature maps, and the feature
maps in *Scaling Monosemanticity*) — precedent worth inheriting rather than
inventing.

*What UMAP costs, and how it is stated:* UMAP preserves local neighbourhoods
and does not preserve global distance. So the honest reading of the atlas is
"these features are near each other" and never "these two areas are far apart."
That is a spec requirement, not a caveat in a docstring — the UI ships with no
axes, no coordinate readout, and no distance affordance.

*The number that makes it checkable:* **k-nearest-neighbour preservation** —
for a sample of features, the fraction of their k nearest neighbours in the
2304-d decoder space still among their k nearest in 3D. This is the atlas's
peer to the diagnostics every other pass carries:

| pass | its check |
| --- | --- |
| SAE | `explained_variance` — how much of the residual the SAE recovers |
| lens | `final_layer_agreement` — must be 1.0 |
| attribution | `reconstruction_max_rel_gap` — the decomposition is exact |
| **atlas** | **kNN preservation** — how much neighbourhood structure survived |

If it comes back at 0.31, the record says 0.31 and the UI can surface 0.31.
Three dimensions from 2304 will lose a great deal; the point is that how much
is a measured number rather than an unexamined assumption.

### 4. UMAP output is warped through the existing shell function

UMAP's 3D output is an arbitrarily shaped, arbitrarily scaled blob. Normalise
it to the unit ball, then push each point through `shapeEllipsoid()` — the
same function the shell mesh and the sulcus lines already use — and the cloud
fills the rendered brain instead of floating in it.

*Alternatives:* affine-fit to the shell's bounding box (leaves corners empty and
points outside the mesh); a 2D layout with depth faked (reintroduces a
meaningless axis). Rejected.

*What is deliberately not done:* the layout is not mirrored or symmetrised. A
real brain is bilateral; this cloud crosses the midline for no reason at all,
and pretending otherwise would invent a left/right meaning. Stated in the
legend, and a spec requirement.

### 5. Clusters named from a member, gated on measured coherence

HDBSCAN over the projected positions — density-based rather than k-means,
because the number of natural groupings is not known ahead of time and because
"noise" is a legitimate answer for a feature that belongs to no cluster.

Naming: embed each member's label, take the **medoid** — the member whose label
embedding is closest to the cluster's mean — and use *that member's own label*
as the area's name. So an area is named `"bridge and crossing"` because a
specific feature says that, and the UI can disclose which one.

Gated on coherence: mean pairwise cosine of the members' label embeddings,
recorded alongside the same statistic for a randomly assembled group of equal
size. A synthesised summary over an incoherent cluster is precisely the kind of
plausible-looking invention this codebase refuses elsewhere.

**The gate is relative to the baseline, not an absolute value — and that was
learned the hard way.** The first calibration used an absolute 0.25. The pilot
build over 98,304 real features then measured a mean cluster coherence of
**0.281** against a random baseline of **0.232**, and duly named 14 clusters
whose members agreed with each other by 0.049 more than a random group of the
same size did. The names read plausibly, which is exactly what made it
dangerous. Explanation embeddings from one explainer share a similarity floor
well above zero, so an absolute threshold does not measure agreement at all —
it measures the floor. The gate is now `coherence > baseline + margin`, with
the margin recorded on the atlas, and a cluster whose baseline could not be
measured stays unnamed rather than falling back to an absolute test.

*Alternatives:* LLM-summarise each cluster (fluent, unfalsifiable, adds a third
explainer to a project already tracking two); most-frequent n-grams (brittle,
and reads as a name without being one). Rejected.

*Also recorded per cluster:* the explainer distribution of its members. A
cluster whose labels come predominantly from one explainer while the model uses
several is flagged as a possible explainer artifact — the only defence
available once text enters the pipeline at all.

### 6. Cross-source agreement as a third check

Cluster the same features a second time from their explanation embeddings and
record the agreement (adjusted Rand index) with the decoder-direction clusters.
Not a gate — the shipped positions are always the decoder ones — but if two
independent representations carve the space the same way, the areas are
describing something real. This is the same move as
`verify_neuronpedia_mapping.py`: prove the thing agrees with an independent
source rather than asserting it.

### 7. The layout lives beside the labels and travels as a side table

A `layout` table in the existing label SQLite, keyed the same way the `labels`
table is, plus a row recording the atlas's version, seed, parameters, content
hash and its three measurements.

On the wire, `Trace.layout` keyed `"layer/index"` — the same shape and the same
reasoning as `Trace.labels` (schema.py: a feature recurs across positions, so
per-occurrence copies would be the same values written out thousands of times).
The trace also records the atlas version and hash the positions came from, so a
client holding a different atlas can detect the mismatch instead of drawing a
trace against the wrong map.

*Size:* ~6,234 positions for a 31-token trace. As three floats plus a cluster
id that is a few hundred KB of JSON — noticeable next to a trace that is already
a few MB, and worth quantising if it bites.

*Alternative:* ship all 425,984 positions to the browser as a static asset
(3 x int16 = **2.56MB**, plus cluster ids). Rejected as the primary path: the
client only ever lights the ~6k features a trace actually reports, so shipping
70x that to place them is waste. But a **subsample** is still needed —

### 8. A small static subsample for the idle brain

With no trace loaded there are no active features, and a brain that is an empty
shell until you type something loses the "it has structure" read entirely. So a
~20k-node subsample of the atlas plus the area centroids and names ships as a
static asset (~120KB at int16) and renders inactive. It is decoration with a
stated provenance: a uniform sample of the same atlas, drawn dim, never lit.

### 9. The grid is already the transport

```
        tokens ->
   L0   [ ][ ][ ][ ][ ][ ]        selection (token 8, L20)
   ...  [ ][ ][ ][ ][ ][ ]              |
   L20  [ ][ ][ ][X][ ][ ]  <-----------+
   ...  [ ][ ][ ][ ][ ][ ]              |
   L25  [ ][ ][ ][ ][ ][ ]              v
                             the brain lights that cell's features

   scope:  [ cell ]      [ token, all layers ]   [ whole trace ]
              16                  416                ~6,234
```

`onSelectCell` and `onSelectPosition` already drive both surfaces and already
satisfy the shared-selection requirement. A play control just steps the layer
index. No new state model, and the existing spec requirement about a single
selection survives unchanged in substance.

Scope aggregation across positions (trace scope) needs a stated function — max
activation per feature is the leading candidate, since sum rewards features that
recur rather than features that fire hard — and the spec requires whichever is
chosen to be named in the UI.

### 10. Rendering and picking

One `THREE.Points` draw call with a custom shader: static position and
cluster-id attributes, plus a per-node `activation` attribute rewritten on
selection change. Additive blending, `depthWrite: false`, inside the existing
translucent shell, through the bloom pass already in the composer. 20k dim
plus ~6k lit is nothing for the GPU.

Picking: a k-d tree over the *active* nodes only (~6k), built on scope change.
GPU id-buffer picking would handle all 426k and is not needed, because only lit
nodes are inspectable — an unlit node has nothing to report but its label.

Areas render as centroid markers with a soft radial falloff sized by member
spread. A density isosurface would look better and is deferred; the spec
constrains what an area may *claim*, not how it is drawn.

### 11. Build-time dependencies stay build-time

`umap-learn` and the clusterer are imported by `scripts/build_feature_atlas.py`
and by nothing else. `schema.py` keeps its "imports nothing but pydantic"
property, and the service does not gain a UMAP dependency to serve a
precomputed table.

## Risks / Trade-offs

- **kNN preservation may be poor enough to undermine the view.** 2304 -> 3 is a
  brutal reduction. → It is measured and published before anything is built on
  top of it; the build script comes first in the task order for exactly this
  reason. If it is very low, the fallbacks are a higher-dimensional intermediate
  (PCA to ~50 before UMAP, standard practice), a per-layer atlas, or restricting
  the atlas to labelled features that actually fire across a set of traces.
- **Clusters may be artifacts.** → Positions never touch label text; coherence
  is measured against a shuffled baseline; explainer distribution is recorded
  per cluster; cross-source agreement is recorded. Four independent hedges, all
  reported rather than gating.
- **An area name is one member's label standing for many.** → Same lossiness as
  a blended band, handled the same way the band did it: the name discloses which
  member it came from, and the members remain inspectable.
- **The atlas invites region-thinking.** A named blob on a brain is a very
  strong invitation to read a brain region. → The prohibition is a spec
  requirement covering nodes, areas and implied bilateral meaning, plus the
  no-metric-space requirement that removes the distance affordances.
- **Top-16 of ~78 reads as "everything that fired".** → Coverage is a spec
  requirement: show 16 against `l0`, the way `attn_topk_coverage` reports what
  the attribution top-8 keeps.
- **BOS features are meaningless.** A known Gemma Scope artifact, already
  excluded from every summary statistic. → Excluded from the lit set and stated.
- **26 resident SAEs (~7.9GB) for the feature pass.** Fine on this box's unified
  pool; not fine on a 16GB discrete GPU. → The pass already accepts a layer
  subset; expose it so a smaller deployment can ask for fewer layers, and record
  which layers ran so the UI can name the gaps.
- **UMAP is stochastic.** An atlas that reshuffles between builds is worthless.
  → Fixed seed, pinned library versions, a content hash over the positions, a
  version identifier, and the version recorded on every trace drawn against it.
- **Visual legibility is unproven.** A point cloud inside a translucent
  double-sided shell with bloom may read as fog. → Prototype the render against
  a real atlas early, before the areas, LOD, filtering and transport are built
  on top of it.
- **Deleting the bands is a visible loss.** It is a real one: the brain was the
  only renderer of the `answer` / `echo` / `other` classification, and nothing
  drew `crossover_layer`. → The grid gains both, from the same `lens.ts`
  module — a colour mode over its own layer x token cells, where the reading is
  per-cell and unblurred rather than binned into seven bands.
- **`Trace.layout` grows the trace JSON** by a few hundred KB. → Quantise, or
  drop to a compact parallel-array form, if it becomes the dominant term.

## Migration Plan

1. **Wire the passes.** `TracePass` gains `"sae"` and `"labels"`; `JobPhase`
   and `_PHASE_ORDER` gain `"sae"`. Independently useful and independently
   testable — features reach the browser as data before any of this is drawn.
2. **Build the atlas and publish its numbers.** `build_feature_atlas.py`, the
   `layout` table, the three measurements. This is the gate: if kNN preservation
   is unusable, that is known here, before any rendering work.
3. **Schema 1.4.** `Trace.layout` plus the atlas reference. Additive, so a 1.3
   reader ignoring unknown fields still loads a 1.4 trace, and a 1.4 reader
   handles a 1.3 trace as one with no layout — the same additive-enrichment
   property the schema has had since 1.1.
4. **Render nodes only.** Point cloud, activation lighting, scope selector,
   per-node inspection. The bands stay during this step so the brain is never
   in a broken state.
5. **Retire the bands.** Delete the band, ring, crossover and hover-panel paths
   and the now-unused band helpers in `lens.ts`; the per-cell classification the
   grid uses stays.
6. **Areas, LOD, transport, label filter.**

**Rollback:** each step is independently revertable, and the frontend must
render the shell alone when no atlas is available (a spec requirement), so a bad
atlas is removable without a frontend deploy. The schema addition is additive
and needs no down-migration. The label DB's `layout` table can be dropped and
rebuilt in isolation, as `import_neuronpedia.py`'s output already can.

## Open Questions

- The coherence threshold for naming an area, and the target cluster
  granularity. Both are calibration against the built atlas's own numbers, and
  the specs require the threshold to be *recorded*, not to have a particular
  value.
- Whether trace-scope aggregation is max or something else. The spec requires
  the choice to be stated in the UI either way.
- Whether areas eventually render as density isosurfaces rather than centroid
  glows. Purely visual; constrains nothing in the specs.
- Whether a second, label-embedding atlas is worth shipping alongside as a
  comparison view once the first one's numbers are known.
