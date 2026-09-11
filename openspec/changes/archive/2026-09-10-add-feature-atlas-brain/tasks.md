## 1. Get features onto the wire

- [x] 1.1 Extend `TracePass` in `backend/app/service/models.py` to accept `"sae"` and `"labels"`, and verify a request naming an unknown pass is still rejected with 422 before a job is enqueued
- [x] 1.2 Add `"sae"` to `JobPhase` and `_PHASE_ORDER` in `backend/app/service/jobs.py`, and verify a test asserts a reading from an earlier phase is rejected once the job has moved on
- [x] 1.3 Run the SAE pass in the trace job in `backend/app/service/app.py`, reporting per-layer progress, and verify a trace requested with `"sae"` comes back with non-empty `features` and `l0` on every requested layer
- [x] 1.4 Run the label pass when requested, and verify a trace requested with `"sae"` plus `"labels"` comes back with `labels` populated for the features it reports
- [x] 1.5 Reject `"labels"` without `"sae"` with a 422 naming the missing dependency, and verify no job is enqueued
- [x] 1.6 Fail the job with a named error when the label pass is requested and no label store is available, and verify the job reports failure rather than returning a trace whose features silently lack labels
- [x] 1.7 Expose the SAE pass's layer subset through the request so a smaller deployment can ask for fewer than 26 layers, record which layers ran on the pass record, and verify a partial request reports only those layers
- [x] 1.8 Mirror the `TracePass` and `JobPhase` changes in `frontend/src/lib/api-types.ts` and verify the frontend typechecks against a trace carrying features

## 2. Build the atlas and publish its numbers

- [x] 2.1 Add `umap-learn` and the density clusterer to `backend/requirements.txt` as build-time dependencies, and verify neither is importable from `app/schema.py` or the service module graph
- [x] 2.2 Add a `layout` table and an atlas-record table to `backend/app/labels.py` beside `labels`, keyed the same way, and verify a round-trip write/read of positions and cluster ids
- [x] 2.3 Write `backend/scripts/build_feature_atlas.py`: load the requested layers' `W_dec`, PCA to an intermediate dimensionality, UMAP to 3D under a fixed seed, and verify it writes a position for every (layer, feature) pair in the requested set — for the POC that set is the pilot six (0, 5, 10, 16, 20, 25), 98,304 pairs from the decoder source and 98,210 from the label source. The script defaults to all 26 layers and the full 425,984 pairs remain its intended scope; building that atlas is deferred to a follow-on change (see 2.12)
- [x] 2.4 Normalise the projection to the unit ball and warp it through the same shell shaping the renderer uses, and verify every position lands inside the rendered shell's bounds
- [x] 2.5 Measure and record kNN preservation against the 2304-d decoder space, with the neighbourhood size and sample count, and verify the recorded figure is reproduced by an independent check over a fresh sample
- [x] 2.6 Record the atlas version, seed, parameters and a content hash over the positions, and verify a rebuild with the same inputs reproduces the hash while a changed seed produces a different version
- [x] 2.7 Cluster the projected positions and verify every feature is assigned exactly one cluster or explicitly marked as belonging to none
- [x] 2.8 Name each cluster from its medoid member's own label, gated on measured label-embedding coherence against a shuffled baseline of equal size, and verify clusters below the threshold are recorded unnamed with their measured coherence
- [x] 2.9 Record each cluster's explainer distribution, and verify a cluster whose members' labels come predominantly from one explainer is identifiable from the record
- [x] 2.10 Cluster the same features from their explanation embeddings and record the agreement with the decoder-direction clusters, and verify a low agreement is recorded rather than causing the build to substitute the embedding clustering
- [x] 2.11 Skip naming and cross-source agreement when the label store holds no embeddings, and verify the atlas still builds with every cluster unnamed and the record states naming was not attempted
- [x] 2.12 **Gate:** build the real atlas, publish kNN preservation, per-cluster coherence and cross-source agreement in the change notes, and confirm with the user that the numbers support building the view before starting section 4

  **Resolved: the pilot six ship as the POC's scope.** The numbers support the
  view, on the label source: kNN preservation **0.292** (k=20, 4,000 sampled,
  against **0.0002** for a random layout of the same 98,210 features), **32**
  clusters with **27** earning a name at coherence **0.465** against a shuffled
  baseline of **0.233** (margin 0.232, gate 0.150), and explainer influence
  (AMI) **0.026**. The decoder source is built and kept, and correctly earns
  **zero** names from its 12 clusters — a continuum, not an archipelago.

  Scope is the pilot six (0, 5, 10, 16, 20, 25), not all 26, and that is a
  deliberate decision rather than an unfinished one — see design decision 2b.
  All 26 layers remain the goal; the import and rebuild that get there are a
  follow-on change.

## 3. Carry the layout to the client

> **POC scope.** The atlas covers the pilot six layers (0, 5, 10, 16, 20, 25),
> and so does the label store — both hold exactly 98,210 labelled features, the
> same set. A trace that pins `sae_layers` to those six therefore gets a label
> and a position for every feature it records, and the unplaced and unlabelled
> paths stay the rare cases they were designed to be. Tasks 3.3, 3.4, 3.6 and
> 4.4 are still built: they are correctness requirements, not a consequence of
> the pilot's size, and they are what a 26-layer trace against a 6-layer atlas
> will need. See design decision 2b.

- [x] 3.1 Bump the trace schema to 1.4 in `backend/app/schema.py` with `Trace.layout` keyed `"layer/index"` and an atlas reference recording version and content hash, and verify a 1.3 trace still loads as one with no layout
- [x] 3.2 Attach positions for the features a completed trace reports, and verify a trace with features and an available atlas carries a position for each reported pair plus the atlas identity
- [x] 3.3 Return features without positions when the atlas has no entry for them, and verify the response distinguishes an unplaced feature from one with a position
- [x] 3.4 Record the absence of an atlas rather than emitting zeroed positions, and verify a trace built with no atlas present carries features, no positions, and a stated absence
- [x] 3.5 Add an endpoint serving atlas positions, cluster assignments, recorded names and the atlas record without a trace and without loading the model, and verify it responds with the model unloaded
- [x] 3.6 Report the atlas's absence from that endpoint rather than an empty layout, and verify the two cases are distinguishable by a client
- [x] 3.7 Generate the idle subsample asset — a uniform sample of atlas nodes plus area centroids and names — and verify its size and that it is a strict subset of the built atlas
- [x] 3.8 Mirror schema 1.4 and the atlas endpoint's shapes in `frontend/src/lib/api-types.ts`, and verify the frontend typechecks

## 4. Render nodes

- [x] 4.1 Load the idle subsample and render it as an inactive point cloud in a single draw call inside the existing shell, and verify the brain shows structure on first load with no trace requested
- [x] 4.2 Render the shell alone with a stated reason when no atlas is available, and verify no node is placed at an arbitrary position
- [x] 4.3 Light nodes from the trace's recorded activations via a per-node activation attribute, and verify two features with different recorded activations render at different prominence while a dim node beside a bright cluster stays dim
- [x] 4.4 Render a feature the atlas cannot place as an unplaced entry with its activation, and verify no position is invented for it
- [x] 4.5 Detect and state an atlas version mismatch between the trace and the loaded atlas, and verify the positions are not presented as corresponding to that trace
- [x] 4.6 Add the scope selector for cell / token / whole-trace, stating the scope, its feature count and the aggregation used for trace scope, and verify each scope lights the set the trace records for it
- [x] 4.7 Exclude the first sequence position from the lit set with a stated reason, and verify no BOS feature is ever lit at any scope
- [x] 4.8 State the shown-versus-fired count from `l0` alongside the lit nodes, and verify the figure matches the trace's own `l0` for the displayed layers
- [x] 4.9 Build a k-d tree over the active nodes on scope change and add per-node inspection showing layer, index, activation, label if present, and the Neuronpedia link, and verify a feature with no explanation shows the absence rather than an empty label
- [x] 4.10 Render every node inactive with a stated reason when the trace has no features, name the layers without data when coverage is partial, and verify a missing layer is not shown as a layer in which nothing fired

## 5. Retire the layer bands

- [x] 5.1 Remove the band painting, band rings, crossover ring and band hover panel from `frontend/src/components/Brain.tsx`, and verify the brain renders with the point cloud alone and no band artifacts remain on screen
- [x] 5.2 Remove the band helpers left without a consumer from `frontend/src/lib/lens.ts`, keeping the per-cell classification the grid uses, and verify the grid still renders the lens classification and the crossover marker unchanged
- [x] 5.3 Rewrite the brain legend for nodes and areas, and verify it states that a node is an SAE feature, that an area is a cluster of similar residual-stream directions, and that neither is a brain region
- [x] 5.4 Confirm no axis, coordinate readout, distance scale or measurement affordance is rendered, that the layout's local-versus-global limits are stated, and that the atlas's kNN preservation figure is reachable from the interface
- [x] 5.5 Give the lens classification the renderer the migration assumed it had: colour the trace grid's cells by `answer` / `echo` / `other` under a stated colour mode, mark `crossover_layer` on the layer axis, and verify both are on screen and that the mode is unavailable with a stated reason on a trace with no lens readouts

  **Why this was not in the plan.** 5.2 and design decision 1 both justified
  deleting the bands with "the grid already renders it". It did not — the grid
  colours cells by residual L2 norm, and nothing anywhere drew `crossover_layer`.
  Retiring the bands would therefore have dropped the `answer`/`echo`/`other`
  reading from the interface entirely rather than moving it. Building the
  renderer makes the migration true; the spec's REMOVED-requirement note still
  says the grid is "unchanged", which it no longer is.

## 6. Areas, detail levels, transport, filtering

- [x] 6.1 Render atlas clusters as areas labelled only with the names the atlas recorded, and verify an unnamed cluster renders unnamed and is not summarised in the interface
- [x] 6.2 Derive area prominence from member activations at the displayed scope, stating the combination used and how many of the area's features were active, and verify an area with no active members is not lit
- [x] 6.3 Disclose the source of a named area's name — the member feature its label came from and the coherence measured — and verify it is reachable from the area
- [x] 6.4 Add level-of-detail so areas read without individual identities and nodes resolve on approach, and verify the overview is legible without focusing any single node
- [x] 6.5 Drive the lit set from the shared (layer, token) selection so a grid cell click lights that cell's features, and verify selecting from either surface updates both and a new trace resets both to one default
- [x] 6.6 Add a transport control stepping the selected layer, and verify it stops at the last layer without wrapping and is inert with a stated reason on a trace with no feature data
- [ ] 6.7 Light features per layer from live job progress without lighting a layer the service has not reported, and verify the lit set never runs ahead of the last reading and a layer whose features have not arrived is shown as computing rather than lit

  **Blocked on the service, not the view.** The second half is built and
  verified: while a layer-counting phase runs the brain states which layer is
  being computed, lights nothing from it, and never advances past the last
  reading. The first half cannot be built against the current API — a job's
  features reach the client only with the finished trace (`JobStatusResponse.trace`
  is null until `done`), so there is nothing to light per layer while the pass
  is running. Doing it needs a partial-feature channel: progress carrying each
  layer's features as it finishes them, or a readable partial trace. That is an
  api-service change and a spec delta there, not a frontend task.
- [x] 6.8 Add label-text filtering of the lit nodes, stating matched-out-of-active, and verify a query with no matches says so and that unlabelled features are excluded with that stated

## 7. Verification

- [x] 7.1 Run `pytest backend/tests -q` and verify the suite passes with new coverage for the two passes, the `sae` phase ordering, schema 1.4 round-tripping, layout attachment, and the atlas record's three measurements
- [x] 7.2 Verify the atlas build is reproducible end to end: rebuild from the same inputs and seed, and confirm identical positions, identical cluster assignments and a matching content hash
- [x] 7.3 Trace a prompt with `sae`, `labels` and `lens`, and verify in the running app that features light on the brain, areas glow, the transport walks depth, label filtering narrows the lit set, and the grid still shows the lens classification
- [x] 7.4 Verify each degraded path in the running app: no atlas, no features, partial layer coverage, features without labels, and an atlas version mismatch each state what is missing rather than rendering an empty or misleading view
- [x] 7.5 Update `README.md` — the layout table, the phase table, the measured-numbers table with the atlas's diagnostics, and the "two known gaps" note now that the explanation embeddings have a consumer
- [x] 7.6 Run `openspec validate add-feature-atlas-brain --strict` and confirm the change is valid before archiving
