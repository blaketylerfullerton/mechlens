# Circuits workspace feature spec

Status: proposed feature; local backlog. All four tickets are open and are not
external issue IDs. Check a ticket off only after its acceptance criteria have
been verified.

This spec supersedes the earlier influence-explorer MVP. The product starts
from an output prediction and traces contributing computation backward. Feature
interventions support validation within that experience.

See the [circuit tracing plan](circuit-tracing-plan.md) for technical sequencing,
research references, and the prototype decision gate. CIR-01 is complete; its
measurements and the go decision are in the
[CIR-01 report](circuits-cir01-report.md).

## Objective

Let a user generate an answer, select an output token, choose **Explain this
prediction**, and explore a readable attribution graph connecting input tokens,
internal features, and that prediction. Users can inspect evidence for feature
interpretations, follow branches backward, test selected hypotheses, and save
their work.

This is a partial, approximate explanation of computation for a particular
prediction. It is not a transcript of thoughts or a complete account of an
answer. The interface must make unexplained computation and approximation
limits accessible alongside the graph.

## First release scope

- One verified Gemma 2 2B model/transcoder combination, selected by prototype.
- Short, unsteered prompts and one selected output token per analysis.
- Exact-token replay and output-targeted attribution using `circuit-tracer`.
- A focused graph with input tokens, transcoder features, output, and error or
  unexplained-computation nodes where the method provides them.
- Backward branch exploration, feature evidence, annotations, and grouping.
- Saved graphs, immutable source snapshots, background jobs, and bounded compute.
- A supported intervention workflow for testing selected graph hypotheses.

Start with compatible pretrained transcoders. Training new transcoders, arbitrary
model support, whole-answer tracing, live tracing during generation, automated
circuit discovery across prompts, and complete attention-mechanism explanations
are outside this release. Do not enable steered-trace analysis until the original
interventions can be replayed exactly.

The existing residual-SAE Explore view remains useful. Transcoder features have
their own identities, hooks, labels, and evidence; they cannot inherit residual
SAE feature IDs, labels, or suppression operations by matching layer/index.

## User workflow

1. Generate a completed answer in Explore using the supported model.
2. Select an output token and choose **Explain this prediction**.
3. Circuits shows the selected target, exact explanatory prefix, tracing method,
   supported model/transcoder set, and bounded analysis settings.
4. Start analysis and see real queue/execution progress. Once saved, the graph
   opens focused on the selected output and its strongest retained contributors.
5. Select a feature to follow incoming branches, inspect its token location and
   activation examples, and distinguish proposed labels from measured quantities.
6. Pin or group features and annotate a possible explanation. Expand retained
   graph detail without silently scheduling additional computation.
7. Choose **Test this feature** for a supported intervention. Compare the chosen
   output before/after and inspect whether the result supports the hypothesis.
8. Save and reopen the graph, annotations, and linked intervention results.
   Return to the original trace and exact token location when needed.

For generated token `j`, the explanatory prefix is `token_ids[:j]` and the target
is `token_ids[j]`, including BOS in position indexing. Never include the answer
in its own prefix. Selecting another output token creates or loads an analysis
for that token's prefix; it does not relabel the current graph.

## Measurement and presentation contract

### Attribution graphs

Use the upstream tracing method behind a small backend adapter. Pin library
version, model/revision, tokenizer settings, runtime/dtype, transcoder artifact
identities, exact input IDs, target token, and graph construction/pruning settings.
Record unavailable provenance as unknown and do not claim exact historical
reproduction when source weights cannot be identified.

Edges represent contributions under the selected attribution method, not
original-model intervention results. Document the method's treatment of
attention, nonlinearities, reconstruction error, and pruning. Preserve available
quality metrics with their definitions; never rename one as a universal
"percent explained" or a probability that an explanation is true.

Pruning, computation limits, and display limits are separate. Retain coverage
metadata and error contributions. Omitted edges/features are not zero effects.
Do not silently hide unexplained computation to make a graph appear complete.

### Feature evidence and readable paths

Feature occurrences include transcoder artifact, feature index, read/write sites
as appropriate, and token position. Cross-layer features may write at multiple
layers and must not be reduced to an inaccurate single-layer connection.

Labels and examples must belong to the same transcoder artifact. If unavailable,
show a feature ID and an explicit unlabeled/no-examples state. Fetching external
feature data is explicit in the backend configuration and provenance.

Group labels and explanatory annotations are interpretations. Preserve their
authorship and links to underlying nodes. A highlighted path is a selected part
of the graph, not proof that the model executed a single sequential thought.
A prose-summary generator is optional follow-up work, not a release dependency.

### Intervention validation

Reuse exact-token replay, scoring, controls, and job infrastructure from the
existing experiment engine where appropriate. Implement a transcoder-aware
intervention adapter for the selected tracing components. A residual-SAE decoder
suppression cannot be applied to a transcoder node without a validated mapping.

Record the intervention site/operation and whether execution modifies the
replacement model or the original model. Replacement-model tests must not be
presented as original-model validation. CIR-01 must establish a supported
original-model test for selected hypotheses before the product release proceeds.

Use identical input IDs for baseline, zero-strength control, and intervention;
compare output probability and log probability. Establish dtype-aware tolerances,
preserve failed-control diagnostics, and block interpretation after failed
controls. Record actual activation changes where supported and describe exactly
what the operation changes. A successful feature test does not validate every
edge on a displayed path or prove mediation.

## Architecture and persistence

Add versioned graph-analysis artifacts, separate from source trace edges and
linked intervention records. Store an immutable completed source trace snapshot,
exact tokens/target, provenance, method settings, coverage/quality information,
retained graph data, and lifecycle/error state. Store user annotations/groupings
separately from immutable measurement data.

Service jobs currently live in memory and the browser retains its last trace in
IndexedDB. Support registering a validated browser-restored snapshot when its
original service job no longer exists. A source trace ID alone is insufficient
for restart-safe inspection.

Proposed service surface, finalized after the prototype adapter is understood:

- `POST /circuits/analyses`: validate a completed snapshot and enqueue attribution.
- `GET /circuits/analyses`: paginated saved-analysis summaries.
- `GET /circuits/analyses/{id}`: lifecycle, progress, provenance, and graph summary.
- `GET /circuits/analyses/{id}/graph`: bounded retained graph detail.
- `POST /circuits/analyses/{id}/cancel`: request cooperative cancellation.
- `PUT /circuits/analyses/{id}/annotations`: save annotations/groupings.
- `POST /circuits/analyses/{id}/experiments`: enqueue a supported validation test.
- `GET /circuits/experiments/{id}` and `POST /circuits/experiments/{id}/cancel`:
  inspect/cancel a linked test.

Use atomic persistence and bounded reads. On restart mark unfinished analyses
and experiments interrupted. Completed graphs, source snapshots, and saved
validation results remain viewable without loading weights or contacting a
remote feature service.

Coordinate tracing with inference, training, steering, and model switching under
the existing compute-ownership rules. Do not assume the attribution runtime can
share the currently loaded model instance safely. Serialize or unload/load as
required by the measured memory budget. Expose honest progress and cancellation
boundaries; if upstream tracing cannot stop promptly, make that limitation clear.

## Implementation tickets

### CIR-01 — Prove output-targeted tracing on the target hardware

- [x] Complete CIR-01. Go — see the [CIR-01 report](circuits-cir01-report.md).

Dependencies: none. This is the decision gate for the remaining implementation.

Build an isolated prototype using `circuit-tracer`, Gemma 2 2B, and a compatible
pretrained transcoder set. Use the upstream graph viewer initially. Test exact
prefix/target handling, feature data, graph quality, and intervention semantics
before committing the main application to a dependency or artifact format.

Acceptance criteria:

- Record pinned library/model/transcoder versions, licenses/access requirements,
  dependency compatibility, artifact sizes, and reproduction commands.
- Generate saved graphs for several short prompts with explicitly chosen output
  targets, including at least one generated token from a mechlens trace.
- Verify exact token IDs including BOS and exclusion of the answer from its
  prefix. Establish baseline agreement with the corresponding original-model
  forward pass under documented settings/tolerances.
- Inspect graphs and matching feature evidence in the upstream viewer. Document
  concrete interpretable examples and an example that is incomplete or unhelpful.
- Establish how selected transcoder hypotheses can be tested on the original
  model; exercise baseline/zero control/intervention and distinguish replacement
  model behavior. Document unsupported intervention types.
- Measure runtime and peak memory on the target hardware for multiple short
  prefix lengths and graph budgets, including model/transcoder loading costs.
- Document attention treatment, error/quality metrics, pruning, progress hooks,
  cancellation boundaries, and serialization/export formats from the actual version.
- Write a go/no-go report. Proceed only if graphs are useful, identity/replay and
  original-model validation are sound, and resource costs are practical. A failed
  gate triggers a scope decision, not silent substitution of an influence graph
  or unplanned transcoder training.

### CIR-02 — Durable tracing and validation backend

- [ ] Complete CIR-02.

Dependencies: CIR-01 go decision and adapter contract, both in the
[CIR-01 report](circuits-cir01-report.md). Start with its §9: one environment,
inline `apply_softcap`, add dependency ceilings, pin `circuit-tracer`.

Integrate the validated tracing adapter, source snapshots, graph storage,
annotations, job execution, and linked intervention tests. Reuse existing
measurement primitives where their semantics match the selected components.

Acceptance criteria:

- Requests validate supported identities, completed snapshots, exact tokens,
  eligible output targets, context limits, and bounded graph settings before compute.
- Graph records preserve input/feature/output/error node identities, edge method,
  quality/coverage information, pruning settings, and complete available provenance.
- Browser-restored traces can be snapshotted; stale/incompatible identities and
  unsupported steered/local-dictionary requests fail explicitly.
- Tracing and validation serialize with conflicting inference/training/model
  switches. Progress and cancellation reflect the actual upstream capabilities.
- Queued/running cancellation, failures, hook/resource cleanup, atomic writes,
  interrupted-job recovery, and model switching have relevant automated coverage.
- Saved graphs and linked validation results survive restart and load without
  weights. Bounded graph reads support expansion without a new attribution run.
- Annotation edits cannot alter measured graph data; linked validation records
  identify their intervention semantics and retain failed-control diagnostics.
- Exact replay and scoring have independent checks; relevant existing trace,
  steering, experiment, and training coordination tests continue to pass.

### CIR-03 — Explain-this-prediction workspace

- [ ] Complete CIR-03.

Dependencies: CIR-01 sample graphs and interaction findings (see the
[CIR-01 report](circuits-cir01-report.md) §10 for the viewer recommendation and
the constraints the workspace inherits); CIR-02 API contract and execution for
end-to-end completion.

Add output-token entry points and a Circuits page focused on backward exploration.
Reuse React Flow or adapt the upstream viewer after assessing fit in the
prototype. Add source/target controls, graph navigation, an evidence inspector,
annotations/grouping, saved analyses, and linked intervention results.

Acceptance criteria:

- A user selects a generated token, runs **Explain this prediction**, and sees
  an actual saved attribution graph rooted in that prediction.
- Incoming-branch navigation, pinning, grouping, and bounded expansion expose
  retained graph data; expanding outside computed coverage requires an explicit run.
- Feature evidence uses correct transcoder identity. Unlabeled features, missing
  examples, error nodes, approximation limits, and quality metrics remain visible.
- Layer/token layout handles cross-layer read/write semantics correctly and
  does not imply one literal chronological chain of thoughts.
- **Test this feature** exposes only validated operations and shows baseline,
  changed output, controls, and execution model alongside the relevant graph node.
- App-owned navigation preserves analysis/source context across tabs and rejects
  stale results when traces or targets change. Returning to Explore restores the
  source token location without pretending a transcoder feature is a residual SAE feature.
- Saved analyses, annotations, source snapshots, and test results reopen after
  restart without compute. Unsupported, queued, running, cancelled, interrupted,
  failed, failed-control, empty, and complete states are clear.
- Keyboard navigation and an accessible table/list expose graph measurements
  and feature evidence. Long tokens and dense graphs remain usable.
- Frontend build/lint and a browser walkthrough against the real backend pass.

### CIR-04 — Validate explanation quality and release readiness

- [ ] Complete CIR-04.

Dependencies: CIR-01, CIR-02, and CIR-03.

Evaluate the integrated experience on the supported hardware and publish a
reproducible validation report with saved analyses and linked experiments. Fix
correctness/usability issues before releasing the supported path.

Acceptance criteria:

- Complete the real workflow: generate, select output, trace, follow branches,
  inspect evidence, test a hypothesis, annotate, reopen, and return to the source.
- Evaluate several prompts, related prompt variations, repeated runs, and
  unrelated controls. Include failures/unhelpful graphs rather than reporting
  only compelling examples or treating labels as verified explanations.
- Report whether selected original-model interventions support graph-derived
  hypotheses, preserving disagreements and avoiding claims that individual tests
  validate a complete path. Record tested scope and selection criteria.
- Validate numerical tolerances and report available graph fidelity/coverage
  metrics with definitions and method limitations.
- Measure end-to-end latency, peak memory, storage, and usability over the supported
  prefix/graph budgets; choose defaults and document limits from those measurements.
- Exercise cancellation, model switching, restart recovery, and offline viewing
  of saved results. Confirm no stale trace/feature identity leaks into the UI.
- Relevant automated checks and the real-model walkthrough pass with no unresolved
  correctness issue in the supported release path. Document remaining research limits.

## Delivery boundary

CIR-01 is the next step, not a commitment to implement the entire UI before
feasibility is known. On a go decision, implement CIR-02 and CIR-03, then complete
CIR-04 before release. Integrate with ongoing navigation/training/job changes.

The release delivers a useful, evidence-backed **why this token?** explorer for
one supported setup. It does not promise complete access to model thoughts or
reliable explanations for every prompt. Estimate delivery time after CIR-01
establishes compatibility, graph usefulness, and measured compute costs.
