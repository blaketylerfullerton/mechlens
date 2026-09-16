# Circuit tracing implementation plan

Status: proposed; revised to target output-centered attribution tracing.
No runtime behavior is added by this document.

The [Circuits feature spec](circuits-feature-spec.md) is the product contract and
contains the four local implementation tickets with acceptance criteria. This
plan supersedes the earlier suppression-first influence-explorer delivery order.

## Product decision

Build **Explain this prediction**: select a generated output token in Explore,
open Circuits, and follow an attribution graph backward through contributing
features to input tokens. Make feature evidence, unexplained computation, and
selected intervention tests available in the same workspace.

The user wants to inspect how an answer was produced. An interface requiring
manual source-feature selection and showing only its downstream suppression
effects does not fulfill that primary workflow. Keep intervention measurement
as validation infrastructure, while moving attribution feasibility to the start.

A graph is a partial approximation of computation for a fixed prefix and target.
Neither a highlighted path nor a readable group label is a literal transcript
of thoughts. Do not promise a complete explanation of arbitrary conversations.

## Available foundation and first decision

The upstream [`circuit-tracer` project](https://github.com/decoderesearch/circuit-tracer)
provides attribution graph generation, visualization, and intervention tools.
Its documentation lists pretrained per-layer and cross-layer transcoders for
Gemma 2 2B. This makes integration a concrete first option; it does not establish
compatibility or acceptable performance in mechlens without a prototype.

CIR-01 must pin the actual library and artifact revisions, verify access/licensing,
and run the upstream viewer on the target hardware before we commit to a custom
viewer or dependency architecture. Begin with one compatible Gemma 2 2B setup.
Do not train new transcoders as an implicit fallback.

The existing Explore dictionaries encode residual-stream activations. Transcoders
approximate model components and define a different feature space. Their feature
IDs, read/write hooks, labels, and intervention semantics need distinct identity.
A cross-layer feature can contribute at multiple write sites. No layer/index
join can safely identify it with an existing residual SAE feature.

## Existing code to reuse

- `frontend/src/App.tsx`: workspace navigation and trace-tagged selection. Add
  output-target and analysis context, preserving source snapshots across tabs.
- `frontend/src/components/TraceViewer.tsx` and related token controls: entry to
  **Explain this prediction** and return navigation to the original token.
- `frontend/package.json`: React Flow is installed. Decide whether to use it or
  adapt the upstream viewer after inspecting actual prototype graphs.
- `backend/app/experiments.py`: reuse fixed-prefix scoring, controls, provenance,
  and hook-cleanup patterns. Refactor exact-token support as needed. Its existing
  residual-SAE suppression operation is not automatically a transcoder intervention.
- `backend/app/service/jobs.py` and `scheduling.py`: reuse compute coordination,
  extending lifecycle/progress for attribution and validation without breaking
  inference/training/model-switch exclusion.
- `frontend/src/lib/trace-storage.ts`: current browser trace persistence explains
  why source snapshots must be registered and saved on the backend for durable analyses.
- `backend/tests/test_experiments.py`: extend independent-forward and cleanup
  checks to the validated replay/intervention adapters.

Do not reuse current trace attribution edges by renaming them: their component
contribution magnitudes are not the new feature attribution graphs.

## Technical sequence

### CIR-01 — Prototype and go/no-go

Generate real graphs for several short prompts and selected output tokens using
the upstream runtime and viewer. Verify exact token IDs, feature evidence,
original-model comparison, and the chosen intervention mechanism. Benchmark
latency, loading cost, peak memory, and graph budgets on the target hardware.

For generated token `j`, replay `token_ids[:j]` and target `token_ids[j]`.
Verify BOS and tokenizer behavior explicitly; do not re-tokenize decoded trace
text or include the target token in its own explanatory prefix.

Inspect the pinned method's attention assumptions, replacement errors, pruning,
quality metrics, serialization format, progress, and cancellation boundaries.
Compare graph-derived hypotheses with supported original-model interventions;
replacement-model intervention success alone is insufficient validation.

Proceed only when the prototype demonstrates useful graphs, reproducible input
handling, sound intervention semantics, and practical compute costs. If it fails,
document the specific limitation and revisit supported scope before building the
rest. Do not silently substitute an influence graph for the requested experience.

### CIR-02 — Durable backend

Wrap the validated engine behind a narrow adapter. Persist immutable source
snapshots, graph/provenance/coverage artifacts, lifecycle state, and linked
validation experiments. Keep annotations separate from measured data.

Add the service surface defined in the spec, bounded graph retrieval, atomic
writes, cancellation, and restart reconciliation. A browser-restored completed
trace must remain usable after the original in-memory job disappears.

Resolve resource ownership explicitly: a replacement-model runtime may require
a separate model instance or sequential unloading/loading. Benchmark rather than
assuming it can share the inference instance or fit beside training resources.
Report actual attribution/encoding/validation progress instead of forcing these
operations into the existing monotonic trace phases.

### CIR-03 — Output-centered workspace

Build the path from selected answer token to focused attribution graph. Let
users explore incoming branches, inspect matching evidence, pin/group/annotate,
and test supported features. Keep source/target context in App-owned state.

Separate computed coverage from pruning and initial display limits. Expanding
retained graph detail reads saved data; additional attribution requires an explicit
run. Preserve error contributions and method limits. Provide an accessible
measurement list/table and explicit unavailable-evidence states.

Feature interpretations should be readable but traceable to underlying nodes
and evidence. Automated prose storytelling is not required for the first release.

### CIR-04 — Integrated validation and release

Run the complete application workflow on the supported hardware. Save examples
with related/unrelated prompts, repeats, informative and unhelpful graphs, and
original-model intervention comparisons. Record disagreements and unknowns.

Set prefix/graph/resource defaults using measured costs. Verify restart,
cancellation, model switching, offline result viewing, and exact source navigation.
Complete relevant tests and UI checks; do not substitute software checks for
real-model evidence. The spec defines the full acceptance criteria.

## Measurement guardrails

- Attribution edges and experimental total-effect measurements retain distinct
  method labels and provenance. Neither implies a uniquely responsible cause.
- Source trace IDs, model names, and layer/index pairs are insufficient identity.
  Pin available revisions/artifact fingerprints and mark unavailable data unknown.
- Feature evidence must match the transcoder artifact. Residual-SAE labels and
  decoder directions are not interchangeable with transcoder data.
- Control and intervention runs share exact tokens. Failed zero controls remain
  saved and visible, and block interpretation of the linked test.
- Record probability/log-probability changes separately from feature activation
  or attribution units. Do not call attribution weight a probability of correctness.
- An intervention tests its specified operation, not every graph edge or an
  entire visual chain. Claim mediation only with a separately validated method.
- Preserve reconstruction/error contributions and coverage limits; omitted data
  is not zero. Explain quality metrics using their actual upstream definitions.
- Saved inspection needs no loaded weights. New tracing and intervention work
  requires the validated model/artifact combination and explicit compute jobs.

## Scope and estimates

Ship one supported short-prompt, single-output-token workflow first. Whole-answer
analysis, arbitrary models, steered replay, broad automatic searches, richer
attention tracing, and new transcoder training require separate scope decisions.

Estimate calendar time after CIR-01. The main uncertainties are dependency and
artifact compatibility, target-hardware performance, feature evidence quality,
and whether the resulting graphs explain the behaviors the user wants to study.
Integrate with ongoing uncommitted navigation, training, and job changes.

## Research references

- [`circuit-tracer` documentation](https://github.com/decoderesearch/circuit-tracer):
  runtime, available pretrained transcoders, graph export/viewer, and interventions.
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html):
  replacement components, attribution graphs, validation, and methodological limits.
- [On the Biology of a Large Language Model](https://www.transformer-circuits.pub/2025/attribution-graphs/biology.html):
  examples and limitations of interpreting individual predictions.

Evaluate the actual pinned implementation rather than assuming every newer
research method is integrated into its runtime.
