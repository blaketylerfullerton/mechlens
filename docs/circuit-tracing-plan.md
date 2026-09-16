# Circuits workspace implementation plan

Status: proposed implementation plan, September 15, 2026. This document adds no runtime behavior.

## Product decision

Add a **Circuits** workspace tab beside Explore, Training, and Dictionaries. Explore remains the activity overview; Circuits provides room for a readable 2D graph, experiment controls, and numerical comparisons. Connect them with **Explore influence** on an active feature and **Inspect in Explore** on a graph node. Carry the trace, dictionary, token position, layer, and feature selection between views.

The first release is an **influence explorer** inside Circuits. It measures what changes when a feature's decoded contribution is reduced. Later releases add searches for inputs to a selected feature and, if suitable model components are available, attribution-based circuit tracing. Label the graph's measurement method prominently so the name of the workspace does not imply a complete explanation.

### First useful experience

1. Generate a trace in Explore and select an active feature.
2. Choose Explore influence. Circuits opens with that exact feature occurrence selected.
3. Choose the output token to measure and a bounded range of downstream layers/positions. Default to the next-token prediction after the selected position, with a visible option to extend the prefix to a later prediction.
4. Run baseline, zero-strength control, and feature suppression on identical input token IDs.
5. See later features that increased/decreased and the change in the selected output token's probability.
6. Click a result for baseline/changed values, intervention details, and its location in Explore. Save and reopen the experiment.

For a generated token at position j, explaining that token means replaying the prefix ending at j−1 and targeting token_ids[j]. Do not include the answer token in its own explanatory prefix. Features at j or later cannot explain that prediction. Show this explicitly in selection behavior; explaining an earlier output requires selecting an eligible earlier feature.

## Existing foundations

- `frontend/src/App.tsx`: workspace navigation and trace-tagged layer/token selection; add a Circuits destination and feature-level navigation context.
- `frontend/src/components/TraceViewer.tsx` and `Brain.tsx`: entry points and location highlighting.
- `frontend/package.json`: React Flow (`@xyflow/react`) is already installed. Reuse it for a read-only graph with custom measurement nodes; inspect the existing canvas wrapper before reuse because its deletion controls are inappropriate for measured data.
- `backend/app/experiments.py`: `measure_feature` already performs fixed-prefix baseline, zero control, decoded-contribution suppression, and steering, with output scores and provenance. Extend a shared measurement core rather than implement a second intervention engine.
- `backend/tests/test_experiments.py`: tiny-transformer tests already cover direct-forward agreement and hook cleanup.
- `backend/app/service/jobs.py` and `scheduling.py`: existing single-worker jobs and model-selection serialization. Extend these for experiments; preserve exclusion with inference/training and model switches.
- `docs/measurement-plan.md`: existing measurement semantics and limitations.

The experiment function currently tokenizes a prompt and measures output only. It needs exact-token replay, downstream capture, explicit dictionary identity, persisted service results, and a UI. Current attribution edges are component contribution magnitudes; they cannot become feature-to-feature causal arrows by renaming them.

## Measurement contract

### Reproducible inputs

Each experiment pins a completed source trace snapshot, exact prefix IDs (including BOS), output position/target token ID, model identity, hook, dictionary identity, source feature occurrence, and measurement settings. Use token IDs from the trace instead of re-tokenizing its text. Validate context length, positions, vocabulary IDs, dimensions, model/dictionary compatibility, and finite intervention strengths before expensive work.

Initially support unsteered Gemma Scope traces using supported loaded model/dictionary combinations. Disable running on steered traces until their interventions can be replayed exactly. Local dictionaries remain inspectable in Explore; enable experiments for them only when their artifact identity and compatible hooks are resolved, with explicit missing-layer messages. Never silently substitute Gemma Scope for a local dictionary.

### Controlled comparisons

- Run in evaluation mode on the identical fixed prefix; no sampled continuation becomes an input to a comparison.
- Suppress at one layer and position using `residual − activation × decoder_direction`, preserving the original reconstruction error. Record measured activation before and after; this operation does not guarantee the feature re-encodes to zero and can affect other features.
- Require baseline and zero-control agreement for output scores **and captured downstream activations**, with dtype-aware tolerances established by validation. A failed control invalidates interpretation.
- Record absolute activation deltas and baseline/changed values, target probability, log probability, and their deltas. Avoid relative percentages near zero.
- Re-encode downstream residuals, including features newly activated by the intervention. Use the union of baseline/changed candidates; do not read only the old trace's truncated feature lists. Always retain explicitly selected targets even if they fall outside display top-k.
- Restrict downstream feature nodes to later layers and causally reachable positions at or after the source position within the fixed prefix. Same-hook re-encoding belongs in intervention diagnostics, not a downstream arrow.

### What an arrow means

An experimental arrow means **changing this source produced this measured change at the destination**. It may include many intermediate routes. It is not a direct connection weight, a percentage of responsibility, or proof that the source uniquely caused the destination.

Show the intervention beside the effect sign: “Destination decreased when source was suppressed” is clearer than an unexplained negative edge. Keep feature activation units and output probability units separate. A tiny/no measured effect means no detected effect under these settings, not that a feature is irrelevant in every context. Feature labels remain descriptions, not verified explanations.

## Data and API design

Create a separately versioned `CircuitExperiment` artifact. Do not overwrite trace `edges` or mutate source traces.

Store:

- Experiment ID/status/timestamps and source trace ID plus content fingerprint.
- Exact tokens, prediction position and target ID; model revision when available, runtime/dtype, dictionary artifact IDs/hashes, hook, and source direction fingerprint. Record unavailable provenance as unknown.
- Source occurrence: dictionary, hook, layer, position, feature index. Layer/index alone is insufficient across dictionaries or token positions.
- Intervention variants and actual before/after activation; downstream scope and candidate-selection settings.
- Control metrics, baseline/changed measurements, elapsed time, and truncation/coverage metadata.
- Graph nodes/edges derived from measurements, each edge referencing its experiment and variant. Keep omitted measurements distinguishable from zero effects.

Proposed service surface:

- `POST /circuits/experiments`: validate and enqueue; return experiment/job IDs.
- `GET /circuits/experiments/{id}`: status/progress and completed results.
- `GET /circuits/experiments`: list saved experiments.
- `POST /circuits/experiments/{id}/cancel`: cooperative cancellation between forward passes or encoding chunks.

Persist completed results with atomic replacement and bounded-size summaries. Track cancellation/error states; mark interrupted jobs on restart rather than displaying them as still running. Completed results must remain viewable without a loaded model. Use existing trace storage conventions when choosing the artifact directory.

Cache/reuse a baseline only when exact tokens, model/dictionary fingerprints, runtime settings, and capture scope match. Share a baseline within a candidate sweep. Do not assume a trace's displayed top-k scores are a complete baseline.

## Interface

- **Header:** source trace, tokenized prefix, chosen output, dictionary, and saved experiment selector.
- **Graph:** layer-ordered 2D layout with token-position labels, focused on one intervention and its measured destinations. Start with at most 30 visible nodes; expanding the view reads saved results rather than automatically scheduling work.
- **Inspector:** feature label and location, baseline/changed activation, output score changes, measurement method, and Inspect in Explore.
- **Controls:** suppression first; advanced strength comparisons later. Show scope and estimated forward-pass count before starting a sweep.
- **States:** no trace, incompatible/missing dictionary, queued, measuring N/M, encoding, cancelled, failed control, no detected effects, and complete.
- **Accessibility:** a sortable measurement table provides the same information as the graph; keyboard selection and text distinguish increase/decrease in addition to color.

Keep source trace and selected feature in App-owned context so changing tabs does not reset them. Tag selections with trace/dictionary identity to reject stale navigation. Direct entry to Circuits offers saved experiments or guidance to select a feature in Explore.

## Implementation stages and acceptance criteria

### 1. Exact-token experiment core

Refactor `experiments.py` into a token-based core with the existing CLI prompt wrapper. Introduce explicit source/target selection and provenance validation. Add downstream capture and bounded encoding. Preserve existing CLI behavior.

**Done when:** tiny-transformer tests match independent manual interventions; exact token IDs survive replay; generated-target indexing is correct; no-op controls match downstream measurements; new activations are retained; earlier positions remain unaffected; hooks are removed on success and failure. Verify suppression records its actual effect instead of claiming an exact clamp.

### 2. Saved asynchronous experiments

Add experiment schemas/storage and service routes. Extend job progress without breaking existing trace/training phases. Resolve source snapshot/model before execution and enforce the existing resource locks. Add cancellation and restart recovery for experiment records.

**Done when:** API tests cover invalid/stale identities, queue serialization/model switching, cancellation, failure propagation, atomic persistence, and reload of completed results after restart. Existing trace/steering job tests pass.

### 3. Circuits workspace MVP

Add `CircuitsPage`, a graph/table, result inspector, API hook, and feature navigation actions. Begin with one selected source and one suppression experiment. Display the output effect alongside downstream feature effects.

**Done when:** a user can select a real feature in Explore, run the experiment, inspect measured effects, return to the exact source cell, and reopen saved results. Verify empty/error/cancel states, long token labels, keyboard access, and changing traces during a job. Run frontend build/lint and an end-to-end walkthrough.

**Release boundary:** stages 1–3 deliver a useful influence explorer. They do not depend on training transcoders.

### 4. “What influenced this?” search

Allow a destination feature or output token to be the target. Rank a bounded set of eligible earlier active feature occurrences as candidates, then suppress each independently and measure the same destination. Start with an explicit maximum of 20 candidates; show tested scope and omitted candidates. Activation-based candidate selection is a coverage heuristic, not evidence of importance.

Reuse the baseline and capture only the requested targets during the sweep to limit memory. Offer expanded scope as another explicit run. Add optional moderate strength comparisons and related/unrelated prompt comparisons after the single-prompt search works.

**Done when:** results rank measured effects, preserve negative/no-effect results in the table, show tested count, support cancellation, and never present untested candidates as established causes. Candidate selection and numerical thresholds are recorded. Joined arrows from independent experiments remain labelled as separate total-effect measurements; do not claim their visual chain proves mediation.

### 5. Attribution-based circuit tracing research gate

Investigate compatible pretrained transcoders or another validated attribution method for the supported model. Assess artifact availability/licensing, memory, replacement fidelity, and per-prompt cost before committing to integration or training. Reuse the workspace, but give attribution estimates a distinct edge method and validate selected hypotheses with original-model interventions.

**Go/no-go:** demonstrate useful graphs and measure agreement with original-model intervention results on several prompts. Report unexplained computation and approximation failures. Training a new transcoder is a separately scoped research effort if suitable artifacts are unavailable.

## Performance and real-model validation

Budget roughly one baseline + one zero control + one forward pass per intervention variant, plus downstream SAE encoding. An N-candidate suppression sweep begins around N+2 passes when a shared baseline/control is valid. Measure actual latency and peak memory on the target hardware before promising timings.

Capture only selected hooks/positions, stage bounded residuals on CPU as needed, and load/encode dictionaries in chunks under an explicit memory budget. Avoid retaining every layer's full SAE activations on GPU. Test a short and longer prefix and multiple downstream scopes.

On the actual supported Gemma setup, save examples with active and inactive features, repeated identical runs, related prompts, and unrelated controls. Establish numerical tolerances for the deployed dtype; a failed control must remain visible. Release requires a reproducible real-model walkthrough, not an assumption that a feature's label predicts its effect.

## Delivery order and scope

Implement stages 1 → 2 → 3, validate the MVP on target hardware, then build stage 4. Treat stage 5 as a separate decision informed by experiments. The MVP is a moderate feature project spanning measurement, persistence, and UI; automatic circuit reconstruction is substantially more uncertain. Estimate calendar time after benchmarking stage 1 and confirming model/dictionary support.

Current workspace files already contain uncommitted navigation/training/job changes. Integrate with those changes during implementation rather than replacing them with a previous version.

## Research reference

[Circuit Tracing: Revealing Computational Graphs in Language Models](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html) describes feature attribution graphs using interpretable replacement components and validation by interventions. Its distinction between approximation and original-model validation motivates the separate experimental and attribution methods above.
