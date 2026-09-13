# SAE training tickets

Status: local backlog. All tickets are open; these are not external issue IDs.
Scope and decisions live in the [training plan](sae-training.md).
Check a ticket off only when its acceptance criteria have been verified.

## First milestone

### SAE-01 — Prove the training adapter and choose presets

- [ ] Complete SAE-01.

Dependencies: none.

Build a minimal SAELens training spike using a tiny test model and streamed
activations. Inspect the installed API, choose one architecture and dataset
preset, and document the configuration for a single Gemma residual layer.

Acceptance criteria:
- A tiny run performs optimization and writes a reloadable SAE artifact.
- Metrics can be captured locally without requiring a hosted logging account.
- Checkpoint contents and restoration limitations are documented, distinguishing
  exact resume from starting a new run with existing SAE weights.
- A short benchmark on the intended training hardware records throughput and
  peak memory; default budgets are justified by that measurement.

### SAE-02 — Register and validate custom SAE artifacts

- [ ] Complete SAE-02.

Dependencies: SAE-01 artifact format.

Add local artifact registration/loading and extend model/SAE provenance beyond
the fixed Gemma Scope release in `sae_cache.py` and `identity.py`.

Acceptance criteria:
- Manifest stores the provenance and content identity specified in the plan.
- Model, hook, dimensions, and preprocessing mismatches produce clear errors.
- Feature lookups cannot attach labels or layouts from a different dictionary.
- Existing Gemma Scope artifacts and saved traces remain supported.

### SAE-03 — Add durable runs and compute coordination

- [ ] Complete SAE-03.

Dependencies: SAE-01 worker requirements.

Create persistent run records, bounded metric storage, worker lifecycle
management, cancellation, and API endpoints. Coordinate training with existing
trace/steering execution across process boundaries.

Acceptance criteria:
- Run history and metrics survive backend restart; unfinished runs become
  interrupted rather than appearing to run forever.
- Cancellation and worker failure preserve valid checkpoints and release
  compute ownership; partial writes are not exposed as usable artifacts.
- Training and inference do not overlap on the shared device in this milestone.
- Status/metric reads stay responsive; API tests exercise restart recovery,
  queued cancellation, running cancellation, and failure cleanup.

### SAE-04 — Execute and evaluate single-layer training

- [ ] Complete SAE-04.

Dependencies: SAE-01, SAE-02, SAE-03.

Connect the training adapter to real run execution. Prepare Gemma and the
dataset, stream activations, train, checkpoint, evaluate, and register results.

Acceptance criteria:
- Base model weights stay frozen; only the selected SAE is optimized.
- The resolved dataset/model settings and seed are saved with the run.
- Real phase, token, throughput, loss, sparsity, and reconstruction metrics
  reach persistent storage; held-out evaluation is separately identified.
- Final evaluation includes downstream language-model loss with reconstruction
  substitution, and defines the dead-feature measurement window.
- A tiny integration run completes and its artifact reloads for encoding.
- Checkpoint restoration is tested if exposed; otherwise the API/UI clearly
  report that exact resume is unavailable.

### SAE-05 — Build the Training page and live run details

- [ ] Complete SAE-05.

Dependencies: SAE-03 API contract; SAE-04 for end-to-end verification.

Add viewer/training navigation, run configuration, model preparation states,
run history, metric charts, cancellation, and artifact actions in the current
visual style.

Acceptance criteria:
- The supported Gemma model, dataset preset, layer, feature count, sparsity
  setting, and token budget can be selected and validated before submission.
- Download progress reflects available byte counts; otherwise show an honest
  indeterminate state. Authentication/license errors include recovery guidance.
- Charts use actual metrics and tokens processed; ETA appears only when a
  throughput estimate is available, and resource readings identify the backend.
- Reloading or navigating away preserves the run; queued, interrupted, failed,
  and cancelled states are distinct and actionable.
- Frontend build/lint pass and a browser walkthrough verifies the real API flow.

### SAE-06 — Open trained SAEs in the viewer

- [ ] Complete SAE-06.

Dependencies: SAE-02, SAE-04, SAE-05.

Connect the completed run's artifact action to trace generation and inspection
using the selected dictionary at its trained layer.

Acceptance criteria:
- A saved SAE produces inspectable feature activations for a fresh prompt.
- Untrained layers are distinguished from layers with zero feature activity.
- Custom features show IDs and an unlabeled state; the old Gemma Scope atlas
  and labels are never reused for them. Unsupported actions are explicit.
- Switching back to the standard Gemma Scope viewer still works.
- Regression checks cover identity isolation and the existing trace/steer paths.
- An end-to-end Gemma run on the intended hardware is documented with its
  configuration, runtime, memory, and evaluation results.

## Follow-up backlog

### SAE-07 — Support more Hugging Face models

- [ ] Complete SAE-07.

Dependencies: first milestone.

Accept repository IDs for a documented list of supported families. Validate
hooks/tokenizers and manage downloads, revisions, and memory-safe switching.
Done when a second supported family trains and opens in the viewer, and an
unsupported model is rejected before a training job starts.

### SAE-08 — Expand run management

- [ ] Complete SAE-08.

Dependencies: SAE-03, SAE-04.

Implement verified exact resume if still missing, run comparisons, custom
datasets, and then multi-layer scheduling as separate implementation slices.
Resume must restore optimizer/scheduler, random state, counters, and data-stream
state; a weights-only restart is labeled as a new run. Measure added memory and
compute before publishing multi-layer presets.

### SAE-09 — Collect feature activation examples

- [ ] Complete SAE-09.

Dependencies: SAE-02, SAE-06.

Scan a corpus and store bounded examples with token context, activation values,
sampling metadata, and dictionary identity. Done when a feature's examples can
be inspected in the viewer and remain isolated from other SAE checkpoints.

### SAE-10 — Generate and evaluate auto-interp labels

- [ ] Complete SAE-10.

Dependencies: SAE-09.

Generate candidate explanations, score them using held-out examples and
controls, and record explainer model/prompt provenance. Define provider and
cost settings before running external labeling jobs. Done when the viewer
shows explanations with their evaluation/provenance, unsuccessful explanations
remain visibly unverified, and any generated atlas uses the same dictionary.
