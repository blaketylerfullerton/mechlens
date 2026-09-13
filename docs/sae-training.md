# SAE training workspace

Status: planned; no training functionality is implemented by this plan.

Build a second page in mechlens where a user can prepare a model, train a
sparse autoencoder (SAE), watch progress, and inspect the saved result.
Keep the current viewer as the exploration workspace and add a Training entry
point using the same visual design.

Implementation work is tracked in [SAE training tickets](sae-training-tickets.md).
These are repository-local tickets; none have been published to an issue tracker.
The repo also uses OpenSpec for detailed implementation changes. When work on
a milestone begins, its proposal/specs should reference this plan rather than
duplicate the roadmap.

## What training means

The language model stays frozen. A text dataset runs through it to produce
activations at a selected layer and hook; a separate SAE learns to reconstruct
those activations using a sparse set of features. Downloading the model alone
is therefore insufficient: each run also needs a dataset and token budget.

Use the existing SAELens dependency for training, behind a small adapter. Verify
the installed version's training, logging, checkpoint, and restoration APIs
before committing to the adapter contract. Reference:
[SAELens training documentation](https://github.com/decoderesearch/SAELens/blob/main/docs/training_saes.md).

## First milestone

Deliver one complete path: select the existing Gemma base model, prepare its
weights, train one residual-stream layer, save the SAE, and inspect it.

- Model: `gemma-2-2b`, using the existing loader and an explicit model revision.
- Activation site: one selected `blocks.L.hook_resid_post` per run.
- Data: one vetted text dataset preset, with recorded revision, split, text
  field, tokenizer, sequence length, seed, and token budget. Select the preset
  during the training spike and reserve separate evaluation data.
- Training: one verified SAE architecture and conservative presets. Expose
  layer, feature count, token budget, and the architecture's sparsity control;
  put learning rate and batch size under advanced settings.
- Monitoring: persistent run history, phase, tokens processed/target,
  throughput, estimated time remaining once measurable, loss, reconstruction
  quality, average active features (L0), dead-feature fraction, and resources.
- Controls: start, cancel safely, view errors, and open saved artifacts.
  Checkpoints survive interruption. Exact resume is enabled only after all
  required training state can be restored and tested.
- Exploration: load the saved SAE for its trained layer and inspect activations
  and feature IDs. Show a clear unlabeled state and no borrowed atlas or labels.

Compute runs on the machine hosting the Python backend, which may differ from
the browser's machine. Benchmark the intended training device before choosing
default budgets or quoting runtime. A tiny CPU run is for correctness; it does
not establish useful Gemma training performance on a Mac or GPU server.

## User workflow

1. Open Training and choose the supported model. Show cached/download/loading
   states and actionable Hugging Face authentication or license errors.
2. Choose layer, dataset preset, and training size. Validate compatibility and
   show the selected device plus a measured or explicitly approximate resource
   estimate before starting.
3. Start the run. Show preparation, training, evaluation, and saving as distinct
   phases; graph metrics against tokens processed.
4. Close and reopen the page without losing the run. A backend restart marks
   unfinished work interrupted and preserves its metrics and valid checkpoints.
5. Open a completed SAE in the viewer. Only trained layers have SAE features;
   the rest remain available for the viewer's supported non-SAE measurements.

## Backend design

### Run execution and persistence

The current `backend/app/service/jobs.py` queue is in memory and serves short
trace jobs. Add a durable training run store and a managed training worker.
Keep metadata and metric reads responsive while training executes. Coordinate
GPU ownership across training, tracing, and steering; a subprocess cannot rely
on the existing process-local `_forward_lock` alone. For the first version,
serialize compute jobs and make waiting visible in the UI.

Proposed run states: queued, running, cancelling, cancelled, completed, failed,
and interrupted. Separate state from phase and progress counters. Publish
checkpoints atomically, preserve the last valid artifact after failure, and
reconcile unfinished jobs on startup. Store metrics at bounded intervals so
neither disk use nor polling responses grow without limit.

Proposed API surface, to finalize during implementation: create/list/get runs,
read metrics after a cursor, cancel a run, and list/load produced SAE artifacts.
Resume is a separate operation once restoration semantics are verified.

### Artifact identity and viewer integration

Extend the existing identity checks rather than bypassing them. Each artifact
records model/tokenizer revisions, activation hook and layer, dimensions,
normalization/preprocessing, SAE architecture and configuration, dataset
provenance, seed, package versions, run ID, and checkpoint content identity.
Record unknown provenance explicitly instead of implying reproducibility.

Validate the selected model, hook, dimensions, and preprocessing before using
an SAE. Labels, feature lookups, steering, and layouts must resolve against the
actual dictionary identity. A feature index in a new SAE has no relationship
to the same index in Gemma Scope. Preserve existing Gemma Scope behavior and
legacy traces while adding custom artifacts.

### Evaluation

Keep training metrics distinct from held-out evaluation. Report reconstruction
error/explained variance, L0, and a defined inactivity window for dead features.
Include a held-out language-model loss comparison with SAE reconstructions
substituted at the trained hook to measure downstream fidelity. Save metric
definitions with the report; no single score establishes interpretability.

Prefer streamed activation batches initially. Full activation caches are an
optional later optimization because their disk footprint can be substantial.

## Later milestones

1. **More models:** accept Hugging Face IDs for explicitly supported model
   families, validate tokenizer and hooks, expose download progress, and manage
   model switching without briefly retaining two large models in memory.
2. **Run management:** exact checkpoint resume if the first milestone only
   supports saved artifacts, comparisons, custom datasets, and multiple layers.
   Sharing a model forward pass saves activation-generation work, but each
   additional SAE still adds training compute and memory.
3. **Feature examples:** scan a corpus and save bounded high-activation examples
   with token context, sampling provenance, and dictionary identity.
4. **Auto-interp:** generate candidate explanations from examples, evaluate on
   separate examples/controls, and store explainer provenance and scores. Then
   build a dictionary-specific atlas where supported.

Automatic labeling, arbitrary Hugging Face architecture support, distributed
training, and lab-scale quality claims are outside the first milestone.

## Completion criteria

A user can prepare Gemma, launch a small single-layer training run from the
page, observe real metrics, revisit its history after restarting the backend,
and load the saved SAE in the viewer without incorrect labels or layout.
Cancellation, worker failure, incompatible artifacts, and GPU contention have
explicit tested behavior. A documented run on the intended training hardware
records throughput, peak memory, evaluation results, and configuration.
