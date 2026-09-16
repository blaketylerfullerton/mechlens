# Training UX implementation plan

## Objective
Guide users through Set up → Train and check → Explore features, with one clear next action and preserved context.

## Changes
- Separate new-run setup from selected-run progress/results. Keep run history accessible.
- Show model, text source, and honest run-size presets first. Consolidate advanced controls and summarize the submitted settings.
- Derive preset selection from actual values. Prepare the selected model before enabling training; expose the loaded repository from the backend.
- Reflect actual training/validation/terminal states. Promote exploration after completion, expose failures, and retain detailed metrics/reports/downloads.
- Keep Training mounted across navigation so draft text and selected runs survive. Add a return link from custom Explore.
- Let users select an activated feature in Explore to load/collect checkpoint-specific examples and generate optional candidate labels with configured profiles.
- Scope asynchronous example results to the run/checkpoint/feature, reload saved reports, and show actionable operation errors.

## Verification
- Typecheck/build and lint the frontend.
- Run relevant backend training API tests.
- Exercise setup, model mismatch, presets/custom settings, queued/running/evaluating/completed/interrupted runs, navigation retention, and feature examples with mocked API responses in a browser.
- Inspect desktop and mobile layouts; do not launch real training or model downloads during UX verification.

## Acceptance
A first-time user can start from defaults, understand what is happening, open a saved result, select an active feature, and return to the same training context. Technical reports remain available without dominating the flow. Passing pipeline checks never implies proven interpretability.


## Multi-layer training and stored dictionaries
- Added One layer / Choose layers / All layers. Each selected layer gets the same per-layer budget, its own dictionary, and a place in the sequential compute queue.
- Added batch progress, automatic following of the active layer, per-layer inspection, and stopping all remaining layers.
- Added weight/resume-state storage estimates and actual per-run usage/free disk space.
- Retain the newest checkpoint only, after atomic save and registration; the final checkpoint also preserves resume state.
- Stored dictionaries support viewing and explicitly confirming deletion of an individual run and its owned files. The base model and other runs remain separate.
- Deletion rejects active runs and pending/running jobs that reference that dictionary, while allowing unrelated work to continue.
- Frontend build passed; three focused checks passed for queue validation/cancellation, deletion isolation/dependencies, and checkpoint retention, using temporary storage.


## Navigation and saved-work organization
- Workspace navigation is Explore / Training / Dictionaries.
- Dictionaries replaces the separate history and storage panels with All / In progress / Ready / Needs attention filters.
- Multi-layer batches appear as one expandable entry. Each layer retains its own explore, details/resume, and confirmed deletion actions.
- Destructive actions live in each run's overflow menu; storage totals and disk space remain visible in the library header.
- Training opens an active run, or setup when idle. Explicit library links can still open a completed run's report.
- Completed runs link back to their entry in Dictionaries, and setup drafts survive navigation.
