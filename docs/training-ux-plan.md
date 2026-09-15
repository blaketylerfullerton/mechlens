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
