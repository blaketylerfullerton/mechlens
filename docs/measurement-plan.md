# Trustworthy measurements: implementation and Spark validation

The goal is to establish what a displayed feature means, keep each experiment's data separate, and measure whether changing a feature changes a chosen output. This work leaves the UI and Mac device handling alone.

## Implementation plan

1. **Names match features.** Require SAE provenance before labels/layout; derive label width from that provenance; reject explicit mismatches; validate loaded SAE model/hook/width metadata; select an atlas from the matching release and width.
2. **Reruns contain fresh results.** Apply enrichment to a private trace copy, publish on success, replace all fields owned by a pass, and invalidate labels/layout after SAE reruns. Fingerprint atlas inputs, annotations and results; publish all atlas tables in one transaction. Invalidate embeddings when their explanation changes.
3. **Measure an actual effect.** Reject attribution on steered traces until replay plus an intervention contribution is implemented. Identify current attribution weights as L2 magnitudes. Compare independent decoder/text clusterings. Add a saved, fixed-prefix experiment with baseline, zero control, decoded-feature suppression and multiple steering strengths.

These changes are implemented. The tests exercise mismatched dictionaries, missing provenance, stale positions, partial failures, atlas publication rollback, changed annotations, independent cross-source comparisons, hook cleanup, zero controls and direct intervention agreement.

## Behavior changes to know about

- A pass rerun is **replacement**, including when a subset of layers is requested. For example, rerunning SAE on layer 20 clears older features on other layers and invalidates all labels/layout. Run labels/layout again afterward. This prevents a single pass record from describing mixed measurements.
- Older traces with no SAE release/width record must be re-encoded before labeling or placement. Existing records carrying release/width remain readable; no unseen weight revision is inferred for them.
- `enrich --labels` derives width from the trace. Explicitly passing a different width fails.
- Atlas rebuilds produce a new identity if inputs, annotations or measured output change. Changing labels can change a decoder atlas's artifact identity without changing its positions. Exact publication retries are harmless; conflicting retries fail.
- Previously built atlases remain readable. Their historical cross-source metrics are not retroactively corrected; rebuild them before relying on those metrics. The corrected label-atlas comparison now loads decoder weights for sampled features, so the build requires SAE access and more time/memory than a text-only build.
- Attribution on steered traces fails clearly instead of returning an explanation of an unsteered run. Full steered-attribution replay and causal feature graphs remain separate work.

## Run a controlled experiment on the DGX Spark

Use the repository's Python environment and run from `backend/`. Choose a feature from a trace you have inspected. The example below uses a feature mentioned in the README, but is **not** a claim that the feature controls this answer.

```bash
python -m app.cli experiment \
  --prompt "The Golden Gate Bridge is located in the city of" \
  --prompt "The famous Golden Gate Bridge is in" \
  --control-prompt "The Eiffel Tower is located in" \
  --layer 20 --feature-idx 1370 --width 16k \
  --target " San" --coefficients -5 -1 1 5 \
  --out traces/bridge-experiment.json
```

`--target` must encode to exactly one token, including its leading space. If it does not, the command rejects it; use `--target-token-id` for an explicit vocabulary ID. This experiment measures the next token, not the probability of a multi-token answer.

Every variant gets the same tokenized prompt. No generated continuation is fed back, so output differences cannot be explained by changing the input halfway through. By default, the intervention is at the last prompt position; `--position` selects another absolute token position including BOS. The measured output remains the next-token distribution at the end of the prompt.

For each prompt the report includes:

- Baseline target probability and log probability.
- A zero-strength control; the command fails if its target scores differ beyond tolerance.
- Suppression by subtracting the active feature's decoded contribution from the original residual, preserving other contributions and SAE reconstruction error. This is a residual intervention, not guaranteed to clamp the encoder's post-intervention feature activation to exactly zero. Before/after activations are recorded.
- Requested positive/negative steering strengths and their probability/log-probability changes.
- Exact input token IDs, intervention position, target ID, SAE release/width/id, direction fingerprint, runtime versions, model name, dtype and available revision metadata. An unavailable model revision is recorded as null; the report does not claim to pin unavailable metadata or every model/SAE weight.

The report is saved only after all requested examples complete, using an atomic file replacement. It is an experiment record, not a generated trace; it does not change `/trace` or `/steer` persistence.

## Read the result simply

1. The zero control must match baseline. If it fails, do not interpret the rest.
2. Check the feature was active before interpreting suppression. Removing a contribution of zero tells you little.
3. Look for a repeatable change across related prompts. One changed probability is a measurement on one input, not a general discovery.
4. Check unrelated controls. An intervention that disrupts every prompt may be a broad disturbance rather than a specific mechanism.
5. Inspect both steering strengths and the measured before/after activations. Large coefficients can move the model far from its ordinary behavior.

A passing software test demonstrates that the measurement procedure works. A Spark experiment must still establish whether your selected Gemma feature has a meaningful effect. No full Gemma/Gemma Scope experiment was performed on the local Mac during implementation.
