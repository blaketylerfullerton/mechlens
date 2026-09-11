# Backend review — language-model brain mapping

Reviewed checkout `baac95a` on September 10, 2026. This is a review, not an implementation change. Scope: backend application modules, service, enrichment passes, atlas builder, label import/verification scripts, tests, setup, and specifications; frontend contracts were consulted where relevant. Existing untracked `.DS_Store` files were left alone.

**Assessment:** Keep the existing capture/pass architecture. It is a useful activity-inspection foundation with real numerical diagnostics. Before adding causal mapping, fix scientific-data identity, enrichment replacement, intervention replay, and executable setup. The biggest risk is a plausible-looking map whose labels, positions, or attribution do not describe the same computation.

**Validation performed**

- Full suite: `.venv/bin/python -m pytest backend/tests -q` fails during collection because `app.cli` imports missing `app.viewer`.
- With that test module excluded, offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest backend/tests --ignore=backend/tests/test_cli.py -q -ra`: **284 passed, 13 failed, 14 errors, 1 skipped** in 24.59s. Atlas-build failures/errors arise from missing `umap` in this local environment. `umap-learn` is declared in requirements: this is an environment/setup gap, not evidence of 27 independent algorithm bugs. The real-SAE test was skipped by its opt-in flag.
- Additional isolated probes used temporary SQLite databases and the cached tiny-stories-1M CPU model. They reproduced stale layout data, incompatible atlas acceptance, stale rows on rebuild, changed labels retaining old embeddings, permissive request validation, and attribution replay errors.
- Did not load full Gemma/Gemma Scope, download weights, rebuild the full atlas, or reproduce README pilot measurements. The Mac device mismatch is established by tracing the two device-selection paths, not a full Gemma run.

**1. P1 — Every CLI command is currently broken**

`backend/app/cli.py:43` imports `DEFAULT_PORT` from `.viewer`; no `backend/app/viewer.py` or viewer package is tracked. Because this is a top-level import, even `trace`, `enrich`, `show`, and help fail before argument dispatch. The main suite cannot collect `test_cli.py` for the same reason.

Fix: retire the old viewer command and dependency or restore its implementation; run the documented CLI commands and the full suite from a fresh checkout. This is the first cleanup because it prevents using the saved-trace workflow at all.

**2. P1 — HTTP SAE enrichment selects the wrong device on Macs**

`backend/app/model_cache.py:32` selects MPS when available. `backend/app/sae_cache.py:42` selects CPU whenever CUDA is unavailable. The default service provider follows the latter, but `backend/app/service/app.py:199` constructs `SAEPass` with the model's device. `backend/app/passes/sae.py:85` moves inputs there before passing them into the CPU SAE.

On an MPS model, this sends MPS inputs to CPU weights. The same assumption prevents deliberately keeping SAEs on CPU beside a CUDA model.

Fix: derive the encoding device from each SAE's parameters, or have one explicit placement policy used by both loader and pass. Test MPS-model/CPU-SAE and CUDA-model/CPU-SAE configurations without requiring full-size weights.

**3. P1 — Feature identity is not enforced across model, SAE, labels, and atlas**

`backend/app/passes/sae.py:161` checks hook and normalization but not model identity. `backend/app/passes/labels.py:48` uses the pass's default width instead of deriving/checking it against the preceding SAE record. `backend/app/passes/layout.py:88` selects by source/version without checking release or width; `backend/app/labels.py:555` likewise does not filter latest atlas by bundle identity.

Concrete user path: after encoding a 65k trace, a separate `enrich --labels` defaults back to 16k. Overlapping feature indices get plausible but unrelated labels. Building a newer 65k label atlas can also make it the default atlas for 16k service traces.

Probe: LayoutPass accepted an atlas carrying `width='65k'` and `release='other-release'` and attached its position to the synthetic trace. The SAE compatibility guard accepted an unrelated `test-model` name.

Fix: define a bundle identity covering model/revision, SAE release/id/width/hook, and label source. Validate it at every pass boundary. Treat `(layer, index)` as an identity only inside that bundle.

**4. P1 — Re-enrichment can silently mix old and new results**

`backend/app/passes/__init__.py:35` replaces the pass record after a run, but does not manage dependent data. SAE/lens/attribution passes only touch selected layers. Labels and layout merge into existing dictionaries (`passes/labels.py:75`, `passes/layout.py:122`), leaving entries absent from the new result. Missing-atlas handling returns without clearing an old layout.

Probe: a trace retained position `(9,9,9)` while its new layout record said `atlas_available=False` and zero features placed. Switching from a broad atlas to a narrower one can mix coordinates from both under the newer identity. Re-encoding only some layers with a different SAE width leaves other layers from the old dictionary while the sole SAE record describes the new run.

Fix: define replacement versus incremental enrichment explicitly. Replace owned side tables atomically; clear or invalidate dependent labels/layout when features change. For incremental layer updates, record provenance per layer. Preserve the previous consistent trace if a pass fails partway through.

**5. P1 — Atlas versions do not identify their actual inputs; rebuilds can leave mixed tables**

`backend/app/atlas.py:506` hashes release, width, seed, and parameters, not source embeddings/weights or feature keys. `backend/scripts/build_feature_atlas.py:604` uses that identity even when imported labels/embeddings change. `backend/app/labels.py:445` upserts layout rows without removing rows absent from a rebuild. Layout, clusters, and record are committed separately.

Probe: writing two rows under a version and rebuilding it with one row still returned two rows. A changed embedding snapshot reuses the old version; removed input features and obsolete clusters can survive. The positions hash changes, but old data is overwritten under the same version instead of remaining available for replay. An interrupted build can publish only part of the update.

Fix: fingerprint input data, ordered feature identities, relevant build/code versions, and outputs. Publish a new immutable atlas in a single transaction. Include feature keys and cluster assignments in the artifact fingerprint, not coordinates alone. Test changed inputs and interrupted publication, not just identical-seed rebuilds.

**6. P1 — Attribution does not replay steering**

`backend/app/passes/attribution.py:225` reruns stored tokens without applying `trace.steering`. Its compatibility guard accepts steered traces. The stored residuals therefore come from one computation and the cached attention/MLP tensors from another. Merely adding an intervention hook is insufficient: the decomposition also needs to account for the intervention's own additive contribution.

Probe on tiny-stories-1M: unsteered reconstruction gap was `0.0`; steering layer 1 with a constant direction produced a gap of `0.999956`, yet the pass completed and emitted edges. The chosen site's emitted edge-weight sum remained the unsteered value.

Fix: reject steered attribution until supported, then replay a fully specified intervention and represent its contribution. Add baseline-versus-intervention reconstruction tests. The existing diagnostic detects the discrepancy numerically, but does not prevent publishing incompatible attribution.

**7. P2 — Current edge weights are magnitudes, not conserved or causal contributions**

`backend/app/passes/attribution.py:109` assigns vector norms to edges. Norms discard sign and cancellation; the sum of norms need not equal the residual's norm. The reconstruction check compares full vectors, not serialized edge weights. The specification's sum-to-residual-norm requirement is therefore not implemented.

Probe with attention truncation effectively disabled: a site's edge weights summed to `2.078919`, while its residual norm was `0.763265`, despite a perfect vector reconstruction check.

This does not invalidate the underlying vector decomposition. It invalidates interpreting edge thickness as additive importance or as evidence that a source caused a particular answer. Also, `backend/app/schema.py:119` only identifies a source feature; the implied target is a whole LayerState. It cannot unambiguously represent multiple destination SAE features at that site.

Fix: name the existing metric as contribution magnitude and reconcile the spec. Before feature circuits, define explicit source/target feature sites, a signed target-specific quantity, intervention/error/bias terms, and a validation method. Separate descriptive flow from measured effects on an output or target feature.

**8. P2 — The default atlas's cross-source diagnostic is mislabeled**

`backend/scripts/build_feature_atlas.py:563` invokes `cross_source_agreement` for both source types. That function (`:403`) always clusters explanation embeddings. For `source='labels'`, it compares the projected label-derived clustering against another clustering of label embeddings; no decoder directions participate.

Consequently `cross_source_ari` is not decoder-versus-label agreement for the default atlas. It cannot serve as independent corroboration that label-defined regions reflect model geometry.

Fix: load both representations for the same feature subset, compare their independently produced clusterings, and record source/metric/sample identities. If that comparison was not run, report it as unavailable or rename the actual within-source diagnostic.

**9. P2 — Label refresh can silently separate a label from its embedding**

`backend/app/labels.py:431` retains the previous embedding with `COALESCE` whenever an update has no vector, while replacing text/explainer metadata. A normal import without `--embeddings` after an earlier embedding import can create this state.

Probe: changed text to an unrelated description; the stored embedding remained `[1,0]` from the prior description. Layout and coherence then analyze old text while naming/display uses new text.

Fix: record explanation identity or text hash with the vector. Preserve a vector only if the associated explanation is unchanged; otherwise invalidate it or require a matching replacement.

**10. P2 — Resource limits and lifecycle are inadequate for sustained use**

`backend/app/service/models.py:23` has no prompt length, generation-budget upper bound, context-budget validation, or finite steering-coefficient guard. A budget of `10**12` and an infinite coefficient both passed model validation in a probe. `capture.py` allocates residual storage from that budget before generation.

`backend/app/service/jobs.py:81` has an unbounded queue and retains completed traces indefinitely. There is no cancellation, retention limit, or graceful worker shutdown. The expensive SAE load in `service/app.py:282` happens on the request path, outside the serialized worker/forward lock, so concurrent steering requests can load weights while generation runs.

Fix: validate tokenized input plus generation against model context and a configured memory budget; require finite coefficients; bound queued work and completed-result retention; add cancellation between steps/layers. Move SAE loading and intervention preparation into serialized execution. Keep single-process deployment explicit until jobs and model ownership are redesigned.

**11. P2 — Memory policy assumes the original large-memory machine**

`backend/app/sae_cache.py:68` caches up to 32 SAEs by entry count. The service constructs a dictionary of all requested SAEs before encoding any layer (`service/app.py:201`). All 26 fp32 16k SAEs need approximately 7.9 GB just for their tensors, plus model and activations. Larger widths increase this substantially; an entry-count cache is not a memory limit.

Generation reruns the whole prefix; attribution materializes `[positions, positions, d_model]` float32 contributions (`passes/attribution.py:282`). At 1,024 positions and Gemma's 2,304 dimensions, that one tensor is about 9 GiB before intermediates. These paths are appropriate only for explicitly bounded short traces.

Fix: implement an explicit device/memory budget, bounded SAE residency, and per-layer loading/eviction. Chunk attribution by destination position. Consider KV caching only after preserving and testing capture/intervention semantics.

**12. Capability gap — HTTP traces cannot become durable, replayable experiments**

`backend/app/service/app.py:261` discards residual arrays and retains only a Trace in memory; process restarts lose job access. `/steer` runs no SAE, labels, layout, or lens passes. `TracePass` excludes attribution. The UI therefore cannot request the same rich analysis for a steered trace or later enrich the original HTTP capture from its residuals.

Progress contains counters only (`service/models.py:88`), so live per-layer feature lighting remains blocked exactly as the README acknowledges.

Fix: use one capture/enrichment pipeline for CLI, ordinary HTTP traces, and steered traces. Persist an experiment record and residual sidecar with configurable retention. Add baseline linkage, identical-token replay for controlled comparisons, and partial feature events when real live mapping is needed.

**13. Capability gap — The current map is semantic/activity mapping, not a causal brain graph**

The default atlas is built from explanation embeddings. Its cluster coherence uses those same embeddings, so this is a useful semantic-consistency check, not independent validation of functional model regions. Low aggregate explainer AMI likewise does not establish that all local regions are free of description-style effects. The brain-shaped warp supplies a display shape, not anatomical evidence.

The README's pilot covers six of 26 layers. `load_label_embeddings` (`build_feature_atlas.py:183`) omits features without embeddings, contrary to the spec's all-features requirement. SAE capture retains top 16 activations per cell (`passes/sae.py:95`); activity outside that set is lost from the JSON. Activation rank alone does not establish causal importance.

Define the intended claim before phase 8: for example, “on this input, changing source feature A changes target feature B or output score C.” Build controlled ablation/patching experiments, retain the SAE reconstruction error, and test predictions on held-out prompts. Keep coverage and omitted features explicit. A semantic atlas can remain the navigation surface while causal experiments provide separate evidence.

**14. P2 — Reproducibility and validation need stronger contracts**

Requirements use open-ended lower bounds; there is no backend package/lock manifest or recorded dependency environment in traces. Model/SAE revisions and tokenizer identity are not captured. `store.py:79` accepts schema versions without explicit compatibility policy; residual loading checks sidecar shape against declared shape but not all Trace dimensions, dtype, token ordering, or a content checksum. JSON updates overwrite the existing file directly.

Tests include useful real CPU and synthetic architectural checks, but several model fixtures skip on any loader exception, which can also hide dependency regressions. The README's `316 passed` is historical, not the state verified here.

Fix: pin a supported runtime, record bundle/environment fingerprints, validate trace invariants on load, and use atomic writes. Keep deterministic offline tests mandatory; allow network-dependent integration tests to skip only for explicit availability reasons.

**15. Lower-priority cleanup**

- `Makefile:20` looks for `venv`, while this checkout uses `.venv`; its launcher requires `setsid`, which is absent from this machine's PATH. Provide a portable launcher and detect the documented environment.
- `backend/app/trace 2.py` duplicates obsolete smoke-generation code and uses a different model-loading path. Remove it after confirming nothing depends on it.
- Read-only label/atlas operations import SAE loading and initialize writable SQLite schema. Separate lightweight data access from model/SAE runtime loading, and split build/test dependencies from serving dependencies.
- Reconcile stale phase numbers, missing phase-document references, and specs that contradict actual behavior. Do this alongside fixes rather than rewriting all documentation first.

**Recommended sequence**

1. Restore the CLI and a reproducible green baseline; fix device placement.
2. Enforce bundle identity and correct enrichment replacement/invalidation.
3. Make atlas publication immutable/atomic; couple text and embeddings correctly; repair the cross-source diagnostic.
4. Unify steered and ordinary capture, preserve experiments, and make intervention replay correct.
5. Add bounds, cancellation, and explicit memory management.
6. Design and validate one small feature-to-feature causal experiment before expanding to a whole graph.

Avoid a wholesale rewrite or splitting this into microservices. Most of the immediate weaknesses are missing contracts between otherwise useful components.
