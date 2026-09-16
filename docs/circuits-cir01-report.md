# CIR-01 — output-targeted tracing on the target hardware

Status: complete. This is the go/no-go report for
[the Circuits feature spec](circuits-feature-spec.md) and
[the circuit tracing plan](circuit-tracing-plan.md).

Everything below was measured on this machine on 2026-09-15. The prototype, its
lock file and its scripts are in `prototypes/circuits/`; the raw JSON each number
came from is in `prototypes/circuits/out/` (gitignored — regenerate with the
commands in §3).

## 1. Verdict

**Go. No architectural condition — a single environment works.**

`circuit-tracer` builds useful, output-targeted attribution graphs for mechlens
traces on this hardware in seconds, not minutes. Exact-token replay works: the
prefix goes in as token ids and the target as an explicit logit, so nothing is
re-tokenized. A plain forward pass through the replacement model agrees with
mechlens's own model to **one bfloat16 ULP**. Interventions run on the original
model, not a replacement-only surrogate, and their zero controls are exact.
Feature evidence is real and, in the good cases, plainly interpretable.

The dependency conflict that looked architectural is not. mechlens declares
`transformers>=4.43` — a floor, no ceiling — and never required 5.x; that is
merely what pip resolved. A single environment holding mechlens *and*
`circuit-tracer` resolves cleanly and runs **the same 346 passing / 21 failing
tests as the current venv, with an identical failure set**. The only code change
is inlining a three-line helper. See §9, which was rewritten after that was
measured.

Two things the spec asked for do **not** exist and must shape CIR-03 rather than
be discovered by it: transcoder features have **no labels** (§5.3), and there is
**no cancellation of any kind** (§7.2).

## 2. Pinned versions

| | |
| --- | --- |
| `circuit-tracer` | `7f66876689f59e92fc3641650d38b8bd41749ec4` (2026-09-11), reported as `0.5.3.dev2+g7f6687668` |
| source | `github.com/decoderesearch/circuit-tracer`, MIT. `safety-research/circuit-tracer` is the same repo at the same SHA |
| model | `google/gemma-2-2b`, base, gated — an accepted licence is required |
| transcoders | `mntss/gemma-scope-transcoders`, revision `9250a2d4860ce5ed5c96c14d5882b7d8162809a3`, 3.98 GB / 29 files |
| torch | 2.14.0+cu130, `cp312-manylinux_2_28_aarch64` wheel from plain PyPI |
| transformers | 4.57.3 · transformer-lens 3.2.1 · nnsight 0.8.0rc1 · numpy 2.5.3 |
| python | 3.12.3 · Linux aarch64 |
| GPU | NVIDIA GB10, compute capability 12.1, 128.5 GB unified memory |

Full resolution in `prototypes/circuits/requirements.lock`.

Notes on provenance that CIR-02 inherits:

- The PyPI package named `circuit-tracer` (0.5.0) is a **third-party fork**
  (`Caerii`), not this repo. It is not used here. Pinning by commit is the only
  way to name a stable artifact, since upstream publishes no tagged release to
  PyPI.
- The transcoder repo is **mostly feature evidence, not weights**: 27 of its 29
  files are under `features/`, ~3.8 GB of the 3.98 GB total.
- `mntss/gemma-scope-transcoders` is what the upstream README recommends for
  Gemma-2-2B. The `gemma` CLI shortcut resolves to a *different* repo,
  `mwhanna/gemma-scope-transcoders` (11.83 GB). The cross-layer sets are larger
  again (`mntss/clt-gemma-2-2b-426k`, 33 GB). Per-layer was used throughout.
- mechlens's `Trace` schema carries **no model revision field**, so the revision
  of the weights that produced a saved trace is genuinely unknown and is recorded
  as `null` rather than back-filled from whatever is on disk today. CIR-02 should
  add the field.

## 3. Reproduction

```bash
cd prototypes/circuits
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock

python3 export_trace.py 0405a85811f4 --step 12
.venv/bin/python trace_graph.py     --export out/0405a85811f4-12.json --name ggb-francisco
.venv/bin/python verify_identity.py --export out/0405a85811f4-12.json
.venv/bin/python intervene.py       --export out/0405a85811f4-12.json \
                                    --graph  out/ggb-francisco.pt --top 3 --constrained
.venv/bin/python feature_evidence.py 21:6271 24:2455
.venv/bin/python bench.py --tokens 8 16 32 64
.venv/bin/circuit-tracer start-server --graph_file_dir out/graph_files
```

## 4. Identity and baseline agreement

From `out/identity-0405a85811f4-12.json`. Target: trace `0405a85811f4`,
`"The Golden Gate Bridge is located in the city of San"` → `" Francisco"`,
prefix = 12 token ids, target position 12.

| check | result |
| --- | --- |
| BOS present at position 0 | pass — id 2, and positions are indexed including it |
| target excluded from its own prefix | pass — `len(prefix) == 12 == target_position` |
| replacement vs. mechlens forward pass | pass — **1 bfloat16 ULP** |
| argmax agreement | pass — both 12288 (`" Francisco"`) |

```
                logit     log prob      probability
replacement    7.5       -0.0273847    0.9729868
mechlens       7.46875   -0.0279327    0.9724538
abs error      0.03125                 0.00053304
```

The logit tolerance had to be **restated during this work, and that is a
finding, not a formality.** An absolute bound of 1e-2 was the obvious first
choice and it is wrong: at magnitude 7.5, one bfloat16 ULP is 0.03125, so 1e-2 is
finer than the dtype can represent and would fail two models that agree as
closely as bfloat16 permits. The two logits here are *adjacent representable
bfloat16 values* — bit distance 1. **Logit tolerances must be expressed in ULPs
of the compute dtype**; probabilities, derived in fp32 after the softmax, can
keep an absolute bound (5.3e-4 against 1e-3 here). CIR-02 should carry this rule
into the adapter rather than rediscover it.

Both models are built with `from_pretrained_no_processing` in bfloat16 on cuda,
matching `backend/app/model_cache.py`. No LayerNorm folding, no weight centering.

### The re-tokenization hazard is real

Decoding the 12-token prefix and re-encoding it yields **13 tokens**: the decode
emits a literal `<bos>` string and the re-encode prepends another BOS.

```
ids in : [2, 651, 17489, 22352, 16125, 603, 7023, 575, 573, 3413, 576, 3250]
ids out: [2, 2, 651, 17489, 22352, 16125, 603, 7023, 575, 573, 3413, 576, 3250]
```

Passing ids directly avoids this entirely — `attribute()` accepts
`list[int]` — so it is a hazard avoided rather than a failure. It is recorded
because it is the concrete evidence for why the pipeline must never re-tokenize
decoded trace text.

### BOS is zeroed for transcoder purposes

`ReplacementModel` sets `zero_positions = slice(0, 1)` for this model family.
Position 0 therefore carries no feature nodes **by construction**. A UI must not
render that absence as "nothing happened here"; it is suppression, not
measurement.

## 5. Graph quality

### 5.1 The structure CIR-02 has to store

Node order in the dense adjacency matrix is
`[active_features, error_nodes, embed_nodes, logit_nodes]`, with
`n_layers * n_pos` error nodes and one embed node per position. Verified on the
Golden Gate graph: 5000 + (26 × 12 = 312) + 12 + 1 = **5325**, matching
`adjacency_matrix.shape == [5325, 5325]`.

| graph | pos | active | selected | attribution | `.pt` | target prob |
| --- | --- | --- | --- | --- | --- | --- |
| `ggb-francisco` | 12 | 9214 | 5000 | 6.2s | 114 MB | 0.9727 |
| `france-9` (`" contrasts"`) | 9 | 7148 | 5000 | 5.2s | 110 MB | 0.1807 |
| `france-6` (`" a"`) | 6 | 4412 | 4412 | 2.2s | 84 MB | 0.2051 |

Pruning for display is separate from computation, as the spec requires:
`prune_graph(node_threshold=0.8, edge_threshold=0.98)` reduces `ggb-francisco` to
1284 nodes and 219,391 links. **Omitted edges are not zero effects** — the dense
matrix retains them, and error nodes survive pruning.

### 5.2 A graph that works

Target `" Francisco"` after `"...the city of San"`. The three highest-influence
feature nodes all sit at position 11 (`" San"`) in late layers: L21/6271,
L22/1503, L24/2455. Their activation examples (§5.3) are unambiguous "San +
city-name" detectors. The mechanism the graph proposes — *this is the "San" of a
"San <city>" place name, so predict a city* — is exactly what the evidence shows,
and ablating those features moves the output the predicted direction (§6).

### 5.3 Feature evidence: examples yes, labels no

Evidence is **not** in the graph. `clerp` (the label field) is empty in every
node of every graph file produced here, and `transcoder_list` is `[]`. The viewer
fetches evidence at display time from the transcoder repo on HuggingFace:
`features/index.json.gz` gives a byte range into `features/layer_<L>.bin`, and
that range holds a 4-byte little-endian length followed by gzipped JSON. Feature
indices are **cantor-paired** `(layer, feature_idx)` in the frontend.

A fetched record contains `index`, `examples_quantiles`, `top_logits`,
`bottom_logits`, `act_min`, `act_max`. **There is no label field.** These features
have never been auto-interpreted. So:

- CIR-03 must show feature id + examples with an **explicit unlabeled state**. It
  cannot show a name, because none exists.
- mechlens's existing Neuronpedia labels are for **Gemma Scope residual SAEs**
  and cannot be joined to these by layer/index. The spec already forbids this;
  the absence of any label here removes the temptation.
- This is an obvious candidate for mechlens's existing auto-interp machinery
  (`docs/auto-interp.md`) as follow-up work, but it is not a release dependency.

L21/6271, activation range 7.75–139.0:

```
act 139.00 on 'San'   ...'has been identified⏎San Rafael, CA – Five confirmed and one presumptive case'
act 137.00 on 'San'   ...'⏎San Diego State University researchers have discovered that ancient herpes-like'
act 137.00 on ' San'  ...'6,000 wireless sensors from the San Francisco company'
```

L24/2455, range 6.375–163.0, is the same family and its `top_logits` include
`' Leandro'` — "San Leandro". That is the graph's proposed mechanism showing up
directly in the evidence.

`top_logits` are noisy in general, though. For L21/6271 they are
`['BrowserModule', ' MainAxisSize', 'ExecuteReader', 'Hochspringen', 'MessageOf']`
— meaningless for a feature whose examples are unmistakable. **A UI should not
present `top_logits` as an explanation.**

### 5.4 A graph that does not work

Required by the spec, and easy to find. Target `" contrasts"` in
`"The capital of France is a city of| contrasts"` at p=0.18 — a low-confidence
continuation of a completion that already went sideways (the model did not say
"Paris").

One of its top-influence features is genuinely interpretable: **L18/14811**, whose
`top_logits` are `[' conflicts', ' Conflict', ' contradictions', ' conflict',
' contradiction']` and whose examples fire on *"leads you to something you know is
false"*, *"in the event of conflict"*, *"resolve the apparent discrepancy"*. A
contradiction/opposition feature driving `" contrasts"` is a real explanation.

But **L25/11867**, ranked among the top contributors, is not interpretable at all.
Its strongest examples peak on `' dreadful'`, `' the'`, `','`, `' a'` with no
common theme, and its `top_logits` are French morphological fragments
(`' réguli'`, `' supérieurs'`, `' démocr'`). And it is the feature whose
intervention **contradicts the graph** (§6.2). A workspace that renders this node
identically to L18/14811 would be presenting noise with the same authority as
signal.

## 6. Intervention semantics

### 6.1 These are original-model tests

`feature_intervention` computes `value - current_activation`, multiplies by the
transcoder's decoder vector, and **adds that delta into the original model's
forward pass**. It does not swap the model for a transcoder reconstruction. This
is structurally the same move `backend/app/experiments.py::measure_feature`
already makes with `sae.W_dec[f]`, which is why the spec's requirement to
"establish a supported original-model test" is satisfied rather than blocked.

**Zero controls are exact**: setting a feature to the value it already has
reproduces baseline with `max_error == 0.000e+00` on every run, across all six
feature/setting combinations tested. That is better than the `1e-5` threshold
`experiments.py` uses in fp32, and it holds because the computed delta is
identically zero rather than merely small.

One readback subtlety, which the scripts now report honestly: **the activation at
the intervened site does not change.** The delta is added downstream of the
transcoder encoder, and that encoder's input at that layer is untouched. Evidence
that anything happened is (a) the output scores and (b) feature activations at
*later* layers. Reporting the site's readback as "after" would imply a write that
never occurred; `intervene.py` reports `activation_at_site` plus
`downstream_features_changed` instead. CIR-02's adapter must do the same, and the
same caveat applies to `experiments.py`'s existing `activation_after` field.

### 6.2 Results

`ggb-francisco`, target `" Francisco"`, baseline p = 0.972987:

| feature | mode | p | Δp | downstream changed |
| --- | --- | --- | --- | --- |
| L21/6271 | ablation | 0.968654 | −0.004333 | 154 |
| L21/6271 | ablation, constrained | 0.956482 | −0.016505 | 143 |
| L22/1503 | ablation | 0.968554 | −0.004433 | 141 |
| L22/1503 | ablation, constrained | 0.946606 | −0.026381 | 139 |
| L24/2455 | ablation | 0.954174 | −0.018813 | 60 |
| L24/2455 | ablation, constrained | 0.940460 | −0.032526 | 58 |

All six move the output down, as the graph predicts. The magnitudes are small
because the target sits at p=0.97 and is heavily over-determined — removing one
feature out of thousands barely dents a saturated prediction. **That is a real
limit on what a single-feature test can demonstrate on confident predictions**,
and CIR-03 should not present a −0.004 change as a dramatic confirmation.

Constraining all layers (`constrained_layers=range(0, 26)`, i.e. direct effect
with frozen attention and LayerNorm, no propagation) consistently produces
**larger** effects than letting effects propagate. CIR-02 must therefore pick one
default and name it in the record, because the two answer different questions.

`france-9`, target `" contrasts"`, baseline p = 0.180673 — where the disagreement
lives:

| feature | mode | p | Δp |
| --- | --- | --- | --- |
| L18/14811 | ablation | 0.143252 | −0.037420 |
| L18/14811 | ablation, constrained | 0.148356 | −0.032317 |
| L25/11867 | ablation | 0.194136 | **+0.013464** |
| L25/11867 | ablation, constrained | 0.152680 | −0.027993 |

**L25/11867 is listed as a positive contributor by the attribution graph, and
ablating it raises the target probability.** Under the constrained setting the
sign flips back. This is exactly the kind of disagreement the spec says must be
preserved rather than smoothed over: attribution is an approximation, an
intervention tests one operation, and the two can disagree. It is also the
uninterpretable feature from §5.4 — a useful correlation, but a single case, not
a rule.

### 6.3 Unsupported operations

- **No steered-trace attribution.** Not attempted, per the spec. mechlens's
  attribution pass already refuses steered traces.
- **No residual-SAE operation on a transcoder node.** Different feature space,
  different hooks; no validated mapping exists and none was built.
- **No "test this path".** An intervention tests its own operation. Nothing here
  validates a chain of edges or establishes mediation.
- **Generation-time intervention** (`feature_intervention_generate`) exists in the
  library but was not exercised; the release scope is a single fixed prefix.

## 7. Method and operational properties

### 7.1 Progress

`attribute()` has **no progress callback**. It exposes `verbose` and
`update_interval`, writes phase lines to the `attribution` stdlib logger, and
renders a tqdm bar over feature nodes. Its five phases are: precompute, forward
pass, input vectors, logit attributions, feature attributions.

A job runner can get **phase-level** progress by attaching a `logging.Handler` to
that logger. Within-phase counts would require redirecting tqdm. This does not
fit mechlens's existing monotonic `JobPhase` enum (`generating`/`sae`/`lens`),
which the plan already anticipated.

### 7.2 Cancellation — there is none

No cancel parameter, no callback, no cooperative check. Once `attribute()` is
entered it runs to completion. The only boundaries a caller controls are before
the call and after it returns.

Given the runtimes in §8 (2–47s), the honest engineering answer is to **report
attribution as uninterruptible** and keep the granularity coarse, rather than
pretend to a cancel that silently does nothing. If genuine cancellation is
needed, the credible route is running attribution in a separate process that can
be killed — which the dependency conflict (§9) may force anyway.

### 7.3 Cleanup

`TransformerLensReplacementModel.__del__` raises a `TypeError` during interpreter
shutdown (`reset_hooks` → `isinstance` against a torch Parameter shim). Python
swallows it as `Exception ignored in __del__`, so it is cosmetic — but it means
**hook cleanup must be explicit in the caller** and cannot be left to garbage
collection. mechlens's existing pattern (`model.hooks(fwd_hooks=...)` as a
context manager, asserted by `test_hook_removed_even_if_encoding_fails`) is the
right shape to keep.

### 7.4 Serialization

Two formats, both large:

- `Graph.to_pt()` / `Graph.from_pt()` — the full dense adjacency. 84–114 MB for
  6–12 token prefixes here, up to 342 MB at 64 tokens / 7500 nodes.
- `create_graph_files()` — pruned JSON for the viewer. 19–43 MB per graph.

Both are far too large to hand to a browser whole. §10 covers the consequence.

## 8. Resource costs

From `out/bench.json`. Model + transcoders load once: **10.5s, 12.3 GB peak**.

| prefix | 2000 nodes | 5000 nodes | 7500 nodes | peak mem |
| --- | --- | --- | --- | --- |
| 8 tok | 2.43s / 20 MB | 4.01s / 109 MB | 2.61s / 171 MB | 12.5 GB |
| 16 tok | 3.99s / 24 MB | 7.88s / 118 MB | 11.32s / 252 MB | 13.8 GB |
| 32 tok | 7.88s / 34 MB | 15.91s / 138 MB | 22.88s / 281 MB | 17.2 GB |
| 64 tok | 16.25s / 57 MB | 32.90s / 183 MB | 47.00s / 342 MB | 24.5 GB |

(time / `.pt` size. The 8-token/7500 cell is faster than 8/5000 because only 6323
features were active, so the node cap never bound.)

Time scales roughly linearly in both prefix length and node budget. Peak memory
depends on prefix length only — the node budget does not move it.

**Coexistence is not a problem on this box.** mechlens's own footprint, measured
in its venv: gemma-2-2b alone **6.47 GB**, plus one 16k Gemma Scope SAE **6.77 GB**.
Attribution peaks at 24.5 GB in the worst cell. Against 128.5 GB of unified
memory, the two can sit side by side in separate processes with room to spare.
On a discrete 16 GB GPU they could not, which is worth recording as a portability
limit rather than a local one.

**Suggested defaults for CIR-02**, from these numbers: prefix ≤ 32 tokens,
`max_feature_nodes` 5000 — ~16s and 17 GB, with graphs that are rich enough to be
interesting. Treat 64 tokens as the supported ceiling.

## 9. The dependency question — settled by experiment

This section originally presented three options and recommended a sidecar
process. That recommendation was **wrong, and the experiment that corrected it is
recorded here** because the reasoning behind the error is easy to repeat.

The apparent conflict:

```
circuit-tracer  requires  transformers >=4.56, <=4.57.3
mechlens venv   has       transformers 5.16.1
```

The error was reading the *installed* version as a requirement. mechlens's
`backend/requirements.txt` declares `transformers>=4.43` and
`transformer-lens>=2.0` — floors with no ceiling. It has never claimed to need
5.x. So the real question was never "which one do we give up", it was "does
mechlens actually run on 4.57.3", and that is measurable.

### What was measured

A clean venv was built from `backend/requirements.txt` **plus** `circuit-tracer`
at the pinned commit. It resolved with no conflicts:

| | resolved |
| --- | --- |
| transformers | 4.57.3 |
| transformer-lens | 3.2.1 |
| sae-lens | **6.49.1** — the version `requirements.txt` actually pins |
| torch | 2.14.0 |
| circuit-tracer | `7f66876` |

Note that sae-lens resolves to the *declared* 6.49.1, where the current venv has
drifted to 6.50.0. The unified environment is closer to what the repo says than
the environment the repo is being developed in.

`pip check` reports no conflicts. Running mechlens's backend suite in it:

```
current venv    21 failed, 346 passed, 1 skipped
unified venv    21 failed, 346 passed, 1 skipped
```

The failure *sets* were diffed, not just the counts: **identical, test for test.**
All 21 are the pre-existing failures of §11. Zero regressions. Attribution also
runs unchanged in that environment — the Golden Gate graph reproduces at 6.3s
with the same target probability 0.97265625 and the same 5000 selected features.

### The one real coupling

Exactly one thing blocked it, and it is not a version constraint. mechlens's
`backend/app/passes/lens.py:42` does:

```python
from transformer_lens.utilities.activation_functions import apply_softcap
```

That helper exists in transformer-lens 3.8.1 but not 3.2.1, and **transformer-lens
3.8.1 requires `transformers>=5.9.0`** — that, not circuit-tracer, is what was
really pulling mechlens onto transformers 5.x. The function is three lines of
Gemma logit soft-capping:

```python
def apply_softcap(x, cap):
    if cap is None or cap <= 0:
        return x
    return cap * torch.tanh(x / cap)
```

Inlining it into mechlens removes the coupling. The 346/21 result above was
produced with exactly that shim in place, so the number is measured, not
projected.

### Recommendation

**One environment.** Concretely:

1. Inline `apply_softcap` (and `softcap_enabled`) into mechlens, alongside its
   existing use in `passes/lens.py`. Keep `test_lens_pass.py`'s direct test,
   pointed at the local copy.
2. In `backend/requirements.txt`, add ceilings that reflect reality:
   `transformers>=4.56,<=4.57.3` and `transformer-lens>=2.16,<3.3`. These are
   tighter than today's declarations but they describe the combination that is
   actually tested.
3. Add `circuit-tracer` pinned by commit.

What this costs, stated plainly:

- **transformer-lens is frozen at 3.2.x** until circuit-tracer relaxes its
  transformers ceiling. mechlens's suite passes identically on 3.2.1, so there is
  no functional loss today, but upstream TransformerLens updates are off the
  table meanwhile.
- **Attribution runs in-process and cannot be cancelled** (§7.2). A sidecar would
  have bought real cancellation by making the work killable. Given measured
  runtimes of 2–47s, reporting attribution as uninterruptible is the honest and
  adequate answer, but it is a genuine trade and not a free win.
- Memory now holds both mechlens's model and the replacement model in one
  process: ~6.5 GB + ~24.5 GB peak. Comfortable against 128.5 GB here; not
  viable on a 16 GB discrete GPU.

If either of the first two becomes painful later, the sidecar remains available
as a refactor. It is not needed to start, and starting with it would add a
process boundary and a serialization format to buy something the measurements say
is not yet a problem.

## 10. Handoff

### To CIR-02 — adapter contract

Start by applying §9: inline `apply_softcap`, add the ceilings, pin
`circuit-tracer`. That is the whole environment story — there is no sidecar to
build.

**Inputs to validate before any compute:** model identity and revision;
transcoder repo + revision sha; that the source trace is complete; exact prefix
token ids; that the target is a `generated` step and `len(prefix) == target
position`; prefix within the supported ceiling; `max_feature_nodes` within
budget.

**Store per analysis:** the immutable source snapshot (prefix ids, target id,
target position, trace id); full provenance as in §2 with unavailable fields as
`null`; `active_features` as `(layer, pos, feature_idx)`; the adjacency matrix
with its `[features, errors, embeds, logits]` ordering and node counts; pruning
thresholds *separately* from computation limits; `logit_probabilities`;
`scan_name` as the evidence-identity key. Annotations go in a separate table
from measured data.

**Do not store** what it can recompute cheaply, but **do** solve the size
problem: 114 MB dense `.pt` per 12-token graph does not scale. Store the pruned
graph plus enough of the dense matrix to support bounded expansion, and make
expansion beyond computed coverage an explicit new run.

**Carry forward from this report:** ULP-based logit tolerances (§4); exact-zero
control expectation (§6.1); `activation_at_site` never changes (§6.1); one named
default for `constrained_layers` (§6.2); phase-level progress only (§7.1);
attribution is uninterruptible in-process (§7.2); explicit hook cleanup (§7.3).

**Refactor needed in mechlens:** `experiments.py::measure_feature` takes
`prompt: str` and re-tokenizes it. §4 shows why that is unsafe for replaying a
saved trace. It needs a token-ids entry point. It is also CLI-only today — no
HTTP route exposes it — so "reuse the existing experiment engine" means
promoting it to a service, not wiring to something that already exists.

Also add a **model revision field to the `Trace` schema** (§2).

### To CIR-03 — viewer decision

**Recommendation: build on `@xyflow/react`, do not adapt the upstream viewer.**

The bundled viewer is excellent for CIR-01 and was used throughout — it serves,
it renders every graph, and it wires up feature evidence for free. But it is a
standalone D3 application with its own routing, its own data-fetch layer that
goes directly to HuggingFace and a CloudFront bucket, and its own state model. It
does not compose with App-owned navigation, trace-tagged selection, or the return
path to the source token, all of which CIR-03 requires. React Flow is already in
`frontend/package.json` (currently used only by the unused
`components/ai-elements/` kit).

**Constraints CIR-03 inherits:**

- **Payload.** 22–43 MB of pruned JSON, 1284 nodes / 219,391 links for a
  12-token prefix. The frontend needs a bounded, server-paged graph read; it
  cannot fetch a graph file.
- **Unlabeled features.** Show id + examples + an explicit unlabeled state
  (§5.3). Never borrow a residual-SAE label.
- **Evidence fetching is a network call** to the transcoder repo, byte-ranged,
  gzipped, cantor-paired index. It must be explicit in backend configuration and
  provenance, and saved graphs must stay viewable when it is unavailable.
- **Don't trust `top_logits` as an explanation** (§5.3).
- **Position 0 has no features by design** (§4).
- **Error nodes and coverage stay visible.** `n_layers × n_pos` of them.
- **Small effects are the normal case** on confident predictions (§6.2). Show
  the numbers; don't dramatize them.

## 11. Limits of this report

- One model, one transcoder set, one machine. Nothing here says anything about
  Gemma-3, cross-layer transcoders, or a 16 GB discrete GPU.
- Three traced targets plus one smoke prompt. Enough to answer feasibility, not
  enough to characterise graph quality in general — that is CIR-04's job.
- Six intervention runs on five features. The single attribution/intervention
  disagreement in §6.2 is a real finding but one data point.
- Interpretability judgements in §5 are the author's reading of activation
  examples, not a measured quantity.
- Numerical agreement was established for one prefix. CIR-02 should run the
  ULP check across many prefixes as an automated test.
- **The mechlens backend suite has 21 pre-existing failures** on `master`
  (`test_service_app.py` atlas/layout and live-trace tests, and
  `test_training.py::test_real_training_roundtrip_and_frozen_model`; 346 pass).
  Confirmed pre-existing by re-running with `prototypes/` removed from the repo
  entirely. Unrelated to this work, untouched by it, and not fixed here — but
  CIR-02 will want a green suite to build against.
