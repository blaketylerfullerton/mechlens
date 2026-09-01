## Phase 3 — Neuronpedia labels

- Neuronpedia has a public API for features, explanations, and activations, with docs at neuronpedia.org/api-doc — and full S3 data exports so you don't hammer their server. For your use case, **download the explanation exports for gemma-scope-2b-pt-res and load them into SQLite**. You'll be requesting thousands of labels per trace; the API is for one-offs, the export is for your lookup table.
- Build a `labels` module: `get_label(layer, feature_idx) → {text, score, neuronpedia_url}`. Fall back to the live API for cache misses, and always include the Neuronpedia URL so users can click through to raw activating examples (your "surface the good ones, link the evidence" plan).
- Feature IDs on Neuronpedia follow the format MODEL_ID@SAE_ID:FEATURE_INDEX — keep that mapping exact, including which SAE width/variant you loaded, or your labels will be for the wrong dictionary.

**Done when:** the trace's top features come back with human-readable labels and links.

---

### The ID mapping is already in SAELens — don't hand-build it

The `MODEL_ID@SAE_ID:FEATURE_INDEX` warning above is real, but the mapping does not have to be
reconstructed by hand. SAELens' `pretrained_saes.yaml` carries a `neuronpedia` field per SAE, and it
survives onto the loaded object as `sae.cfg.metadata.neuronpedia_id`:

    layer_20/width_16k/canonical  ->  gemma-2-2b/20-gemmascope-res-16k   (path: average_l0_71)

Read that field. The canonical L0 pick and the Neuronpedia source set come from the same registry
entry, so they cannot drift apart, and the mapping stays correct if the width ever changes to 65k or
to a non-canonical variant. Without loading any weights:

    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    info = get_pretrained_saes_directory()["gemma-scope-2b-pt-res-canonical"]
    info.neuronpedia_id["layer_20/width_16k/canonical"]

### The API, as it actually responds

`https://www.neuronpedia.org/api/feature/{modelId}/{sourceSet}/{index}` — GET, **no API key needed**,
~1.2MB per feature. The two fields that matter:

- `explanations[]` — `description`, `explanationModelName`, and `scores[]`. Note the plural: a
  feature can carry several explanations from different explainers with different scores, so the
  labels module needs a deterministic pick rule (prefer one explainer, tie-break on score) and should
  record which one it chose. Otherwise labels silently change character between rows.
- `activations[]` — each with `tokens`, per-token `values`, `maxValue`, `maxValueTokenIndex`. This is
  ground truth for verifying the mapping, and it is what makes `scripts/verify_neuronpedia_mapping.py`
  possible.

`maxActApprox` at the top level gives the feature's expected activation scale, which is the number
that catches a *different L0 variant* of the right layer — that case still correlates well but does
not match in magnitude.

### Verify before importing the export

`backend/scripts/verify_neuronpedia_mapping.py` takes a few features, pulls their top activating
example, re-runs those exact tokens through our model and SAE, and compares. Three checks, which
fail differently:

- **argmax position** vs `maxValueTokenIndex` — a mismatch means the wrong dictionary, full stop.
- **correlation** across tokens, ~1.0 — confirms it is the same feature and not a coincidence.
- **scale** vs `maxValue` — catches the same layer at a different L0.

Convert `tokens` to ids with `convert_tokens_to_ids`, never by re-tokenizing the joined string:
re-tokenizing shifts the boundaries and you end up comparing misaligned sequences. Neuronpedia's
chunks come from the training corpus and may not carry BOS while our capture always does, so the
script tries both alignments and reports which won.

**Verified — 10/10, mapping confirmed.** Two features each at layers 0, 5, 12, 20, 25, sampled
uniformly from the dictionary:

    10/10 features matched  (median corr +1.000, median scale 1.00)

Every one matched on argmax, with correlation 0.998-1.000 and scale 0.96-1.01. `gemma-2-2b/{layer}-
gemmascope-res-16k` is the right label source for `gemma-scope-2b-pt-res-canonical` at 16k. Safe to
import the export.

**BOS matters here too, and it nearly produced a false alarm.** Every one of the ten matched on the
BOS-prepended run. Feeding Neuronpedia's chunk without a BOS still correlates at 1.000 — dropping it
*rescales* the feature without reshaping it — but the peak comes in 30-45% low, which trips the scale
check and reads as a mapping failure. Two features looked like mismatches for exactly this reason
before the alignment picker was fixed to let correlation decide only when it actually separates the
two runs. Worth remembering whenever our activations get compared against anyone else's: Gemma
without its attention sink is a quantitatively different model.

### Schema: labels go in a side table, not on Feature

`Feature.label` exists as a placeholder and nothing writes it. Widening it to hold
`{text, score, url}` is the obvious move and the wrong one — measured on the traces on disk:

| trace | feature entries | distinct (layer, feature) | dup |
|---|---|---|---|
| golden-gate | 12,800 | 6,234 | 2.1x |
| paris | 6,588 | 3,562 | 1.8x |

golden-gate.json is already 1.9MB; inlining ~80 chars of text plus a ~70-char URL on every entry
roughly doubles it, and half of that is the same string repeated. Instead, one trace-level map:

    labels: dict[str, FeatureLabel]   # key "20/12345" -> {text, score, explainer}

Still purely additive, `LayerState` untouched, and better for the frontend: one lookup map it can
consult from a feature list, a tooltip, or an attribution graph, rather than reading the label off
whichever copy of the feature it happens to be rendering. Drop the unused `Feature.label`. Bump
`SCHEMA_VERSION` to 1.2.

**Do not store the URL.** It is a pure function of `(model_id, source_set, feature_idx)`. Put the
mapping and the URL template in `PassRecord.params` once and let the frontend build links — then a
wrong mapping is a one-field fix instead of a rewrite of every trace.

### The export chooses the explainer for you — and it is not consistent

Downloaded and tabulated the real export before writing the pick rule, which changed the answer.

`https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{modelId}/{sourceSet}/explanations/`
— same source-set id as the API, 64 gzipped JSONL batches per layer, ~22MB. The old
`/api/explanation/export` endpoint is gone and 400s with a pointer to this bucket.

For layer 20: **16,336 of 16,384 features (99.7%), exactly one explanation each.** So there is no
per-feature choice to make on the bulk path — the export already made it. What there *is*, is a
split across layers:

| layers | explainer | type |
|---|---|---|
| 16, 18, 20, 22, 24 | `gemini-2.5-flash-lite` | `np_acts-logits-general` |
| the other 21 | `gpt-4o-mini` | `oai_token-act-pair` |

So **pinning `gpt-4o-mini` is not an option**: the export simply has no gpt-4o-mini row for those five
layers, and backfilling them from the API is 5 x 16,384 requests at ~1.2MB each. Take what the export
gives, and record `explainer` on every label so the split stays visible. The preference ladder is
therefore only for the API fallback, where one feature can come back with several explanations and
the choice has to be deterministic.

The two explainers write differently, and the difference is not cosmetic — feature 12082 at layer 20
is `"dog walking accessories"` in the export and `"References to dogs as pets, often using possessive
pronouns..."` from the API. Same feature (which is itself a nice independent confirmation of the
mapping), different specificity. **Anything that pools label text across layers — clustering,
embedding, a UMAP layout — will see layers 16/18/20/22/24 as their own region for reasons that have
nothing to do with the model.** Check for that artifact before trusting a layout: if those five
layers separate more than adjacent layers do, re-embed the text yourself with one model instead of
using the shipped vectors.

### Scores are not a selection criterion

Sampling 15 features across 5 layers via the API turned up **one score across 21 explanations**, and
the export carries none at all. Ranking labels by score would be ranking by NULL. Kept as a nullable
column for API-sourced rows; never used to order anything across explainers, since a `recall_alt`
value from one explainer is not comparable to another type's.

### Embeddings ship with the export — with a serialisation trap

Every export row carries a 256-dim L2-normalised embedding of the explanation text, which is most of
what a feature map needs. One catch: the older gpt-4o-mini rows store it as a **JSON string**, the
newer gemini rows as a real list. Parse both or you get a 3,109-character "vector".

They cost real space: one layer with embeddings is a 76MB SQLite file (~2GB for all 26), against
~2MB without. Make them opt-in in the importer.

Worth considering for the layout itself: `W_dec` cosine similarity is the model's own geometry rather
than a caption's, and it is immune to the explainer split entirely. Labels would annotate positions
rather than determine them.

### The SQLite DB is per-SAE-set, and it is a cache with three states

Every gemma-2-2b/16k trace shares the same ~426k explanations (26 x 16384), so the DB belongs outside
`traces/` — `backend/data/neuronpedia.db`, gitignored, keyed by `(source_set, feature_idx)`. The
trace JSON carries only the subset it needs, so it stays self-contained for the browser.

Coverage is not 100%, so the cache must distinguish three states, not two: *has explanation*,
*looked up and there is none*, and *never looked up*. Without a `fetched_at` column the live-API
fallback re-requests every unexplained feature on every run — thousands of requests to re-learn
nothing.

### Build order

1. `scripts/verify_neuronpedia_mapping.py` — confirm the mapping empirically.
2. `app/labels.py` — SQLite store plus `get_label(layer, feature_idx)`.
3. `scripts/import_neuronpedia.py` — bulk-load the S3 explanation export.
4. `app/passes/labels.py` — the pass, wired as `enrich <trace> --labels`.

The pass ignores the `residuals` argument the `Pass` protocol hands it. That is fine; do not widen
the protocol for one pass. Keep it offline by default so tests never touch the network — DB path
injectable, API fallback behind `--fetch-missing`.

(Note: `schema.py`'s header docstring calls `logit_lens` "phase 3". Phase 3 is labels; the logit lens
is unclaimed.)


---

## Built so far

- `app/sae_cache.py` — `neuronpedia_id(layer, width)`, the single place our SAE maps to Neuronpedia's
  `(model_id, source_set)`, read from SAELens' registry.
- `scripts/verify_neuronpedia_mapping.py` — the empirical check above. **10/10 features matched.**
- `app/schema.py` — schema 1.2: `FeatureLabel`, `Trace.labels`, `label_key()`, `Trace.label()`;
  `Feature.label` removed.
- `app/labels.py` — `LabelStore` (SQLite, three-state, chunked batch reads, optional embeddings,
  offline by default with a capped API fallback) and `pick_explanation()`.
- `tests/test_labels.py` — 12 tests, all offline.

Verified end to end: the real layer-20 export loads in 0.7s and answers
`get_many(20, range(16384))` with 16,336 labels.

Still to do: `scripts/import_neuronpedia.py` (step 3) and `app/passes/labels.py` wired as
`enrich <trace> --labels` (step 4).
