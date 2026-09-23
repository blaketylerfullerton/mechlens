# mechlens

<img width="1710" height="817" alt="Screenshot 2026-09-12 at 8 17 38 PM" src="https://github.com/user-attachments/assets/bf3e1e3f-b466-4ca6-b943-e92966416c8d" />

Look inside a language model on your own machine.
 
Under the hood: `gemma-2-2b` under TransformerLens, generated token by token
with the residual stream captured at every layer, then enrichment passes over
the saved trace — Gemma Scope SAE features, Neuronpedia labels, the logit lens,
and attribution. A FastAPI service sits in front of all of it, so tracing and
feature steering are reachable over HTTP.

There is a second half: a training workspace that trains your *own* SAE on one
layer, measures how good it is, and labels its features with a local LLM. See
[Training your own SAE](#training-your-own-sae).

**The inference backend is now installable from this repository:** `pip install -e .`, then `mechlens serve`. The browser UI lives in the Mechlens Cloud repository and runs separately. `mechlens serve --cloud <url>` pairs the GPU with a Mechlens Cloud workspace; the GPU dials out, so no tunnel or open port is needed. See [Serving the inference API](#serving-the-inference-api).


## Where this is going

The end state is a local-first tool you install with one command and point at a
model: a browser opens and you are looking inside it. Ollama's shape, for
interpretability. Today this is a research repo — a venv, a 462MB import script,
a 3-minute atlas build and two dev servers — and closing that gap is worth more
than any feature below it.

The friction a newcomer hits, in the order they hit it:

| | closed by |
| --- | --- |
| Python env, torch/CUDA versions, two servers to start | a published wheel with the built frontend inside it |
| a 462MB label DB produced by an import script | shipping it as a release asset, pulled with a progress bar |
| a 3-minute atlas build | shipping the 560KB asset the same way |
| `gemma-2-2b` hardwired in `model_cache.py` | a small model registry |
| an HF token and a licence click before anything runs | an ungated default model — `gpt2-small` has public SAEs |

Roughly in order:

1. **A hosted static demo.** The frontend renders a saved trace plus the atlas
   asset, and needs no model, no GPU and no backend — so a handful of prebuilt
   traces on a static host demonstrates the whole view. Nobody installs a tool
   they have not seen run.
2. **One command to serve it.** A single package with the built frontend inside
   the wheel and FastAPI serving both halves. No node, no vite, no second port.
3. **A pullable bundle.** The unit is not a model; it is a model *plus* its
   SAEs, its labels and its atlas. One command should fetch all four.
4. **Docker**, for Linux/CUDA users and servers — second-class on purpose. The
   development machine here is aarch64 Blackwell, which is nobody else's
   platform, and a CUDA container gives a Mac user nothing.
5. **Bring your own SAE** — load a SAELens-format file against a supported
   model, auto-interp labels optional. Covers most of "my own model" for almost
   nothing.
6. **Training beyond one layer.** The training workspace
   ([below](#training-your-own-sae)) trains one layer at a time and inspects
   the result in its own page. What it cannot do is hand that SAE to the trace
   viewer — `sae_cache.py` is hardwired to the Gemma Scope release — so a
   trained dictionary gets no atlas position, no attribution and no
   multi-layer view. Closing that is the same work as item 5. See the
   [training plan](docs/sae-training.md) and
   [implementation tickets](docs/sae-training-tickets.md).


**Where the Ollama analogy breaks, so it is not promised.** Ollama ships a
static binary because llama.cpp is C++; torch alone is 2–3GB of wheels and no
packaging changes that. And `ollama run` is inference, whereas training an SAE
is training — hours of GPU time, not seconds, at a quality below Gemma Scope.
A run that reports its explained variance, L0 and dead-feature count is worth
shipping; one that implies it matched a lab's is not, in a project built around
saying how much to trust what you are looking at.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # gemma is gated: accept the license at hf.co/google/gemma-2-2b
npm --prefix ../mechlens-cloud/frontend ci
```

The CLI below needs only the Python half. The browser interface needs both.

## Serving the inference API

From the repository root, in your Python environment:

```bash
pip install -e .
mechlens download   # once per machine: gemma-2-2b (~5GB) + Gemma Scope SAEs (~8GB)
mechlens serve
```

### No automatic downloads

Mechlens never downloads model weights, SAEs, or datasets on its own. Every
command except `mechlens download` runs with Hugging Face in offline mode, so
it only uses what is already in `~/.cache/huggingface`. If something is
missing, the server does not quietly fetch gigabytes. It says what is missing
and tells you to run `mechlens download`.

- `mechlens download` fetches gemma-2-2b and the 26 Gemma Scope SAEs.
  `--no-saes` fetches only the model. Accept the license at
  hf.co/google/gemma-2-2b and run `huggingface-cli login` first.
- Circuit tracing and SAE training need files `mechlens download` does not
  fetch (transcoders, training data). To let those download, run with
  `MECHLENS_ALLOW_DOWNLOADS=1 mechlens serve`.

For an existing environment that already has `backend/requirements.txt` installed, use `pip install --no-deps -e .` to register the command without resolving dependencies again. A fresh install uses the existing backend requirements, including the pinned Git dependency for circuit tracing; it requires Git and can download substantial model-runtime dependencies. This package is not published to PyPI yet.

The API binds to `127.0.0.1:8000`. It starts listening while the model warms up; `GET /health` reports `loading`, `ready`, or `error`. Stop with Ctrl+C. The viewer source lives in `mechlens-cloud/frontend`. Start it with `npm --prefix ../mechlens-cloud/frontend run dev:local` to use the local UI.

```bash
mechlens serve --help
mechlens serve --port 8001 --data-dir /path/to/persistent/mechlens
mechlens trace --help
```

Editable installs reuse `backend/data` when it exists. Otherwise `serve` defaults to `~/.local/share/mechlens`. `--data-dir` or `MECHLENS_DATA_DIR` selects the labels database (`neuronpedia.db`), training runs, and circuit analyses. `MECHLENS_TRAINING_DIR` can still override the training path. Assets are not bundled or downloaded by this command: copy/build the label database and atlas as described below to enable those features. Models are read from Hugging Face's normal cache location. The existing trace CLI retains its existing output-directory behavior.

### Connecting this GPU to Mechlens Cloud

```bash
mechlens serve --cloud https://your-mechlens-cloud.example.com
```

This prints an activation code. Sign in to the cloud app in a browser, open
`/activate`, and type the code. The GPU then serves your workspace until you
stop the process. The terminal and the cloud's Compute page both show what it
is doing: code accepted, model loading, ready, or the error that stopped it.

Nothing is configured on this machine and nothing listens on the internet. The
GPU makes ordinary outbound HTTPS requests to the cloud and asks it for work;
the cloud hands it browser requests, and it answers them from this API on
loopback. If this box can reach the cloud's URL, it works, including behind NAT,
strict firewalls, and managed notebooks that forbid tunnels. A box that only
reaches the internet through a proxy works too: `HTTPS_PROXY` is honored. If
the network drops or the cloud redeploys, the GPU reconnects by itself.

If you already have a public HTTPS URL for this API (for example the
port-forwarding URL a rented GPU host provides), the cloud can call it
directly instead:

```bash
mechlens serve --cloud https://your-mechlens-cloud.example.com \
               --tunnel-url https://abc123-8000.proxy.your-gpu-host.com
```

That URL must reach this API's port, and the port must not be exposed without
the token that `--cloud` generates.

### Authentication for remote connections

Set a secret through `MECHLENS_API_TOKEN` or a token file. Do not pass the secret as a CLI argument or commit it.

```bash
# Create once, keep private; reuse the file on subsequent starts.
(umask 077; python -c 'import secrets; print(secrets.token_urlsafe(32))' > /tmp/mechlens-api-token)
mechlens serve --token-file /tmp/mechlens-api-token
```

Requests then require `Authorization: Bearer <token>`, including `/health`, `/stats`, docs, and all inference routes. Use HTTPS through a tunnel for remote traffic. For a tunnel connector on the same machine, keep the default loopback bind. Explicit non-loopback binding (`--host 0.0.0.0`) requires a token. Existing direct Uvicorn launches also honor `MECHLENS_API_TOKEN`, but only `mechlens serve` enforces the bind-address check. The local frontend does not inject this token; the Cloud proxy attaches it server-side.

### Jupyter / GPU notebooks

Once these changes have been pushed, install the repository at the commit you want:

```python
%pip install "git+https://github.com/blaketylerfullerton/mechlens.git@<commit>"
```

Then run in another cell:

```python
!mechlens serve
```

The cell remains running and the process loads its own model on that machine. It does not reuse an existing notebook model. Interrupt the cell to stop it. This version starts the inference API only: it does not produce a public URL. A manually configured authenticated tunnel is the next integration step; device pairing and automatic tunnel creation will follow. Start with the local flow before testing a notebook provider.

## Running it

```bash
make dev     # backend on :8000, frontend on :5173, both in the background
make down    # stop them
```

`make dev` runs the single viewer source from `../mechlens-cloud/frontend` in
local mode (override the checkout with `CLOUD_FRONTEND=/path/to/frontend`).
Run `npm ci` there first. It waits for the backend to answer `/health` before
printing its URLs. The model loads in the
background, so `/trace` answers 503 until it is ready and the UI retries on its
own. Logs and PIDs live in `.dev/` (gitignored).

It refuses to start on top of a live run, which is not fussiness: a second vite
cannot have :5173, so it silently takes :5174 and you end up with the browser on
one port and a stale server on the other.

```bash
make site    # the landing page on :5180, foreground
```

`site/` is the older standalone landing-page project. The active Cloud landing
page and the single viewer source live in `mechlens-cloud/frontend`.
`frontend/README.md` points to the new viewer location.

## Use

Everything runs from `backend/`:

```bash
python -m app.cli trace -p "The Golden Gate Bridge is located in the city of" -n 20
python -m app.cli enrich traces/<id>.json --sae     # SAE features, no model load
python -m app.cli enrich traces/<id>.json --labels  # Neuronpedia labels, no network
python -m app.cli enrich traces/<id>.json --lens    # logit lens, loads the model
python -m app.cli enrich traces/<id>.json --attribution  # resid/attn/mlp edges, loads the model
python -m app.cli show traces/<id>.json --layer 20
python -m app.cli show traces/<id>.json --lens --token 8
python -m app.cli show traces/<id>.json --attribution --token 8
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels --lens --attribution   # or all at once
```

`--sae` and `--labels` never load gemma. `--lens` and `--attribution` are the
exceptions: `W_U` is 2304 x 256,000, too big to sit in a sidecar beside every
trace, and attention patterns and per-head values aren't in the sidecar at
all — so both need a fresh pass through the model. When you run them as part
of `trace` the model is already resident and it costs nothing extra to load.

Over HTTP, the same passes are opt-in per request. `POST /trace` takes
`passes: ["sae", "labels", "lens"]` and reports an `sae` phase while it runs —
that pass really does walk layers one at a time, unlike generation, so it is one
of the two phases that can honestly report a depth sweep. Asking for `labels`
without `sae` is a 422: the label pass labels the features the SAE pass records,
so the dependency is caught at the boundary rather than on the worker thread.
`sae_layers` restricts the pass to a subset, because 26 resident 16k SAEs come
to ~7.9GB — fine on a unified-memory box, not fine on a 16GB discrete GPU.

The atlas is built separately, once, and is not per-trace:

```bash
python scripts/import_neuronpedia.py --embeddings      # +explanation vectors, ~2GB
python scripts/build_feature_atlas.py                  # all 26 layers
python scripts/build_feature_atlas.py --dry-run        # measure and print, write nothing
```

`--dry-run` exists because an atlas nobody looked at the numbers for is not
worth building on: it reports kNN preservation, cluster coherence against its
baseline, and explainer influence without writing anything.

 

Three things about that output are worth knowing before you read one:

**It is checkable, and it is checked.** Layer 25 is `resid_post` of the last
block, which is precisely what `forward` hands to `ln_final` — so the last
layer's lens is not an approximation of the model's output, it *is* the model's
output, recomputed. `final_layer_agreement` on the pass record must be 1.0; on
gemma the probabilities come back bit-identical. Anything less means the softcap
or the norm is wrong.

One wrinkle that looks like a bug and is not: gemma's late-layer logits sit
around 28, where bf16 steps by 0.125, so the top two candidates sometimes land
on the *same* value and `topk` breaks the tie arbitrarily. Same distribution,
same entropy, different argmax. Those positions count as agreement and are
reported as `final_layer_argmax_ties` rather than absorbed;
`final_layer_max_entropy_delta` is the check that is blind to tie-breaking
altogether. See `phase4.md`.

**The softcap is easy to miss.** Gemma 2 caps logits at 30 with
`cap * tanh(x / cap)`, and TransformerLens applies it in `forward`, *after*
`unembed` — not inside it. Skip it and top-1 still looks about right while every
probability and entropy is quietly wrong. The pass calls the model's own
`apply_softcap` so the two cannot drift.

**Early layers echo, they do not predict.** Residuals near the embedding decode
back to the token already sitting at that position — on gemma, layers 0–9 read
the current token back at 60–85% confidence. That is why entropy is *not* the
headline number: it starts low for a reason that has nothing to do with the
model being sure of an answer. `top1_agreement_by_layer` is the honest
crystallisation curve, `echo_by_layer` says how much of its early portion to
discount, and `crossover_layer` — the first depth where the answer is already
in place for half the positions — is the one number to read.

## Attribution

For every layer and position, `resid_post = resid_pre + attn_out + mlp_out` —
not an approximation but the literal structure of the residual stream, so
decomposing it into edges needs no gradients or patching, just the model's
own intermediate tensors:

```
   L20  resid 340.1  mlp 117.1  attn [pos0=28.7, pos4=14.2, pos20=11.7]
   L21  resid 387.7  mlp 129.8  attn [pos29=27.5, pos0=26.8, pos30=17.1]
   L25  resid 624.1  mlp 236.3  attn [pos30=149.3, pos0=27.8, pos4=26.6]
```

**It is checkable, and it is checked.** `resid_pre + attn_out + mlp_out` should
equal the captured `resid_post` exactly, the same class of claim as the lens's
final-layer identity — `reconstruction_max_rel_gap` on the pass record is that
check. On gemma it reads **≤1.3e-2**, which looks less clean than the lens's
**0.0** until you notice it is a *max* over 26 layers x 31 positions of bf16
noise (mean **~0.004–0.005** per layer, flat with depth) rather than a single
number — a real bug shows up as the mean drifting, not the max's tail.

**Attention needs a per-source decomposition, not a per-head one.** Gemma 2's
attention output is normalised again before it joins the residual stream (the
"sandwich norm"), and that normalisation's scale and gain are computed from
the whole summed output, not per source — so it distributes over a per-source
split exactly, computed once from the total and applied identically to every
term. Grouped-query attention adds a second wrinkle: the cached value vectors
sit at 4 KV-head granularity, not gemma's 8 query heads, and have to be
expanded to line up with the attention pattern before they mean anything.

Edges are truncated to the top 8 source positions by weight, same as the SAE
pass truncates features — `attn_topk_coverage` says how much of the real
total that top-8 keeps, honestly: **90–98%** across the five saved traces,
lower on longer ones where attention has more positions to spread across.

`--labels` reads a local SQLite table built once from Neuronpedia's public export:

```bash
python scripts/import_neuronpedia.py               # 425,679 labels, ~50MB, ~50s
python scripts/verify_neuronpedia_mapping.py       # confirm our features are their features
```

```
layer 20, token 30 ' The'  (l0=80)
  #6631     84.75  ████████████  The Future
  #1370     81.21  ███████████   bridge and crossing
  #3124     29.10  ████          San Francisco, Oakland, Bay Area
  https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/6631
```

## The feature atlas

A trace says which features fired. It does not say where to *draw* them, and a
position has to come from somewhere that means something. So every feature gets
one, once, offline:

```bash
python scripts/build_feature_atlas.py --layers 0,5,10,16,20,25   # ~3 min, reads the label DB
python scripts/build_feature_atlas.py --source decoder           # the other representation, needs the SAEs
python scripts/build_feature_atlas.py --asset-only               # re-emit the idle asset from the DB, no rebuild
```

`--asset-only` exists because the idle asset is derived output — positions,
clusters and metrics all live in the label store once a build has run — and
UMAP is the expensive, stochastic step. Changing a serialised field must not
cost a rebuild, or the atlas risks losing the one property it is required to
have: that a feature is placed identically everywhere it is drawn.

```
atlas 16k-s0-731e35f436  (98,210 features, source=labels 256d)
  kNN preservation    0.292 (k=20, 4,000 sampled, ±0.180)
  clusters            32 (27 named, 5 unnamed)
  cluster coherence   0.465 vs baseline 0.233 (margin 0.232, need 0.150)
  explainer influence 0.026 (0 = areas are meaning, 1 = areas are the labeller)

  largest named areas:
     8,320  coh 0.441 (+0.208)  L5#8356    programming-related constructs and elements in code
     2,296  coh 0.552 (+0.319)  L25#12372  mathematical notations and expressions
     2,136  coh 0.498 (+0.266)  L0#14707   references to legal terminology and concepts
     1,518  coh 0.517 (+0.283)  L25#12912  numeric values and statistical references
```

**It is checkable, and it is checked.** Flattening to three dimensions throws
away a great deal — from 256 for the label source, from 2,304 for the decoder
one — and the whole value of the picture rests on how much survived, so that is
a measurement rather than an assumption. `knn_preservation` is the fraction of
a feature's 20 nearest neighbours in the source space still among its 20
nearest in the layout, and it is this pass's peer to the SAE's
`explained_variance` and the lens's `final_layer_agreement`. It reads **0.292**
on the shipping atlas, against **0.0002** for a random layout of the same
98,210 features. It is published in the interface, not buried in a build log.

**Near means similar; far means nothing.** UMAP preserves local neighbourhoods
and distorts everything larger, so "these two nodes are close" is a real
statement and "these two areas are far apart" is not one. The view therefore
ships with no axes, no coordinate readout and no distance scale: there is no
honest way to answer a question about distance here, so the question is not
offered.

**Two sources, because they answer different questions.** An atlas is projected
from one of two representations and records which:

| | what "near" means | preservation | areas |
| --- | --- | --- | --- |
| `labels` | an explainer *described* two features similarly | **0.292** | 32, 27 named |
| `decoder` | the model *writes* the two features similarly | 0.143 | 12, none named |

The label source is the default, and that reverses this project's original
decision. Positions were first taken from `W_dec` alone, on the grounds that a
layout over label text would inherit Neuronpedia's explainer split — layers 16,
18, 20, 22 and 24 carry `gemini-2.5-flash-lite` explanations and the other 21
`gpt-4o-mini` — and separate those five layers for a reason about the labeller
rather than about gemma. Sound reasoning; false premise. Measured on a pilot
chosen to contain both explainers, the influence is **0.026** — no meaningful
effect. Meanwhile the decoder layout cannot produce areas at all: HDBSCAN in the
full 2,304-dimensional space calls **99.1%** of features noise, and a forced
k-means partition earns **zero** names at k=24 or k=60. It is a continuum, not
an archipelago. Both atlases are built and kept; `app/atlas.py` and
`design.md` carry the numbers.

**A name has to be earned, against the right baseline.** An area is named after
its most central member's own label — a real feature says "URLs and web links",
and the interface can show which one — but only when the members' label
agreement beats a *random group of the same size* by a recorded margin. That
relative test replaced an absolute threshold, and the replacement was not
cosmetic: the first pilot measured a mean coherence of 0.281 against a baseline
of 0.232 and duly named 14 clusters that beat chance by 0.049. The names read
plausibly, which is what made it dangerous. Explanation embeddings from one
explainer share a similarity floor well above zero, so an absolute threshold
measures the floor and not any agreement. Under the relative gate the decoder
atlas names none of its 12 clusters, which is the correct answer.

**What the nodes leave out is stated.** A trace records the top 16 features per
layer out of an `l0` of ~78 that actually fired, so the interface reports both
rather than letting the lit nodes read as the complete set — the same habit as
`attn_topk_coverage`. BOS is excluded. And a node is a feature, not a place: no
named anatomical region computes any of this, and the shell is silhouette.

A trace is two files that travel together:

| file | what |
| --- | --- |
| `<id>.json` | the `Trace` document — tokens, next-token distributions, per-layer state |
| `<id>.residuals.npy` | `[n_tokens, n_layers, d_model]` float32 residual stream (`hook_resid_post`) |

Splitting them is what makes `enrich` cheap: a pass reads the tensor off disk
instead of regenerating, so iterating on pass code costs seconds. The SAE and
label passes never touch gemma at all; the lens needs `W_U` and attribution
needs the attention pattern and per-head values, so both pay the ~10s load.

Neither is committed. `backend/traces/` and `backend/data/` are gitignored — a
trace is a few MB of JSON plus a multi-MB `.npy`, and the label DB (66MB) is
rebuildable from the export in under a minute.

```python
from app.store import load
trace, residuals = load("traces/golden-gate.json")   # mmap=True to page it in lazily
residuals[7, 11]                                     # token 7, layer 11 -> [d_model]
trace.steps[7].layers[11].features                   # its top SAE features
trace.steps[7].layers[11].logit_lens.top_k[0]        # what it would answer here
trace.label(11, 4023)                                # what feature 4023 means
```

Labels live in `Trace.labels` keyed `"layer/index"`, not on each `Feature`: a
feature recurs ~2x per trace, so copying the text onto every occurrence would
roughly double the JSON for no added information.

## Training your own SAE

Everything above reads a *pretrained* dictionary: Gemma Scope's SAEs, labelled
by Neuronpedia. The Training page is the other direction — train one yourself,
on one layer, and find out how good it is.

`POST /training/runs` takes a `TrainingConfig` (`app/training/runner.py`) and
trains a SAELens `StandardTrainingSAE` on `blocks.<layer>.hook_resid_post`:

| | |
| --- | --- |
| `layer`, `features` | which site, and dictionary width (16–65,536) |
| `training_tokens`, `batch_size`, `context_size` | the token budget |
| `learning_rate`, `l1_coefficient` | step size and the sparsity penalty |
| `dataset` | `tiny-stories`, or your own pasted `training_text` |
| `seed` | the whole run is seeded |

Evaluation text must differ from training text, and the validator rejects a run
where it does not. Measuring reconstruction on the data you fit is not a
measurement.

**What a finished run reports.** The same habit as every pass above — a number
you can judge it by, not an assurance:

| | |
| --- | --- |
| `explained_variance` | how much of the residual `decode(encode(x))` recovers |
| `l0` | how many features actually fire per token |
| `dead_fraction` | features that fired in no recent step, with the window recorded |
| `loss_recovered` | substitute the reconstruction back into the model and measure next-token loss against the zero-ablation floor — the check that says whether the dictionary preserves what the model *does*, not just what its activations look like |
| `identity_substitution` | the control for that test: substituting `x` for itself through the same hook and token mask must move the loss by ≤1e-6, or the harness is measuring itself |
| `checkpoint_agreement` | save, reload, and require *exact* equality of encode and decode on held-out activations |
| `compare_huggingface` | optional: the TransformerLens model against a CPU Hugging Face reference on four fixed prompts, with the bf16 cross-device tolerance stated rather than assumed |

**Resume is exact, not approximate.** `POST /runs/{id}/resume` restores the SAE
state, the trainer state *and* the CPU/CUDA/MPS RNG state, and refuses outright
if the installed package versions have moved. A resumed run that silently
diverges from an uninterrupted one would make every number above unreadable.

**Auto-interp, because Neuronpedia has never seen your features.** A trained
dictionary has no labels anywhere, so the workspace makes its own: collect each
feature's top-activating examples (`POST /runs/{id}/examples`), then send them
to a local OpenAI-compatible server (`POST /runs/{id}/interp-jobs`).

```toml
# <training-data-dir>/config/config.toml, or set MECHLENS_CONFIG_DIR
[auto_interp.profile.local]
base_url = "http://127.0.0.1:11434/v1"   # ollama serve
model = "qwen3:14b"
response_mode = "json_schema"
```

The model answers into a fixed schema — `label`, `summary`, `evidence`,
`uncertainty` — and `uncertainty` is not decoration. A label is one LLM's guess
about another model's feature, which is the weakest claim in this repo, and it
says so. Secrets stay as environment-variable *references* in the config, never
values.

`POST /runs/{id}/trace` then traces a prompt through your SAE, with the logit
lens alongside it. That is a single layer and it stays inside the Training page:
your dictionary has no atlas position and no attribution, because the trace
viewer's `sae_cache.py` still loads Gemma Scope and nothing else. Roadmap items
5 and 6.

```bash
# a bounded run through the backend's shared compute queue, saved as JSON
python -m app.training.benchmark --tokens 32768 --layer 12
```

Runs live under `MECHLENS_TRAINING_DIR` (default `backend/data/training/`) in a
SQLite store that also tracks live metrics, batches of configs, per-run disk
usage and checkpoint pruning. See the [training plan](docs/sae-training.md) and
[training validation](docs/training-validation.md); real-model validation on the
intended hardware has not been run yet.

## Layout

| module | role |
| --- | --- |
| `app/schema.py` | the trace schema — the JSON contract everything writes into |
| `app/capture.py` | phase 1: token-by-token loop, residual stream cached at every layer |
| `app/passes/sae.py` | phase 2: Gemma Scope SAE features per (token, layer) |
| `app/passes/labels.py` | phase 3: Neuronpedia labels for those features |
| `app/passes/lens.py` | phase 4: every layer decoded through `ln_final` + `W_U` |
| `app/passes/attribution.py` | phase 5: resid/attn/mlp edges decomposing every layer's residual |
| `app/passes/layout.py` | phase 7: the atlas's position for each feature a trace reports |
| `app/atlas.py` | phase 7: the feature atlas — the shell warp, its diagnostics, the naming gate |
| `app/labels.py` | the label store — SQLite lookup, the atlas tables, a capped API fallback |
| `app/passes/__init__.py` | the `Pass` protocol — take a trace + residuals, fill fields |
| `app/store.py` | save/load: JSON document plus its `.npy` sidecar |
| `app/model_cache.py` | one model per process (loading gemma costs ~6s) |
| `app/sae_cache.py` | one SAE per layer per process (~302MB each at 16k) |
| `app/cli.py` | `trace` / `enrich` / `show` / `experiment` |
| `app/experiments.py` | the fixed-prefix intervention comparison behind `cli experiment` |
| `app/identity.py` | dictionary and artifact fingerprints — what makes "same SAE" checkable |
| `app/service/app.py` | the FastAPI service: `/trace`, `/steer`, `/atlas`, `/feature`, `/stats` |
| `app/service/jobs.py` | background jobs and the shared compute lock both halves queue on |
| `app/training/runner.py` | the SAE training loop, its checkpoints and its exact resume |
| `app/training/api.py` | the `/training/*` routes — runs, batches, storage, examples, interp jobs |
| `app/training/store.py` | SQLite run store: metrics, checkpoints, disk accounting, pruning |
| `app/training/validation.py` | correctness checks — checkpoint round-trip, HF reference agreement |
| `app/training/examples.py` | top-activating examples per trained feature |
| `app/training/auto_interp.py` | labels for those features from a local OpenAI-compatible server |
| `scripts/import_neuronpedia.py` | one-time load of the explanation export into SQLite |
| `scripts/verify_neuronpedia_mapping.py` | proves our SAE features are the ones Neuronpedia labelled |
| `scripts/build_feature_atlas.py` | one-time UMAP layout of every feature; the only module importing umap |
| `../mechlens-cloud/frontend/src/viewer/` | the single viewer source; hosted in Cloud or run in local mode (:5173) |
| `site/` | the landing page; standalone, no backend, not deployed yet (:5180) |

`LayerState.edges` holds `resid`/`attn`/`mlp` contributions once the
attribution pass has run. `kind="sae"` edges — attributing a feature's own
activation to upstream contributions, rather than the residual stream as a
whole — are deferred to a later phase: it needs the SAE encoder's own
reconstruction error in the loop, a different correctness story than the
exact decomposition here.

One caveat worth knowing before comparing labels across layers: Neuronpedia's
export does not use a single explainer. For `gemma-2-2b` at 16k, layers 16, 18,
20, 22 and 24 carry `gemini-2.5-flash-lite` explanations and the other 21 carry
`gpt-4o-mini`, and the two write in visibly different styles. Every label
records its `explainer` for that reason. See `phase3.md`.

## Measurement correctness and controlled experiments

Before interpreting the map, see [the measurement plan and DGX Spark experiment guide](docs/measurement-plan.md).
Enrichment now replaces old results atomically and checks SAE dictionary identity;
atlas builds fingerprint their actual inputs and publish complete versions. The
cross-source check compares decoder vectors with explanation embeddings independently.
Attribution currently refuses steered traces because it cannot yet replay their
intervention correctly. Its existing edges are contribution magnitudes, not causal
importance scores.

`python -m app.cli experiment --help` exposes a fixed-prefix comparison of baseline,
zero control, feature suppression and steering strengths against a chosen next-token
probability. Related prompts and unrelated controls can be saved in one JSON report.
This measures a feature intervention's effect; it does not construct a complete causal graph.

## Tests

```bash
pytest backend/tests -q            # ~79s: a 1M-param model on CPU, and stand-in SAEs
MECHLENS_SLOW=1 pytest backend/tests -q   # adds a real Gemma Scope SAE (302MB download)
```

The atlas tests deserve a note on why they are the slow ones.
`umap.UMAP(random_state=...)` is single-threaded — that is the price of an
artifact whose whole value is being the same map every time — so
`test_atlas_build.py` exercises the real pipeline end to end on 192 synthetic
features with planted structure rather than on 425,984 real ones. The geometry,
the measurements and the naming gate are tested separately in `test_atlas.py`,
on a dozen hand-written vectors.

Two of those tests are worth knowing about because they check things a comment
cannot. `test_shape_ellipsoid_matches_the_typescript_it_ports` lifts
`shapeEllipsoid` straight out of `Brain.tsx`, runs it under `node`, and compares
against the Python port to 1e-12 — the mesh shapes its surface with one and the
atlas places nodes inside it with the other, so the two agreeing is not
optional. And `test_atlas_deps.py` imports each module in a fresh interpreter to
confirm `umap` never reaches the service: the atlas is a precomputed table, so
serving it needs no projection library, and `app/schema.py` still imports
nothing but pydantic.

### Viewer contract tests

The TypeScript lit-set contract tests use the adjacent Cloud checkout. Set
`MECHLENS_VIEWER_ROOT=/path/to/mechlens-cloud/frontend/src/viewer` when it is
elsewhere. These tests require Node 22+ and skip when viewer sources are absent.
