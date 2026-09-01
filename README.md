# mechlens

Mechanistic-interpretability tracing for `gemma-2-2b` under TransformerLens.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # gemma is gated: accept the license at hf.co/google/gemma-2-2b
```

## Phase 1 — capture a trace

```bash
cd backend/app
python run_trace.py -p "The Golden Gate Bridge is located in the city of" -n 20
```

Writes a pair of files to `backend/traces/`:

| file | what |
| --- | --- |
| `<id>.json` | the `Trace` document — tokens, per-position next-token distributions, per-layer state |
| `<id>.residuals.npy` | `[n_tokens, n_layers, d_model]` float32 residual stream (`hook_resid_post`) |

Read them back:

```python
from store import load
trace, residuals = load("../traces/golden-gate.json")   # mmap=True to page it in lazily
residuals[7, 11]                                        # token 7, layer 11 -> [d_model]
```

### Layout

| module | role |
| --- | --- |
| `backend/app/schema.py` | the trace schema — the JSON contract everything downstream writes into |
| `backend/app/capture.py` | token-by-token generation loop with the residual stream cached at every layer |
| `backend/app/store.py` | save/load: JSON document plus its `.npy` sidecar |
| `backend/app/run_trace.py` | CLI entrypoint |
| `backend/app/model_cache.py` | one model per process (loading gemma costs ~6s) |

`LayerState.logit_lens`, `.features` and `.edges` are defined and empty — the
logit-lens, SAE and attribution passes fill them in on a trace that already exists.

## Tests

```bash
pytest backend/tests -q     # ~5s, runs against a 1M-param model on CPU
```
