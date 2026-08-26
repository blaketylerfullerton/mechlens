# Mechlens

DevTools for what's happening inside a transformer.

Full stack application for mechanistic interpretability. It loads a Hugging
Face causal LM with [nnsight](https://nnsight.net/) and lets you trace a
prompt through the model to inspect hidden states, attention patterns, and a
per-layer logit lens, all from a browser UI.

## Features

- **Trace** — run a forward pass over a prompt and see, layer by layer, the
  top-k tokens the model would predict if generation stopped there (logit
  lens), plus raw hidden states and attention weights.
- **Generate** — run standard autoregressive generation from a prompt.
- **Ablate** — zero out a single attention head's contribution before it's
  mixed back into the residual stream, then generate, to see what that head
  was doing.
- **Jacobian lens** — backward-pass complement to the logit lens: per layer,
  decode the gradient of a target token's logit w.r.t. that layer's hidden
  state (instead of the hidden state itself), showing which vocab directions
  are causally on the path to the prediction rather than which one the
  representation currently looks like.
- React + Vite frontend (shadcn/ui components) talking to a FastAPI backend.

## Quick start

Local, no Docker (the fast path — see `Makefile`):

```
./dev.sh
```

This starts the FastAPI backend on `:8000` and the Vite dev server on
`:5173`, and stops both on Ctrl+C. Requires a Python virtualenv at `.venv`
with `backend/requirements.txt` installed, and `npm install` run in
`frontend/`.

Docker Compose is also available (`make up`), though it's not the primary
dev path right now.

## Configuration

Set in `.env` (used by both `dev.sh` and Docker Compose):

- `MODEL_NAME` — Hugging Face model id (default: `HuggingFaceTB/SmolLM2-135M`)
- `DEVICE_MAP` — device placement for the model (default: `cpu`)

## API

- `POST /api/generate` — `{ prompt, max_new_tokens }` → generated text
- `POST /api/trace` — `{ prompt, top_k }` → tokens, hidden states, attention,
  logit lens per layer
- `POST /api/ablate` — `{ prompt, layer, head, max_new_tokens }` → generated
  text with that attention head zeroed out
- `POST /api/jacobian` — `{ prompt, target_token?, top_k }` → per-layer
  gradient norm and top vocab tokens aligned with the direction that most
  increases the target token's logit (defaults to the model's own top
  prediction)

## Roadmap

Ideas for extending the interpretability toolkit, roughly in order of how
much they build on what's already here:

- **Attention heatmaps** — render the attention weights already returned by
  `/api/trace` as a per-layer/per-head heatmap over tokens, instead of just
  raw JSON.
- **Direct logit attribution** — extend the per-layer logit lens to
  decompose the final logit into per-head contributions.
- **Activation patching / causal tracing** — run a "clean" and "corrupted"
  prompt pair, patch an activation from one into the other, and measure how
  much of the output effect is restored.

## Notes

The engine assumes a Llama/Gemma-style architecture (`model.model.layers`,
`model.lm_head`); GPT-2-style models expose these under different attribute
names and would need the paths in `backend/app/engine.py` adjusted.
