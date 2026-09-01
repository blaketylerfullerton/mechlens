## Phase 4 — Logit lens (the HUD)

Cheapest phase — you already have the residuals.

- For each (token, layer): apply `ln_final` then `W_U`, take top-5 tokens + probabilities. Append to each layer-state.
- Gotcha: Gemma uses RMSNorm and has a logit softcap — TransformerLens handles this, but verify your layer-25 lens output matches the actual model output distribution. That's your correctness test.

**Done when:** you can watch the "573 → 583" style crystallization in raw JSON — the top guess converging over depth.

---

## Findings before writing it

Verified against the installed stack (`transformer_lens` 3.8.0) and the traces
already in `backend/traces/`.

### 1. The softcap is *not* inside `model.unembed`

This is the one that silently produces plausible-but-wrong numbers. Gemma 2
applies `cap * tanh(x / cap)` with `cap = 30.0`, and TransformerLens applies it
in `HookedTransformer.forward` — *after* `unembed`, not inside it:

```python
residual = self.ln_final(residual)
logits   = self.unembed(residual)
logits   = apply_softcap(logits, self.cfg.output_logits_soft_cap)   # cfg = 30.0
```

So `ln_final → W_U` alone will not reproduce the captured `logits.top_k`. The
lens has to apply the cap itself. Import the model's own function rather than
re-typing the formula, so the two can never drift:

```python
from transformer_lens.utilities.activation_functions import apply_softcap
```

It is a no-op when the cap is unset, so the same code path works on the tiny
test model (`output_logits_soft_cap = 0.0`) as on gemma.

Softcap compresses the *tail* far more than the head — top-1 comes out roughly
right without it, which is exactly why this is worth an assertion instead of an
eyeball. Entropy is where the error shows up first.

### 2. `ln_final` needs no special handling — but the model does

Gemma's `(1 + w)` RMSNorm scaling is already baked into `model.ln_final.w` by
TransformerLens's weight conversion, and `RMSNorm.forward` upcasts to float32
internally before computing the scale. So:

- Call `model.ln_final(x)` directly. Do not reimplement RMSNorm.
- Feed it the stored **float32** residuals, not a bf16 downcast. The sidecar
  holds an exact widening of the bf16 values that capture saw, and RMSNorm
  upcasts to float32 anyway — so this path is bit-identical to capture. Casting
  down to bf16 first would throw away precision the norm is about to want.

Both `Unembed.forward` (an `F.linear`) and `RMSNorm.forward` broadcast over
leading dims, so a `[n_positions, d_model]` slice works without reshaping to
`[batch, pos, d_model]`.

### 3. Folded vs unfolded weights do not matter here

Our traces record `normalization: "RMS"` — unfolded, since `model_cache.py`
uses `from_pretrained_no_processing`. The SAE pass *refuses* folded traces
(`RMSPre`) because folding moves `resid_post` off the distribution Gemma Scope
was fitted on.

The lens has no such problem. Folding is `RMSNormPre(x) @ W_U_folded ==
RMSNorm(x) @ W_U` — mathematically the same answer. The only real requirement
is that `ln_final` and `W_U` come from the *same* model instance. So no
normalization guard, unlike `passes/sae.py`.

### 4. This pass breaks the "no model load" property

`passes/__init__.py` currently claims "a pass never needs the model", and both
`--sae` and `--labels` deliver on it. This one cannot: `W_U` is
`2304 × 256_000`. The options were a `W_U`/`ln_final` sidecar or just loading
gemma; loading gemma (~10s, weights already in the HF cache) is the honest one.
Cache the model handle via `model_cache.get_model()` and make it injectable the
way `SAEPass.saes` and `LabelsPass.store` are, so tests can stand in
`tiny-stories-1M` on CPU.

The docstrings in `passes/__init__.py` and the README both need the caveat.

### 5. Reuse the capture's summary code, don't re-derive it

`capture._logit_summary` already does topk + full-vocab entropy in float32 with
the exact conventions the trace records. Promote it to public
`capture.logit_summary` and call it from the lens. 806 calls for a 31-token
trace (31 × 26) is milliseconds, and sharing the code means the correctness
test below is testing the *lens*, not two parallel implementations of softmax.

### 6. Memory: go layer by layer

`[31 × 26, 2304] @ [2304, 256_000]` in one shot is 825MB of float32 logits.
Per layer it is 32MB. Loop over layers.

### 7. BOS is fine here, unlike the SAE pass

Position 0 is excluded from every SAE statistic because Gemma Scope SAEs see an
off-manifold activation there and light up meaninglessly. The lens is the
model's *own* unembed — position 0's prediction is a real prediction. Include
it, and say so, because the asymmetry with `passes/sae.py` looks like a bug
otherwise.

## The correctness test

Layer 25 is `resid_post` of the last block, which is precisely what
`ln_final` consumes in `forward`. So the final layer's lens must reproduce the
captured `logits` for every position:

```
lens(residuals[pos, n_layers - 1]).top_k[0].token_id == steps[pos].logits.top_k[0].token_id
```

Assert it in `tests/`, and record it on the `PassRecord` as
`final_layer_agreement` (must be 1.0) so a saved trace can be judged without
re-running anything — the same way `l0_mean` and `explained_variance_mean`
judge the SAE pass.

Probabilities are compared with a tolerance, not exactly: capture ran the
matmul on CUDA in bf16 and the lens may re-run it on a different device.

### The one thing this check gets wrong on the first try

Comparing top-1 *token ids* fails on ~1 position in 16, and not because of a
bug. Found on the first gemma run:

```
position 5, token ' is'
  model: (' one', 0.243803, logit 28.625), (' a', 0.243803, logit 28.625), ...
  lens : (' a',   0.243803, logit 28.625), (' one', 0.243803, logit 28.625), ...
  entropy: model 3.006342, lens 3.006342
```

Identical distribution, identical entropy — only the order of the tied pair
differs. Gemma's late-layer logits sit around 28, where bf16 steps by 0.125, so
two candidates land on the *same value* often enough to see on a 16-token
trace. `torch.topk` then breaks the tie however it likes, and the two call
sites do not index the same shape (capture reads `logits[0, pos]` out of
`[1, seq, vocab]`, the lens reads `logits[pos]` out of `[seq, vocab]`), so they
need not agree.

A strict id comparison reports that as a 93.8% failure and sends you hunting
for a softcap bug that is not there. So:

- A position agrees when the ids match **or** the two top-1 probabilities are
  equal (`TIE_ATOL = 1e-5`).
- Count the ties as `final_layer_argmax_ties` and print them. Absorbing them
  silently would hide a real regression later.
- Keep `final_layer_exact_top1` alongside, so the strict number is still on the
  record.
- Add `final_layer_max_entropy_delta` as the primary numeric check: entropy is
  taken over the full vocabulary, so it is blind to tie-breaking and still
  moves sharply if the softcap is skipped.

Measured across all five saved traces: max prob delta and max entropy delta are
both exactly **0.0**, with 0–2 argmax ties per trace.

The same effect makes `top1_agreement_by_layer[-1]` read 0.94 where
`final_layer_agreement` reads 1.00 — the curve is a plain id comparison and the
check is tie-aware. Worth a comment where both are printed, or it looks like a
contradiction.

## The "done when", as a number

`top1_agreement_by_layer` — the fraction of positions where layer L's lens top-1
already equals the model's final answer. That curve rising toward 1.0 *is* the
"573 → 583" crystallization, and the first layer where it crosses 0.5
(`crossover_layer`) is the single number that says where the model made up its
mind.

**Entropy is not that story told the other way, and assuming it is will mislead
you.** Measured on gemma, `entropy_by_layer` runs
`4.1 → 1.1 (L7) → 2.4 (L15) → 3.2 (L25)`: it *starts* low and *ends* high. The
reason is that early residuals sit near the token embedding and decode back to
the token already at that position — layers 0–9 read the current token back at
60–85% confidence, which is confident but not a prediction:

```
   L3   ' of'      71.81%   ·      <- echoing the token at this position
   L17  ' city'    40.87%
   L19  ' Paris'   22.47%   ◀      <- the answer arrives
   L20  ' Paris'   92.66%   ◀
```

So record `echo_by_layer` too — the fraction of positions where layer L's top-1
is the *current* token — and mark those rows in any display. Without it, a
reader takes L4's 85%-confident reading of the word in front of it for the
model making up its mind, and concludes from the entropy curve that the model
gets *less* sure with depth. Both backwards.

## Scope

- `app/passes/lens.py` — `LogitLensPass`, filling `LayerState.logit_lens`
  (already defined in `schema.py`; no schema bump).
- `capture._logit_summary` → `capture.logit_summary`.
- `--lens` on `cli trace` and `cli enrich`; a per-layer crystallization table in
  `cli show --lens`, with the echo rows marked.
- `tests/test_lens_pass.py`, including the layer-25 identity against
  `tiny-stories-1M` on CPU.

**Done.** 71 passed, 1 skipped. On all five saved traces layer 25 reproduces the
model's own output at every position with prob and entropy deltas of exactly
0.0, and `crossover_layer` lands at 18–21.
