![phase 0](/docs/Screenshot%202026-08-31%20at%208.39.14 PM.png)

## Phase 0 — Model loading & environment

Get Gemma 2 2B running under TransformerLens before anything else.

- Accept the Gemma license on HuggingFace, set up `HF_TOKEN`.
- `HookedTransformer.from_pretrained("gemma-2-2b")` in bf16. It's ~5GB — fits a consumer GPU or M-series Mac (MPS), but decide your target device _now_ because it shapes everything (SAE loading, latency).
- Write a script that generates 20 tokens greedily and prints them. That's your smoke test.

**Done when:** you can generate text and access `blocks.{i}.hook_resid_post` via `run_with_cache`.
Done 
**Commit**: 6d7c6a882367d6a6c87140cd81715f5cfcdb5f3d