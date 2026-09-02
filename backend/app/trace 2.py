"""
Phase 0 smoke test: load gemma-2-2b under TransformerLens, greedy-generate
20 tokens with a manual loop, and verify residual-stream hooks are reachable.

Run:  python scripts/00_smoke_generate.py
Requires: HF_TOKEN in env, license accepted at hf.co/google/gemma-2-2b
"""

import time
import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gemma-2-2b"  # BASE model. NOT -it: Gemma Scope SAEs are trained on base ("pt") activations.
N_NEW_TOKENS = 20
PROMPT = "The Golden Gate Bridge is located in the city of"


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32  # bf16 on CPU is slow; fp32 needs ~10GB RAM


def main() -> None:
    device, dtype = pick_device()
    print(f"device={device} dtype={dtype}")

    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=dtype)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s | layers={model.cfg.n_layers} d_model={model.cfg.d_model}")
    # gemma-2-2b should report: layers=26 d_model=2304

    # --- 1. Greedy generation, manual loop (this becomes the Phase 1 skeleton) ---
    tokens = model.to_tokens(PROMPT)  # shape [1, seq]; BOS prepended automatically
    generated = tokens

    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_NEW_TOKENS):
            logits = model(generated)               # [1, seq, vocab]
            next_tok = logits[0, -1].argmax()       # greedy
            generated = torch.cat(
                [generated, next_tok.view(1, 1)], dim=1
            )
    dt = time.time() - t0
    print(f"\ngenerated {N_NEW_TOKENS} tokens in {dt:.1f}s ({dt / N_NEW_TOKENS:.2f}s/token)")
    print("output:", model.to_string(generated[0]))

    # --- 2. Hook check: can we cache the residual stream? (Phase 1 depends on this) ---
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    resid = cache["blocks.0.hook_resid_post"]
    print(f"\nblocks.0.hook_resid_post shape: {tuple(resid.shape)}")  # (1, seq, 2304)
    assert resid.shape[-1] == model.cfg.d_model

    # confirm every layer's resid_post is present
    missing = [
        i for i in range(model.cfg.n_layers)
        if f"blocks.{i}.hook_resid_post" not in cache
    ]
    assert not missing, f"missing hooks at layers {missing}"
    print(f"all {model.cfg.n_layers} resid_post hooks present. Phase 0 complete.")


if __name__ == "__main__":
    main()