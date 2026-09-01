"""Gemma Scope SAE loading, cached per process — the SAE twin of model_cache.

Release `gemma-scope-2b-pt-res-canonical` carries one SAE per layer of
gemma-2-2b trained on `hook_resid_post`, "canonical" meaning Google's pick of
the L0 sweep for that layer (~100 active features at 16k width).

Memory: a 16k SAE is W_enc[2304, 16384] + W_dec[16384, 2304] + biases ≈ 302MB
in fp32, so all 26 come to ~7.9GB. On a unified-memory box (GB10: one 128GB
pool) that sits next to gemma's 5GB with room to spare and there is no reason
to shuttle them to "CPU" — it is the same physical RAM. On a 16GB discrete GPU,
load with device="cpu" instead: encoding is a matmul plus a threshold, and the
residuals are tiny, so CPU encoding costs about a second per trace.

At 65k or 262k width this stops being free (262k is ~4.8GB per layer) — load
those one layer at a time.
"""

from __future__ import annotations

import time
from functools import lru_cache

import torch
from sae_lens import SAE

RELEASE = "gemma-scope-2b-pt-res-canonical"
DEFAULT_WIDTH = "16k"

# The SAEs are trained on this site; capture.RESID_HOOK must agree with it.
SAE_HOOK = "hook_resid_post"


def sae_id(layer: int, width: str = DEFAULT_WIDTH) -> str:
    return f"layer_{layer}/width_{width}/canonical"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=32)
def get_sae(layer: int, width: str = DEFAULT_WIDTH, device: str | None = None) -> SAE:
    """Load one layer's SAE; later calls with the same args reuse it.

    fp32 throughout: the residuals on disk are fp32, and at these sizes the
    precision is free. Downloads ~302MB per layer on first use, then reads
    from ~/.cache/huggingface.
    """
    sae = SAE.from_pretrained(RELEASE, sae_id(layer, width), device=device or pick_device(), dtype="float32")
    sae.eval()

    # The SAE knows which activation site it was trained on. Trusting the
    # release name here would be how you silently encode resid_pre with a
    # resid_post SAE and get plausible-looking nonsense.
    hook = sae.cfg.metadata.hook_name
    assert hook == f"blocks.{layer}.{SAE_HOOK}", f"layer {layer} SAE expects {hook}"
    return sae


def load_layers(layers: list[int], width: str = DEFAULT_WIDTH, device: str | None = None) -> dict[int, SAE]:
    """Load several SAEs, reporting progress — the first call downloads ~8GB."""
    device = device or pick_device()
    saes: dict[int, SAE] = {}
    t0 = time.time()
    for i, layer in enumerate(layers):
        saes[layer] = get_sae(layer, width, device)
        print(f"\rSAEs {i + 1}/{len(layers)} ({time.time() - t0:.0f}s)", end="", flush=True)
    d_sae = saes[layers[0]].cfg.d_sae
    print(f"\rloaded {len(layers)} {width} SAEs (d_sae={d_sae}) in {time.time() - t0:.0f}s on {device}")
    return saes
