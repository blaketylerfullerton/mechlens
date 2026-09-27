"""SAE loading, cached per process — the SAE twin of model_cache.

Which release, which SAE per layer and which widths exist come from
`profiles`; a release defaults to the default model's. The numbers below are
Gemma Scope's (`gemma-scope-2b-pt-res-canonical`, one SAE per layer of
gemma-2-2b, "canonical" meaning Google's pick of the L0 sweep for that layer).

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

from .profiles import RESID_HOOK, SAERelease, get_release, reads_layer

# The trace-side hook site every SAE is matched against; see profiles.reads_layer.
SAE_HOOK = RESID_HOOK


def sae_id(layer: int, width: str | None = None, release: str | None = None) -> str:
    return get_release(release).sae_id(layer, width)


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def neuronpedia_id(layer: int, width: str | None = None, release: str | None = None) -> tuple[str, str]:
    """Our SAE -> Neuronpedia's (model_id, source_set), e.g. Gemma Scope layer 20 at 16k:

        ("gemma-2-2b", "20-gemmascope-res-16k")

    Read from SAELens' registry rather than built by hand. The canonical L0
    pick and the Neuronpedia source set come from the same yaml entry, so the
    two cannot drift apart, and this stays correct at 65k or on a
    non-canonical variant. Building the string ourselves is how you end up
    labelling features from a dictionary we did not load.

    Verified empirically — see scripts/verify_neuronpedia_mapping.py.
    """
    return get_release(release).neuronpedia_id(layer, width)


@lru_cache(maxsize=32)
def get_sae(layer: int, width: str | None = None, device: str | None = None, release: str | None = None) -> SAE:
    """Load one layer's SAE; later calls with the same args reuse it.

    fp32 throughout: the residuals on disk are fp32, and at these sizes the
    precision is free. Downloads on first use, then reads from
    ~/.cache/huggingface.
    """
    spec: SAERelease = get_release(release)
    sae = SAE.from_pretrained(spec.release, spec.sae_id(layer, width), device=device or pick_device(), dtype="float32")
    sae.eval()

    # The SAE knows which activation site it was trained on. Trusting the
    # release name here would be how you silently encode resid_pre with a
    # resid_post SAE and get plausible-looking nonsense.
    hook = sae.cfg.metadata.hook_name
    assert reads_layer(hook, layer), f"layer {layer} SAE expects {hook}"
    return sae


def expects_centered(sae) -> bool:
    """Whether the SAE was trained on a model loaded with center_writing_weights.

    gpt2-small-res-jb was. Mechlens loads models unprocessed, so its residuals
    carry a per-vector mean those SAEs never saw — explained variance goes
    from ~0.95 to negative on later layers. The fix is exact, not a
    correction: the residual stream is a sum of writes, so centering every
    write is the same as subtracting the finished vector's mean
    (`center_input`). LayerNorm removes that mean anyway, so the model's
    behaviour is untouched.
    """
    metadata = getattr(getattr(sae, "cfg", None), "metadata", None)
    kwargs = getattr(metadata, "model_from_pretrained_kwargs", None) or {}
    return bool(kwargs.get("center_writing_weights"))


def center_input(sae, x: torch.Tensor) -> torch.Tensor:
    """`x` as the SAE's training model would have produced it; see expects_centered."""
    return x - x.mean(dim=-1, keepdim=True) if expects_centered(sae) else x


def load_layers(layers: list[int], width: str | None = None, device: str | None = None,
                release: str | None = None) -> dict[int, SAE]:
    """Load several SAEs, reporting progress — the first call downloads them."""
    device = device or pick_device()
    width = get_release(release).check_width(width)
    saes: dict[int, SAE] = {}
    t0 = time.time()
    for i, layer in enumerate(layers):
        saes[layer] = get_sae(layer, width, device, release)
        print(f"\rSAEs {i + 1}/{len(layers)} ({time.time() - t0:.0f}s)", end="", flush=True)
    d_sae = saes[layers[0]].cfg.d_sae
    print(f"\rloaded {len(layers)} {width} SAEs (d_sae={d_sae}) in {time.time() - t0:.0f}s on {device}")
    return saes
