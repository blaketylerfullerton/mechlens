"""Process-wide model cache.

Loading a model (gemma-2-2b: ~10s) takes (disk deserialize + TransformerLens weight
processing) — the HF weights themselves are already cached under
~/.cache/huggingface, so nothing is re-downloaded. The only way to avoid
paying that cost repeatedly is to keep one process alive and reuse the
handle, which is what get_model() does:

  - scripts / REPL: run under `ipython` and the model survives edits
    (`%load_ext autoreload; %autoreload 2; %run trace.py`)
  - FastAPI: call get_model() once on startup; every request reuses it

Set HF_HUB_OFFLINE=1 to skip the hub revision check on each load.
"""

import os
import time
from functools import lru_cache

import torch
from transformer_lens import HookedTransformer

from .profiles import PROFILES, default_model, get_profile

# Hugging Face repo -> the name TransformerLens loads it by.
SUPPORTED_TRAINING_MODELS = {p.repository: p.load_name for p in PROFILES.values()}


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32  # bf16 on CPU is slow; fp32 needs ~10GB RAM


def get_model(model_name: str | None = None) -> HookedTransformer:
    """Load `model_name` once per process; later calls return the same object.

    The default is resolved here rather than in the cached function's
    signature: lru_cache keys on the arguments actually passed, so
    get_model() and get_model("gpt2") would otherwise be two distinct keys and
    load the model twice. Any name a profile knows (alias, repository) resolves
    to the one TransformerLens loads.
    """
    name = model_name or default_model()
    try:
        name = get_profile(name).load_name
    except KeyError:
        pass  # not in the registry: TransformerLens may still know it
    return _load(name)


@lru_cache(maxsize=1)
def _load(model_name: str) -> HookedTransformer:
    """maxsize=1 on purpose — a second model would not fit alongside the first
    on a 16GB Mac, so asking for one evicts the old handle. Note that eviction
    happens only after the new model is built, so the switch briefly holds
    both; on a memory-tight box, restart the process instead.
    """
    device, dtype = pick_device()
    t0 = time.time()

    # from_pretrained_no_processing skips LayerNorm folding / weight centering,
    # which TransformerLens itself warns against doing in reduced precision.
    # It is also the faster path. Set MECHLENS_PROCESS_WEIGHTS=1 to opt back in
    # (needed if a lens depends on folded LN or centered writes).
    loader = (
        HookedTransformer.from_pretrained
        if os.getenv("MECHLENS_PROCESS_WEIGHTS")
        else HookedTransformer.from_pretrained_no_processing
    )
    from transformers import AutoConfig
    from transformer_lens.loading_from_pretrained import get_official_model_name
    official_name = get_official_model_name(model_name)
    hf_config = AutoConfig.from_pretrained(official_name, trust_remote_code=False)
    revision = getattr(hf_config, "_commit_hash", None)
    model = loader(model_name, device=device, dtype=dtype, **({"revision": revision} if revision else {}))
    model.cfg.model_revision = revision
    model.eval()

    print(
        f"loaded {model_name} in {time.time() - t0:.1f}s | device={device} "
        f"dtype={dtype} layers={model.cfg.n_layers} d_model={model.cfg.d_model}"
    )
    return model
