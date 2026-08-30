"""Process-wide model cache.

Loading gemma-2-2b takes ~10s (disk deserialize + TransformerLens weight
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

MODEL_NAME = "gemma-2-2b"  # BASE model. NOT -it: Gemma Scope SAEs are trained on base ("pt") activations.


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
    get_model() and get_model("gemma-2-2b") would otherwise be two distinct
    keys and load the model twice.
    """
    return _load(model_name or MODEL_NAME)


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
    model = loader(model_name, device=device, dtype=dtype)
    model.eval()

    print(
        f"loaded {model_name} in {time.time() - t0:.1f}s | device={device} "
        f"dtype={dtype} layers={model.cfg.n_layers} d_model={model.cfg.d_model}"
    )
    return model
