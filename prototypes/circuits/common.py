"""Shared loading and provenance for the CIR-01 prototype scripts.

One place to build the replacement model and one place to record what was used,
so every artifact in out/ can be traced back to an exact set of versions. The
spec's rule is that unavailable provenance is recorded as unknown rather than
guessed, which is why several fields here are allowed to come out None.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch

OUT = Path(__file__).parent / "out"

MODEL_NAME = "google/gemma-2-2b"
# Per-layer ("PLT") GemmaScope transcoders, the set the upstream README names for
# Gemma-2-2B. 3.98GB / 29 files. The `gemma` CLI shortcut resolves to
# mwhanna/gemma-scope-transcoders (11.8GB) instead; the cross-layer sets are 33GB+.
TRANSCODER_SET = "mntss/gemma-scope-transcoders"

# mechlens runs gemma-2-2b in bfloat16 on cuda (backend/app/model_cache.py:
# pick_device). Matching it is the point -- a prototype in fp32 would measure a
# model the app never runs.
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transcoder_revision(repo: str = TRANSCODER_SET) -> str | None:
    """The artifact fingerprint. None (unknown) if the hub cannot be reached."""
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo).sha
    except Exception:
        return None


def model_revision(repo: str = MODEL_NAME) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo).sha
    except Exception:
        return None


def provenance(**extra) -> dict:
    """Everything needed to say which model, weights and code produced a number."""
    record = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "circuit_tracer": version("circuit-tracer"),
        "circuit_tracer_commit": "7f66876689f59e92fc3641650d38b8bd41749ec4",
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "transformer_lens": version("transformer-lens"),
        "nnsight": version("nnsight"),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "model": MODEL_NAME,
        "model_revision": model_revision(),
        "transcoder_set": TRANSCODER_SET,
        "transcoder_revision": transcoder_revision(),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    }
    record.update(extra)
    return record


def load_replacement_model():
    """Returns (model, load_seconds). Loading is the dominant fixed cost."""
    from circuit_tracer import ReplacementModel

    t0 = time.time()
    model = ReplacementModel.from_pretrained(
        MODEL_NAME, TRANSCODER_SET, dtype=DTYPE, device=DEVICE
    )
    return model, time.time() - t0


def save_json(path: Path, payload: dict) -> Path:
    """Atomic-ish write so a killed run cannot leave half a result behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)
    return path


def peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6
