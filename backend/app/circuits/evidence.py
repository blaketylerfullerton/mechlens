"""Activation examples for a transcoder feature.

The transcoder artifact ships its own feature evidence next to the weights:
`features/index.json.gz` gives a byte range into `features/layer_<L>.bin`, and
that range holds a 4-byte little-endian length followed by gzipped JSON.

This is proxied through the backend rather than fetched from the browser for
two reasons the spec asks for: the fetch becomes explicit in configuration and
provenance instead of an invisible side effect of opening a graph, and a cached
answer keeps a saved analysis inspectable when the network is gone.

There is no label in the artifact. These features have never been
auto-interpreted, so `label` is always None and a UI must show an explicit
unlabeled state rather than invent a name. Gemma Scope residual-SAE labels
cannot fill the gap: different dictionary, different feature space.
"""
from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from functools import lru_cache

from .adapter import TRANSCODER_SET, transcoder_revision

_TIMEOUT_S = 20


@lru_cache(maxsize=1)
def _index(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.load(gzip.open(hf_hub_download(repo, "features/index.json.gz")))


@lru_cache(maxsize=2048)
def _raw(repo: str, layer: int, feature_idx: int) -> bytes:
    from huggingface_hub import get_token, hf_hub_url

    entry = _index(repo)[str(layer)]
    offsets = entry["offsets"]
    if not 0 <= feature_idx < len(offsets) - 1:
        raise KeyError(f"feature {feature_idx} out of range for layer {layer}")

    request = urllib.request.Request(
        hf_hub_url(repo, f"features/{entry['filename']}"),
        headers={"Range": f"bytes={offsets[feature_idx]}-{offsets[feature_idx + 1] - 1}"},
    )
    token = get_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read()


def _decode(raw: bytes) -> dict:
    length = struct.unpack("<I", raw[:4])[0]
    return json.loads(gzip.decompress(raw[4 : 4 + length]))


def describe(layer: int, feature_idx: int, repo: str = TRANSCODER_SET,
             max_examples: int = 8) -> dict:
    record = _decode(_raw(repo, layer, feature_idx))

    examples = []
    for bucket in record.get("examples_quantiles", []):
        for example in bucket.get("examples", []):
            acts = example.get("tokens_acts_list") or []
            tokens = example.get("tokens") or []
            if not acts:
                continue
            peak = max(range(len(acts)), key=acts.__getitem__)
            examples.append({
                "quantile": bucket.get("quantile_name"),
                "activation": acts[peak],
                "peak_index": peak,
                "tokens": tokens,
                "activations": acts,
            })
    examples.sort(key=lambda e: e["activation"], reverse=True)

    return {
        "layer": layer,
        "feature_idx": feature_idx,
        "transcoder_set": repo,
        "transcoder_revision": transcoder_revision(repo),
        # Always None: the artifact carries no label. See the module docstring.
        "label": None,
        "label_available": False,
        "act_min": record.get("act_min"),
        "act_max": record.get("act_max"),
        # Noisy in practice -- these are the unembed directions the feature
        # writes toward, not a description. Shown, but not as an explanation.
        "top_logits": record.get("top_logits"),
        "bottom_logits": record.get("bottom_logits"),
        "examples": examples[:max_examples],
        "n_examples_available": len(examples),
    }
