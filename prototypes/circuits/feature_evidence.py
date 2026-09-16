"""Fetch the activation examples and logit effects for a transcoder feature.

Feature evidence is not baked into the attribution graph. The viewer fetches it
at display time from the transcoder repo on HuggingFace: `features/index.json.gz`
gives a byte range into `features/layer_<L>.bin`, and that range holds a 4-byte
little-endian length followed by a gzipped JSON record.

Recording this here matters because the spec requires that fetching external
feature data be explicit in configuration and provenance rather than an
invisible side effect of opening a graph. It also pins the identity: evidence
must come from the same transcoder artifact the graph was built from.
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import urllib.request
from functools import lru_cache
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download, hf_hub_url

import common


@lru_cache(maxsize=1)
def _index(repo: str) -> dict:
    return json.load(gzip.open(hf_hub_download(repo, "features/index.json.gz")))


def fetch(layer: int, feature_idx: int, repo: str = common.TRANSCODER_SET) -> dict:
    entry = _index(repo)[str(layer)]
    offsets = entry["offsets"]
    if not 0 <= feature_idx < len(offsets) - 1:
        raise ValueError(f"feature {feature_idx} out of range for layer {layer}")

    url = hf_hub_url(repo, f"features/{entry['filename']}")
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={offsets[feature_idx]}-{offsets[feature_idx + 1] - 1}"}
    )
    token = get_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    raw = urllib.request.urlopen(req).read()

    length = struct.unpack("<I", raw[:4])[0]
    return json.loads(gzip.decompress(raw[4 : 4 + length]))


def top_examples(record: dict, k: int = 5) -> list[dict]:
    """The highest-activating contexts, flattened out of the quantile buckets."""
    out = []
    for bucket in record.get("examples_quantiles", []):
        for ex in bucket.get("examples", []):
            acts = ex.get("tokens_acts_list") or []
            if not acts:
                continue
            peak = max(range(len(acts)), key=lambda i: acts[i])
            out.append({
                "quantile": bucket.get("quantile_name"),
                "activation": acts[peak],
                "peak_token": ex["tokens"][peak] if peak < len(ex.get("tokens", [])) else None,
                "context": "".join(ex.get("tokens", [])),
            })
    out.sort(key=lambda e: e["activation"], reverse=True)
    return out[:k]


def describe(layer: int, feature_idx: int, repo: str = common.TRANSCODER_SET) -> dict:
    rec = fetch(layer, feature_idx, repo)
    return {
        "transcoder_set": repo,
        "transcoder_revision": common.transcoder_revision(repo),
        "layer": layer,
        "feature_idx": feature_idx,
        # There is no label field in this artifact. Features carry examples and
        # logit effects but no human-written description, so any UI must show an
        # explicit unlabeled state rather than inventing one.
        "label": None,
        "label_available": False,
        "act_min": rec.get("act_min"),
        "act_max": rec.get("act_max"),
        "top_logits": rec.get("top_logits"),
        "bottom_logits": rec.get("bottom_logits"),
        "top_examples": top_examples(rec),
        "n_quantiles": len(rec.get("examples_quantiles", [])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("features", nargs="+", metavar="LAYER:IDX")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    records = []
    for spec in args.features:
        layer, idx = (int(x) for x in spec.split(":"))
        d = describe(layer, idx)
        records.append(d)
        print(f"\n=== L{layer} feature {idx}   label: "
              f"{'(none in artifact)' if not d['label_available'] else d['label']}")
        print(f"    activation range {d['act_min']} .. {d['act_max']}")
        tl = d.get("top_logits") or []
        print(f"    top logits:    {tl[:8]}")
        bl = d.get("bottom_logits") or []
        print(f"    bottom logits: {bl[:8]}")
        for ex in d["top_examples"]:
            print(f"    act {ex['activation']:>8.2f} on {ex['peak_token']!r}"
                  f"  ...{ex['context'][-110:]!r}")

    out = args.out or common.OUT / "feature-evidence.json"
    common.save_json(out, common.provenance(features=records))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
