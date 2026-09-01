"""Bulk-load Neuronpedia's explanation export into the local label DB.

    python scripts/import_neuronpedia.py                      # all 26 layers
    python scripts/import_neuronpedia.py --layers 0-5,20      # a subset
    python scripts/import_neuronpedia.py --embeddings         # +vectors (~2GB)

The API is for one-offs; this is the lookup table. A trace wants thousands of
labels at once, and asking neuronpedia.org for them one at a time would be
~1.2MB per feature and rude besides.

    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model}/{source_set}/explanations/

Public bucket, no credentials, 64 gzipped JSONL batches per source set (~22MB).
The old /api/explanation/export endpoint is gone and 400s with a pointer here.

Two things about the data that will bite otherwise
--------------------------------------------------
The `embedding` field is a JSON **string** on the older gpt-4o-mini rows and a
real list on the newer gemini ones. Same 256-dim L2-normalised vector either
way, but len() on the unparsed string gives you 3,109 and no error.

A few thousand features carry more than one explanation, so records are
collapsed with the same `pick_explanation` ladder the API fallback uses —
otherwise the winner is whichever batch S3 listed last.

Features absent from the export are left absent from the DB rather than being
written as "looked up, no explanation". The export is a snapshot: missing here
does not prove unexplained upstream, and a NULL row would suppress a legitimate
API lookup forever. The ~0.3% gap resolves itself the first time the pass runs
with --fetch-missing, which caches the negative properly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.labels import DEFAULT_DB_PATH, LabelRow, LabelStore, pick_explanation  # noqa: E402
from app.sae_cache import DEFAULT_WIDTH, neuronpedia_id  # noqa: E402

BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# 16 parallel fetches got throttled into XML error bodies that gunzip refuses;
# 6 is comfortably under whatever the limit is.
WORKERS = 6
TIMEOUT_S = 180

N_LAYERS = 26  # gemma-2-2b
WIDTHS = {"16k": 16384, "65k": 65536, "262k": 262144}


def parse_layers(spec: str | None) -> list[int]:
    """"0-5,20" -> [0,1,2,3,4,5,20]; None means every layer."""
    if not spec:
        return list(range(N_LAYERS))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < N_LAYERS]
    if bad:
        raise SystemExit(f"layers out of range: {bad}")
    return sorted(set(out))


# --------------------------------------------------------------------------
# the bucket
# --------------------------------------------------------------------------


def list_batches(session: requests.Session, model_id: str, source_set: str) -> list[str]:
    """Every explanations/*.jsonl.gz key for one source set.

    Listed rather than probed batch-0, batch-1, ... until a 404: the batch
    count differs between source sets (the re-explained layers ship 256 rows
    per file against 1024), and a probe loop would quietly stop early on a
    transient error and import a partial dictionary.
    """
    prefix = f"v1/{model_id}/{source_set}/explanations/"
    keys: list[str] = []
    token: str | None = None

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = session.get(BUCKET, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for contents in root.findall("s3:Contents", S3_NS):
            key = contents.findtext("s3:Key", namespaces=S3_NS) or ""
            if key.endswith(".jsonl.gz"):
                keys.append(key)

        if root.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            return sorted(keys)
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)


def fetch_batch(session: requests.Session, key: str) -> bytes:
    response = session.get(f"{BUCKET}/{key}", timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.content


def parse_batch(blob: bytes) -> list[dict]:
    """One batch file -> its raw export records, unmerged."""
    return [
        json.loads(line)
        for line in gzip.decompress(blob).decode().splitlines()
        if line.strip()
    ]


def to_rows(records: list[dict], source_set: str, keep_embeddings: bool) -> list[LabelRow]:
    """Collapse a layer's export records to one row per feature.

    Most features have exactly one explanation, but not all: a few thousand
    across the release carry two or more, usually a stray claude-3-5-sonnet or
    o3-mini alongside the bulk explainer. Left alone they arrive as duplicate
    primary keys and the last writer wins, which means the label for those
    features depends on the order S3 happened to list the batches in.

    So they go through the same `pick_explanation` ladder the API fallback
    uses. Bulk and fallback then agree, and re-running the import is stable.
    """
    by_feature: dict[int, list[dict]] = {}
    for record in records:
        by_feature.setdefault(int(record["index"]), []).append(record)

    rows: list[LabelRow] = []
    for feature, candidates in by_feature.items():
        chosen = pick_explanation(candidates)
        if chosen is None:
            continue

        embedding = None
        if keep_embeddings:
            raw = chosen.get("embedding")
            if isinstance(raw, str):  # older rows serialise the vector as JSON text
                raw = json.loads(raw)
            if raw:
                embedding = np.asarray(raw, dtype=np.float32)

        rows.append(
            LabelRow(
                source_set=source_set,
                feature=feature,
                text=chosen.get("description"),
                explainer=chosen.get("explanationModelName"),
                explanation_type=chosen.get("typeName"),
                embedding=embedding,
            )
        )
    return rows


# --------------------------------------------------------------------------
# one layer
# --------------------------------------------------------------------------


def import_layer(
    store: LabelStore,
    session: requests.Session,
    layer: int,
    width: str,
    keep_embeddings: bool,
) -> dict:
    model_id, source_set = neuronpedia_id(layer, width)
    t0 = time.time()

    keys = list_batches(session, model_id, source_set)
    if not keys:
        raise SystemExit(f"no explanation export found for {model_id}/{source_set}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        blobs = list(pool.map(lambda k: fetch_batch(session, k), keys))

    records = [r for blob in blobs for r in parse_batch(blob)]
    rows = to_rows(records, source_set, keep_embeddings)
    store.upsert(rows)

    explainers: dict[str, int] = {}
    for row in rows:
        explainers[row.explainer or "?"] = explainers.get(row.explainer or "?", 0) + 1

    d_sae = WIDTHS[width]
    return {
        "layer": layer,
        "source_set": source_set,
        "rows": len(rows),
        "duplicates": len(records) - len(rows),
        "coverage": len(rows) / d_sae,
        "explainers": explainers,
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", help="subset, e.g. '0-5,20' (default: all 26)")
    p.add_argument("--width", default=DEFAULT_WIDTH, choices=sorted(WIDTHS))
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="also store the 256-dim explanation vectors (~76MB per layer)",
    )
    args = p.parse_args()

    layers = parse_layers(args.layers)
    print(f"importing {len(layers)} layers at {args.width} into {args.db}")
    if args.embeddings:
        print(f"  with embeddings — expect ~{76 * len(layers) / 1000:.1f}GB")

    session = requests.Session()
    totals: dict[str, int] = {}
    t0 = time.time()

    with LabelStore(args.db, width=args.width) as store:
        for layer in layers:
            result = import_layer(store, session, layer, args.width, args.embeddings)
            mix = ", ".join(f"{k} {v}" for k, v in sorted(result["explainers"].items()))
            print(
                f"  layer {result['layer']:>2}  {result['rows']:>6} features  "
                f"{result['coverage']:6.1%}  +{result['duplicates']:<3} dup  "
                f"{result['elapsed_s']:5.1f}s  {mix}"
            )
            for name, count in result["explainers"].items():
                totals[name] = totals.get(name, 0) + count

        stats = store.stats()

    print(f"\n{sum(totals.values())} explanations in {time.time() - t0:.0f}s")
    print(f"db now holds {stats['labelled']} labels ({stats['with_embedding']} with embeddings)")

    # The split is the thing to notice: the export does not use one explainer
    # throughout, and anything comparing label text across layers has to know.
    if len(totals) > 1:
        print("\nexplainers are mixed across layers — see phase3.md:")
        for name, count in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>7}  {name}")


if __name__ == "__main__":
    main()
