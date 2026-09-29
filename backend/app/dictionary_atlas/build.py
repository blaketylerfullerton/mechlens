"""One atlas build over local dictionaries, run as a child process.

    python -m app.dictionary_atlas.build SPEC.json OUT_DIR

Reads each checkpoint's `W_dec` straight from its safetensors file — no model
is loaded and no prompts are run — projects every row to 3D, warps the result
into the brain shell, clusters it and writes two files into OUT_DIR:

    atlas.json    the payload the viewer draws (the idle-atlas format)
    record.json   provenance and metrics: exactly what produced the map

Progress goes to stdout as one JSON object per line, `{"phase", "done",
"total"}`; a failure ends with `{"error": "..."}` and a non-zero exit. The
parent (`runner.py`) reads those lines and owns every state change.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .. import atlas
from .. import atlas_projection

# Bumped when the payload or record changes shape.
BUILD_CONTRACT = 1


class BuildError(Exception):
    """A failure the user can act on; its message is shown as-is."""


def emit(**fields) -> None:
    print(json.dumps(fields), flush=True)


def artifact_digest(path: Path) -> str:
    """The same identity `training.runner` registers: cfg.json + weights."""
    digest = hashlib.sha256()
    for name in ("cfg.json", "sae_weights.safetensors"):
        with (path / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_decoder(path: Path, features: int, d_in: int) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(path / "sae_weights.safetensors"), framework="pt") as weights:
        if "W_dec" not in weights.keys():
            raise BuildError(f"{path.name} has no W_dec tensor")
        w_dec = weights.get_tensor("W_dec").float().cpu().numpy()
    if w_dec.shape != (features, d_in):
        raise BuildError(
            f"W_dec is {tuple(w_dec.shape)}, expected ({features}, {d_in}) from the manifest"
        )
    if not np.isfinite(w_dec).all():
        raise BuildError("W_dec contains NaN or infinite values")
    return w_dec


def package_versions() -> dict:
    out = {}
    for name in ("umap-learn", "scikit-learn", "numpy", "torch"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run(spec: dict, out: Path) -> dict:
    inputs = spec["inputs"]
    settings = spec["settings"]
    identity = spec["identity"]
    seed = int(settings["seed"])
    started = time.time()

    blocks, keys = [], []
    for index, item in enumerate(inputs):
        emit(phase="loading", done=index, total=len(inputs))
        path = Path(item["checkpoint_dir"])
        if not path.is_dir():
            raise BuildError(f"Layer {item['layer']}: checkpoint folder is missing on this GPU")
        if artifact_digest(path) != item["artifact_id"]:
            raise BuildError(
                f"Layer {item['layer']}: checkpoint files changed since they were registered"
            )
        w_dec = load_decoder(path, int(item["features"]), int(identity["d_in"]))
        blocks.append(w_dec)
        keys.extend((int(item["layer"]), f) for f in range(w_dec.shape[0]))
    emit(phase="loading", done=len(inputs), total=len(inputs))
    vectors = np.vstack(blocks)
    if len(vectors) < 10:
        raise BuildError("An atlas needs at least 10 features")

    emit(phase="mapping")
    if settings["method"] == "pca":
        projection = atlas_projection.project_pca(vectors, seed=seed)
    else:
        projection = atlas_projection.project(
            vectors,
            pca_dim=int(settings["pca_dim"]),
            n_neighbors=int(settings["n_neighbors"]),
            min_dist=float(settings["min_dist"]),
            seed=seed,
            verbose=False,
        )

    emit(phase="placing")
    positions = atlas.place_in_shell(projection)
    atlas.assert_finite(positions)
    if not atlas.within_shell(positions):
        raise BuildError("The projection placed a feature outside the brain shape")

    emit(phase="measuring")
    metrics = atlas.knn_preservation(vectors, positions, seed=seed)

    emit(phase="clustering")
    labels = atlas_projection.cluster(
        positions, int(settings["min_cluster_size"]), verbose=False
    )
    metrics.update(atlas.summarise_clusters(labels))

    emit(phase="writing")
    areas = []
    for cluster_id in sorted({int(v) for v in labels.tolist()} - {-1}):
        mask = labels == cluster_id
        centroid, spread = atlas.cluster_geometry(positions[mask])
        # Local dictionaries have no label embeddings, so no area is named;
        # `coherence: None` says naming was never attempted.
        areas.append(dict(cluster=cluster_id, name=None, name_source=None,
                          n_members=int(mask.sum()), centroid=list(centroid), spread=spread,
                          coherence=None, baseline_coherence=None, explainers=""))

    input_hash = hashlib.sha256()
    input_hash.update(json.dumps(keys).encode())
    input_hash.update(np.asarray(vectors, dtype="<f4").tobytes())
    positions_sha256 = atlas.positions_hash(positions)
    params = dict(settings, source="decoder", metric="cosine" if settings["method"] == "umap" else "pca",
                  layers=",".join(str(item["layer"]) for item in inputs),
                  artifact_ids=[item["artifact_id"] for item in inputs],
                  input_sha256=input_hash.hexdigest(), build_contract=BUILD_CONTRACT)
    release = "local/" + hashlib.sha256(
        ",".join(item["artifact_id"] for item in inputs).encode()
    ).hexdigest()[:16]
    width = str(sum(int(item["features"]) for item in inputs))
    version_id = atlas.atlas_version(release, width, seed, params)

    picks = atlas.subsample_indices(len(keys), int(settings["max_nodes"]), seed=seed)
    quantised, extent = atlas.quantise_positions(positions[picks])
    payload = {
        "atlas_version": version_id,
        "model": identity["model"],
        "source": "decoder",
        "note": (
            f"Decoder directions of {len(inputs)} local dictionar"
            f"{'y' if len(inputs) == 1 else 'ies'}, mapped with {settings['method'].upper()}."
            + (" A uniform sample." if len(picks) < len(keys) else "")
        ),
        "knn_preservation": metrics.get("knn_preservation"),
        "knn_k": metrics.get("knn_k"),
        "explainer_ami": None,
        "extent": extent,
        "quantisation": "int16, position = value / 32767 * extent",
        "n_sampled": int(len(picks)),
        "n_total": len(keys),
        "max_quantisation_error": atlas.relative_error(positions[picks], quantised, extent),
        "nodes": {
            "layer": [keys[i][0] for i in picks],
            "feature": [keys[i][1] for i in picks],
            "cluster": [int(labels[i]) for i in picks],
            "xyz": quantised.reshape(-1).tolist(),
        },
        "areas": areas,
    }
    record = {
        "atlas_version": version_id,
        "release": release,
        "positions_sha256": positions_sha256,
        "n_features": len(keys),
        "n_clusters": int(metrics["n_clusters"]),
        "identity": identity,
        "inputs": [
            {k: item[k] for k in ("run_id", "layer", "artifact_id", "features", "hook", "tokens")}
            for item in inputs
        ],
        "method": settings["method"],
        "params": params,
        "metrics": metrics,
        "versions": package_versions(),
        "started_at": started,
        "finished_at": time.time(),
    }

    out.mkdir(parents=True, exist_ok=True)
    for name, body in (("atlas.json", payload), ("record.json", record)):
        temporary = out / f".{name}.partial"
        temporary.write_text(json.dumps(body, separators=(",", ":"), allow_nan=False))
        os.replace(temporary, out / name)
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m app.dictionary_atlas.build SPEC.json OUT_DIR", file=sys.stderr)
        return 64
    try:
        run(json.loads(Path(argv[1]).read_text()), Path(argv[2]))
    except BuildError as exc:
        emit(error=str(exc))
        return 2
    except MemoryError:
        emit(error="Ran out of memory. Build fewer layers at once, or lower the node limit.")
        return 1
    except Exception as exc:  # the parent shows this; the traceback goes to build.log
        import traceback
        traceback.print_exc()
        emit(error=f"Build crashed: {type(exc).__name__}: {exc}")
        return 1
    emit(phase="done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
