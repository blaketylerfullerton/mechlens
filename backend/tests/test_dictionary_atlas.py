"""Atlas builds over local dictionaries: validation, the child process, cancel.

The dictionaries are tiny synthetic checkpoints written to a temporary
training store, registered the way `training.runner` registers real ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from app.dictionary_atlas.api import router
from app.dictionary_atlas.runner import AtlasBuildRunner
from app.dictionary_atlas.store import AtlasBuildStore
from app.training.store import RunStore

D_IN = 16
FEATURES = 48


def add_dictionary(store: RunStore, layer: int, model: str = "gpt2", seed: int = 0) -> dict:
    run = store.create({"layer": layer, "features": FEATURES})
    path = store.root / run["id"] / "checkpoint-1"
    path.mkdir(parents=True)
    rng = np.random.default_rng(seed + layer)
    # Three groups of features pointing in different directions, so there is
    # structure to map and cluster.
    centres = rng.normal(size=(3, D_IN))
    rows = np.vstack([centres[i % 3] + 0.1 * rng.normal(size=D_IN) for i in range(FEATURES)])
    save_file({"W_dec": torch.tensor(rows, dtype=torch.float32),
               "W_enc": torch.zeros(D_IN, FEATURES)}, str(path / "sae_weights.safetensors"))
    (path / "cfg.json").write_text(json.dumps({"d_in": D_IN, "d_sae": FEATURES}))
    digest = hashlib.sha256()
    for name in ("cfg.json", "sae_weights.safetensors"):
        digest.update((path / name).read_bytes())
    checkpoint = dict(path=str(path.relative_to(store.root)), model=model, model_revision=None,
                      tokenizer=model, tokenizer_revision=None, normalization="LN", d_in=D_IN,
                      n_layers=12, layer=layer, hook=f"blocks.{layer}.hook_resid_post",
                      features=FEATURES, normalize_activations="none", artifact_id=digest.hexdigest(),
                      tokens=1024)
    return store.update(run["id"], status="completed", checkpoint=checkpoint)


@pytest.fixture
def setup(tmp_path):
    training = RunStore(tmp_path)
    builds = AtlasBuildStore(tmp_path)
    runner = AtlasBuildRunner(builds, poll_seconds=0.05)
    app = FastAPI()
    app.include_router(router(training, builds, runner))
    return TestClient(app), training, builds, runner


def wait_for(client, build_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        build = client.get(f"/atlas/builds/{build_id}").json()
        if build["status"] in {"completed", "failed", "cancelled"}:
            return build
        time.sleep(0.1)
    raise AssertionError(f"build did not finish: {build}")


def test_check_accepts_compatible_layers(setup):
    client, training, *_ = setup
    runs = [add_dictionary(training, layer) for layer in (4, 2)]
    body = client.post("/atlas/builds/check", json={"run_ids": [r["id"] for r in runs]}).json()
    assert body["layers"] == [2, 4]
    assert body["features"] == 2 * FEATURES
    assert body["identity"]["model"] == "gpt2"


@pytest.mark.parametrize("problem, message", [
    ("model", "has model 'pythia' but layer"),
    ("unknown", "is not on this GPU"),
    ("training", "has not finished training"),
    ("artifact", "differs from the saved checkpoint"),
    ("same_layer", "one dictionary per layer"),
    ("duplicate", "selected twice"),
])
def test_check_explains_what_is_incompatible(setup, problem, message):
    client, training, *_ = setup
    first = add_dictionary(training, 0)
    second = add_dictionary(training, 1, model="pythia" if problem == "model" else "gpt2")
    ids = [first["id"], second["id"]]
    body = {"run_ids": ids}
    if problem == "unknown":
        body["run_ids"] = [first["id"], "f" * 32]
    elif problem == "training":
        training.update(second["id"], status="running")
    elif problem == "artifact":
        body["expected_artifact_ids"] = {second["id"]: "0" * 64}
    elif problem == "same_layer":
        body["run_ids"] = [first["id"], add_dictionary(training, 0)["id"]]
    elif problem == "duplicate":
        body["run_ids"] = [first["id"], first["id"]]
    response = client.post("/atlas/builds/check", json=body)
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_pca_build_records_what_produced_it(setup):
    client, training, *_ = setup
    runs = [add_dictionary(training, layer) for layer in (0, 3)]
    body = {"build_id": "build-pca-1", "run_ids": [r["id"] for r in runs],
            "settings": {"method": "pca", "min_cluster_size": 5}}
    started = client.post("/atlas/builds", json=body)
    assert started.status_code == 202
    assert "checkpoint_dir" not in started.json()["spec"]["inputs"][0]
    # Re-posting the same id is a no-op, not a second build.
    assert client.post("/atlas/builds", json=body).json()["id"] == "build-pca-1"

    build = wait_for(client, "build-pca-1")
    assert build["status"] == "completed", build["error"]
    record = build["result"]["record"]
    assert record["method"] == "pca"
    assert record["identity"]["model"] == "gpt2"
    assert [i["layer"] for i in record["inputs"]] == [0, 3]
    assert [i["artifact_id"] for i in record["inputs"]] == [r["checkpoint"]["artifact_id"] for r in runs]
    assert record["params"]["min_cluster_size"] == 5

    atlas = client.get("/atlas/builds/build-pca-1/atlas").json()
    assert atlas["atlas_version"] == record["atlas_version"]
    assert atlas["source"] == "decoder"
    assert atlas["n_total"] == 2 * FEATURES
    assert len(atlas["nodes"]["xyz"]) == 3 * atlas["n_sampled"]
    assert set(atlas["nodes"]["layer"]) == {0, 3}


def test_umap_build_is_deterministic(setup):
    client, training, *_ = setup
    runs = [add_dictionary(training, layer) for layer in (1, 2)]
    ids = [r["id"] for r in runs]
    versions = []
    for build_id in ("build-umap-1", "build-umap-2"):
        client.post("/atlas/builds", json={"build_id": build_id, "run_ids": ids,
                                           "settings": {"min_cluster_size": 5}})
        build = wait_for(client, build_id, timeout=300)
        assert build["status"] == "completed", build["error"]
        versions.append(build["result"]["record"]["positions_sha256"])
    assert versions[0] == versions[1]


def test_changed_weights_fail_with_a_clear_message(setup):
    client, training, *_ = setup
    run = add_dictionary(training, 0)
    weights = training.root / run["checkpoint"]["path"] / "cfg.json"
    weights.write_text("{}")
    client.post("/atlas/builds", json={"build_id": "build-bad-1", "run_ids": [run["id"]]})
    build = wait_for(client, "build-bad-1")
    assert build["status"] == "failed"
    assert build["error"] == "Layer 0: checkpoint files changed since they were registered"


def test_cancel_stops_the_child_process(setup, tmp_path):
    client, training, builds, runner = setup
    slow = tmp_path / "slow-python"
    slow.write_text("#!/bin/sh\necho '{\"phase\": \"mapping\"}'\nexec sleep 60\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC)
    runner.python = str(slow)
    run = add_dictionary(training, 0)
    client.post("/atlas/builds", json={"build_id": "build-cancel", "run_ids": [run["id"]]})
    deadline = time.time() + 10
    while client.get("/atlas/builds/build-cancel").json()["phase"] != "mapping":
        assert time.time() < deadline
        time.sleep(0.05)
    assert client.post("/atlas/builds/build-cancel/cancel").json()["status"] == "cancelling"
    build = wait_for(client, "build-cancel", timeout=15)
    assert build["status"] == "cancelled"
    assert client.get("/atlas/builds/build-cancel/atlas").status_code == 409


def test_restart_marks_unfinished_builds_interrupted(tmp_path):
    store = AtlasBuildStore(tmp_path)
    store.create("build-restart", {"inputs": []})
    store.update("build-restart", status="running", phase="mapping")
    AtlasBuildStore(tmp_path).recover()
    build = store.get("build-restart")
    assert build["status"] == "interrupted"
    assert "Retry" in build["error"]
    # A late message from a dead child cannot revive it.
    assert store.update("build-restart", status="running")["status"] == "interrupted"
    assert os.path.isdir(store.directory("build-restart").parent)
