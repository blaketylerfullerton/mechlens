"""Phase 3: attach Neuronpedia's human-readable labels to a trace's features.

The SAE pass leaves every layer-state holding a list of integers. This pass
turns those integers into text, so a trace on disk says what its features
*mean* rather than only which ones fired.

    python -m app.cli enrich traces/golden-gate.json --labels

Labels land in `Trace.labels` keyed "layer/index", not on each Feature: a
feature recurs about twice per trace, and the text plus a URL copied onto every
occurrence roughly doubles the JSON for no added information. See phase3.md.

This pass ignores the `residuals` argument the Pass protocol hands it — there is
nothing to compute here, only a lookup. Reading from SQLite means enriching a
31-token trace costs milliseconds and no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..labels import DEFAULT_DB_PATH, LabelStore, URL_TEMPLATE, source_set_template
from ..sae_cache import DEFAULT_WIDTH, RELEASE
from ..schema import PassRecord, Trace, label_key


@dataclass
class LabelsPass:
    """Fills Trace.labels for every feature the SAE pass recorded."""

    name: str = field(default="labels", init=False)
    width: str = DEFAULT_WIDTH
    db_path: Path = DEFAULT_DB_PATH
    fetch_missing: bool = False
    max_fetches: int = 200
    verbose: bool = True

    # Injectable so tests can stand in a temp DB, and so a long-lived server
    # can hold one connection open instead of reopening it per request.
    store: LabelStore | None = None

    def run(self, trace: Trace, residuals: np.ndarray | None = None) -> PassRecord:
        wanted = _features_by_layer(trace)
        if not wanted:
            raise ValueError(
                f"trace {trace.trace_id} has no SAE features to label — "
                f"run the SAE pass first (enrich --sae)"
            )

        store = self.store or LabelStore(
            self.db_path,
            width=self.width,
            fetch_missing=self.fetch_missing,
            max_fetches=self.max_fetches,
        )
        owned = self.store is None
        t0 = time.time()

        n_wanted = 0
        explainers: dict[str, int] = {}
        try:
            for layer, features in sorted(wanted.items()):
                n_wanted += len(features)
                for feature, label in store.get_many(layer, features).items():
                    trace.labels[label_key(layer, feature)] = label
                    name = label.explainer or "?"
                    explainers[name] = explainers.get(name, 0) + 1
        finally:
            if owned:
                store.close()

        n_labelled = len(trace.labels)
        if self.verbose:
            print(
                f"labelled {n_labelled}/{n_wanted} distinct features "
                f"across {len(wanted)} layers in {time.time() - t0:.2f}s"
            )

        return PassRecord(
            name=self.name,
            params={
                "release": RELEASE,
                "width": self.width,
                "neuronpedia_model": _model_id(self.width),
                # Enough for a frontend to build every link itself. The URL is a
                # pure function of these three, so storing 6000 of them would be
                # storing the same f-string 6000 times.
                "source_set_template": source_set_template(trace.n_layers, self.width) or "",
                "url_template": URL_TEMPLATE,
                "fetch_missing": self.fetch_missing,
                # Neuronpedia's export does not use one explainer throughout —
                # for gemma-2-2b/16k, layers 16/18/20/22/24 are gemini and the
                # rest gpt-4o-mini. A trace that pools label text across layers
                # needs to be able to see that, so the mix is recorded here.
                "explainers": ", ".join(
                    f"{k}:{v}" for k, v in sorted(explainers.items(), key=lambda kv: -kv[1])
                ),
            },
            stats={
                "features_wanted": float(n_wanted),
                "features_labelled": float(n_labelled),
                "coverage": n_labelled / n_wanted if n_wanted else 0.0,
                "n_layers": float(len(wanted)),
            },
            elapsed_s=time.time() - t0,
        )


def _features_by_layer(trace: Trace) -> dict[int, list[int]]:
    """Distinct feature indices per layer, across every token position.

    Deduping here is the point: a 31-token trace holds ~12,800 feature entries
    but only ~6,200 distinct (layer, feature) pairs, so looking up per
    occurrence would double the work and change nothing.
    """
    wanted: dict[int, set[int]] = {}
    for step in trace.steps:
        for state in step.layers:
            if state.features:
                wanted.setdefault(state.layer, set()).update(f.index for f in state.features)
    return {layer: sorted(features) for layer, features in wanted.items()}


def _model_id(width: str) -> str:
    from ..sae_cache import neuronpedia_id

    return neuronpedia_id(0, width)[0]
