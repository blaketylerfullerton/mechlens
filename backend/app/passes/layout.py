"""Phase 7: attach the atlas's positions to the features a trace reports.

The SAE pass says *which* features fired. This pass says where to draw them,
by looking each one up in a prebuilt atlas:

    python -m app.cli enrich traces/golden-gate.json --layout

Nothing is computed here. The atlas is a build artifact (`scripts/
build_feature_atlas.py`), and this is a keyed read out of the same SQLite file
the labels live in — so the pass loads no model, no SAE, and no projection
library, and enriching a 31-token trace costs milliseconds.

Positions land in `Trace.layout` keyed "layer/index" for the reason
`Trace.labels` is: a feature recurs about twice per trace, so three floats
copied onto every occurrence would be the same numbers written out thousands
of times.

Three outcomes, and they are deliberately three rather than two:

    atlas present, feature placed    -> a position in `Trace.layout`
    atlas present, feature unplaced  -> absent from `Trace.layout`,
                                        counted in `features_unplaced`
    no atlas at all                  -> `Trace.layout` empty, and the record
                                        says `atlas_available: false`

An empty `Trace.layout` is ambiguous on its own — no atlas, or an atlas that
placed nothing — so the pass record is what disambiguates, and it is written
in every case including the one where there was nothing to look up. Inventing
a position, or zeroing one, would make an unplaceable feature indistinguishable
from one the atlas puts at the origin; that is the failure this shape exists to
prevent, and `LabelStore.layout` is built the same way for the same reason.

Which atlas: several coexist by design (decoder directions and explanation
embeddings answer different questions and neither subsumes the other), so this
pass asks for one *source* by name rather than for whatever was built most
recently. See design.md decision 2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..labels import DEFAULT_DB_PATH, LabelStore
from ..identity import feature_identity
from ..schema import NodePosition, PassRecord, Trace, label_key
from .labels import _features_by_layer

# The source representation the interface shows by default. Both atlases are
# built and kept; this one is the only layout that yields nameable areas, and
# it is what the shipped idle asset was sampled from. design.md decision 2.
DEFAULT_ATLAS_SOURCE = "labels"


@dataclass
class LayoutPass:
    """Fills Trace.layout for every feature the SAE pass recorded."""

    name: str = field(default="layout", init=False)
    db_path: Path = DEFAULT_DB_PATH
    # Pin one atlas outright. None means "the most recent one built from
    # `source`", which is what a deployment normally wants.
    atlas_version: str | None = None
    source: str | None = DEFAULT_ATLAS_SOURCE
    verbose: bool = True

    # Injectable for the same reasons LabelsPass takes one: a test stands in a
    # temp DB, and a long-lived server holds one connection open rather than
    # reopening it per request.
    store: LabelStore | None = None

    def run(self, trace: Trace, residuals: np.ndarray | None = None) -> PassRecord:
        wanted = _features_by_layer(trace)
        if not wanted:
            raise ValueError(
                f"trace {trace.trace_id} has no SAE features to place — "
                f"run the SAE pass first (enrich --sae)"
            )

        release, width = feature_identity(trace)
        trace.layout = {}
        pairs = [(layer, feature) for layer, features in wanted.items() for feature in features]
        store = self.store or LabelStore(self.db_path)
        owned = self.store is None
        t0 = time.time()

        try:
            record = store.atlas_record(self.atlas_version, source=self.source, release=release, width=width)
            # No atlas is a legitimate deployment state, not an error: the
            # label DB ships without one until `build_feature_atlas.py` has
            # run. The trace keeps its features and gains no positions.
            if record is None:
                if self.verbose:
                    print(
                        f"no atlas available — {len(pairs)} features left unplaced "
                        f"(build one with scripts/build_feature_atlas.py)"
                    )
                return PassRecord(
                    name=self.name,
                    params={
                        "atlas_available": False,
                        "atlas_source_requested": self.source or "",
                        "atlas_version_requested": self.atlas_version or "",
                    },
                    stats={
                        "features_wanted": float(len(pairs)),
                        "features_placed": 0.0,
                        "features_unplaced": float(len(pairs)),
                        "coverage": 0.0,
                        "n_layers": float(len(wanted)),
                    },
                    elapsed_s=time.time() - t0,
                )

            if (record.release, record.width) != (release, width):
                raise ValueError("atlas release/width does not match the trace's SAE dictionary")
            placed = store.layout(record.atlas_version, pairs)
        finally:
            if owned:
                store.close()

        # Only pairs the atlas actually holds. A pair missing from `placed`
        # stays missing from `trace.layout`.
        for (layer, feature), row in placed.items():
            trace.layout[label_key(layer, feature)] = NodePosition(
                x=row.x, y=row.y, z=row.z, cluster=row.cluster
            )

        n_placed = len(placed)
        n_unplaced = len(pairs) - n_placed
        if self.verbose:
            print(
                f"placed {n_placed}/{len(pairs)} distinct features across "
                f"{len(wanted)} layers against atlas {record.atlas_version} "
                f"in {time.time() - t0:.2f}s"
            )

        metrics = record.metrics
        params = record.params
        return PassRecord(
            name=self.name,
            params={
                "atlas_available": True,
                # The atlas's identity travels with the positions so a stale or
                # mismatched layout is detectable rather than silent — two
                # traces placing one feature differently were drawn against
                # different atlases, and this is what says so.
                "atlas_version": record.atlas_version,
                "positions_sha256": record.positions_sha256,
                "release": record.release,
                "width": record.width,
                "seed": record.seed,
                # What "near" means here. A label-source layout maps
                # descriptions of features rather than the model's own
                # computation, and the interface has to be able to say which
                # of the two it is showing.
                "source": str(params.get("source", "")),
                "layers": str(params.get("layers", "")),
                "n_atlas_features": record.n_features,
                "n_atlas_clusters": record.n_clusters,
            },
            stats={
                "features_wanted": float(len(pairs)),
                "features_placed": float(n_placed),
                "features_unplaced": float(n_unplaced),
                "coverage": n_placed / len(pairs) if pairs else 0.0,
                "n_layers": float(len(wanted)),
                # Carried onto the trace so a client can state how much of the
                # original neighbourhood structure survived the projection
                # without a second round trip for the atlas record. This is the
                # atlas's peer to `explained_variance` and
                # `final_layer_agreement`, and it is meant to be shown.
                "knn_preservation": float(metrics.get("knn_preservation", 0.0)),
                "knn_k": float(metrics.get("knn_k", 0.0)),
            },
            elapsed_s=time.time() - t0,
        )
