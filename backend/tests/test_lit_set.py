"""`frontend/src/lib/lit.ts`, exercised against real trace shapes.

Which features a trace lights is where almost every honesty requirement on the
brain view actually lives — the scope, the top-k slice, the BOS exclusion, the
unplaced features, the partial-coverage note. None of that is visual, so none
of it needs a browser to check, and leaving it unchecked because it happens to
be written in TypeScript would be leaving the load-bearing part untested.

Node 22 runs TypeScript directly (`--experimental-strip-types`), so this
imports the real module rather than a transcription of it. A transcription
would fail exactly the way the original could, which is the failure mode
`test_shape_ellipsoid_matches_the_typescript_it_ports` exists to avoid.
"""

from __future__ import annotations

import json
import re
import tempfile
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
LIT = FRONTEND / "src" / "lib" / "lit.ts"


def _node_available() -> bool:
    try:
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    major = int(out.stdout.strip().lstrip("v").split(".")[0])
    return major >= 22


pytestmark = pytest.mark.skipif(
    not LIT.exists() or not _node_available(),
    reason="needs the frontend sources and node >= 22 for --experimental-strip-types",
)


def _stage(into: Path) -> Path:
    """Copy the module and what it imports, with node-resolvable specifiers.

    Vite resolves `./atlas` without an extension and node's ESM loader does
    not, so the copies get `.ts` appended on relative imports. A mechanical
    rewrite of the specifier only — no logic is restated, which is the whole
    point of running the real file.
    """
    lib = FRONTEND / "src" / "lib"
    for name in ("lit.ts", "atlas.ts", "api-types.ts", "kdtree.ts"):
        source = (lib / name).read_text()
        source = re.sub(r"(from\s+'\./[A-Za-z0-9_-]+)'", r"\1.ts'", source)
        (into / name).write_text(source)
    return into / "lit.ts"


def run_lit(trace: dict, scope: str, selection: dict | None) -> dict:
    """Call `litSet` in the real module and hand back its result."""
    with tempfile.TemporaryDirectory() as tmp:
        staged = _stage(Path(tmp))
        harness = Path(tmp) / "check.mts"
        harness.write_text(
            f"import {{ litSet }} from {json.dumps(str(staged))};\n"
            f"console.log(JSON.stringify(litSet("
            f"{json.dumps(trace)}, {json.dumps(scope)}, {json.dumps(selection)})));\n"
        )
        try:
            proc = subprocess.run(
                ["node", "--experimental-strip-types", str(harness)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - broken harness
            raise AssertionError(f"node failed:\n{exc.stderr}") from exc
    return json.loads(proc.stdout.strip().splitlines()[-1])


# -- trace fixtures ---------------------------------------------------------


def _state(layer: int, features: list[tuple[int, float]], l0: int | None = 78):
    return {
        "layer": layer,
        "resid_norm": 1.0,
        "logit_lens": None,
        "features": [{"index": i, "activation": a} for i, a in features],
        "l0": l0,
        "edges": [],
    }


def _step(index: int, layers: list[dict]):
    return {
        "step": index,
        "token": {"position": index, "token_id": index, "text": f"t{index}", "source": "prompt"},
        "logits": {"top_k": [], "entropy": 0.0, "chosen": None},
        "layers": layers,
    }


def _trace(steps: list[dict], layout: dict | None = None, passes: list[dict] | None = None):
    return {
        "schema_version": "1.4",
        "trace_id": "t",
        "created_at": "2026-01-01T00:00:00Z",
        "model": "gemma-2-2b",
        "device": "cuda",
        "dtype": "bf16",
        "n_layers": 2,
        "d_model": 2304,
        "normalization": None,
        "prompt": "p",
        "completion": "c",
        "n_prompt_tokens": 1,
        "n_generated_tokens": 1,
        "stop_reason": "max_tokens",
        "elapsed_s": 0.0,
        "residuals": None,
        "passes": passes if passes is not None else [{"name": "sae", "params": {}, "stats": {}, "elapsed_s": 0.0, "created_at": "x"}],
        "steering": None,
        "labels": {},
        "layout": layout or {},
        "steps": steps,
    }


def _position(x: float, y: float, z: float, cluster: int = 0):
    return {"x": x, "y": y, "z": z, "cluster": cluster}


# Two token positions, two layers, distinct features and activations.
BASIC = _trace(
    steps=[
        _step(0, [_state(0, [(1, 9.0)]), _state(1, [(2, 9.0)])]),  # BOS
        _step(1, [_state(0, [(10, 3.0), (11, 1.0)]), _state(1, [(20, 2.0)])]),
        _step(2, [_state(0, [(10, 5.0)]), _state(1, [(21, 4.0)])]),
    ],
    layout={
        "0/10": _position(0.1, 0.2, 0.3),
        "0/11": _position(0.4, 0.5, 0.6, cluster=1),
        "1/20": _position(0.7, 0.8, 0.9, cluster=2),
        "1/21": _position(-0.1, -0.2, -0.3, cluster=-1),
    },
)


# -- 4.6 scope --------------------------------------------------------------


def test_cell_scope_lights_only_the_selected_layer_and_token():
    got = run_lit(BASIC, "cell", {"layer": 0, "position": 1})
    assert [(n["layer"], n["feature"], n["activation"]) for n in got["nodes"]] == [
        (0, 10, 3.0),
        (0, 11, 1.0),
    ]
    assert got["scope"] == "cell"
    # One cell cannot cover a feature twice, so no aggregation is claimed.
    assert got["aggregation"] is None


def test_token_scope_lights_every_layer_at_one_token():
    got = run_lit(BASIC, "token", {"layer": 0, "position": 1})
    assert {(n["layer"], n["feature"]) for n in got["nodes"]} == {(0, 10), (0, 11), (1, 20)}


def test_trace_scope_aggregates_a_repeated_feature_by_max():
    """Feature 0/10 fires at 3.0 on one token and 5.0 on another."""
    got = run_lit(BASIC, "trace", None)
    assert got["aggregation"] == "max"
    by_key = {(n["layer"], n["feature"]): n["activation"] for n in got["nodes"]}
    assert by_key[(0, 10)] == 5.0, "max, not the last seen or the sum"
    assert by_key[(1, 21)] == 4.0


def test_nodes_come_back_brightest_first():
    got = run_lit(BASIC, "trace", None)
    activations = [n["activation"] for n in got["nodes"]]
    assert activations == sorted(activations, reverse=True)


# -- 4.7 BOS ----------------------------------------------------------------


def test_bos_features_are_never_lit_at_any_scope():
    """Position 0 holds features 0/1 and 1/2, and no scope may light them."""
    for scope, selection in (
        ("trace", None),
        ("token", {"layer": 0, "position": 2}),
        ("cell", {"layer": 0, "position": 2}),
    ):
        got = run_lit(BASIC, scope, selection)
        lit = {(n["layer"], n["feature"]) for n in got["nodes"]}
        assert (0, 1) not in lit and (1, 2) not in lit, f"BOS leaked at {scope} scope"


def test_selecting_bos_itself_lights_nothing_and_says_why():
    got = run_lit(BASIC, "cell", {"layer": 0, "position": 0})
    assert got["nodes"] == []
    assert got["bosExcluded"] is True
    assert "BOS" in got["emptyReason"]


def test_trace_scope_records_that_bos_was_excluded():
    assert run_lit(BASIC, "trace", None)["bosExcluded"] is True


# -- 4.4 unplaced -----------------------------------------------------------


def test_a_feature_the_atlas_cannot_place_is_carried_with_its_activation():
    trace = _trace(
        steps=[_step(0, [_state(0, [])]), _step(1, [_state(0, [(10, 3.0), (99, 7.0)])])],
        layout={"0/10": _position(0.1, 0.2, 0.3)},
    )
    got = run_lit(trace, "token", {"layer": 0, "position": 1})

    assert [(n["layer"], n["feature"]) for n in got["nodes"]] == [(0, 10)]
    assert got["unplaced"] == [{"layer": 0, "feature": 99, "activation": 7.0}]
    # It counts as shown — it fired, it is simply not drawable.
    assert got["shown"] == 2


def test_no_position_is_invented_for_an_unplaced_feature():
    trace = _trace(
        steps=[_step(0, [_state(0, [])]), _step(1, [_state(0, [(99, 7.0)])])],
        layout={},
    )
    got = run_lit(trace, "token", {"layer": 0, "position": 1})
    assert got["nodes"] == []
    assert len(got["unplaced"]) == 1
    for key in ("x", "y", "z", "cluster"):
        assert key not in got["unplaced"][0]


# -- 4.8 shown versus fired -------------------------------------------------


def test_shown_and_fired_travel_together():
    """The pass keeps the top-k of an l0 that is far larger, and the view has
    to be able to say so rather than letting the slice read as the whole."""
    trace = _trace(
        steps=[_step(0, [_state(0, [])]), _step(1, [_state(0, [(10, 3.0), (11, 1.0)], l0=78)])],
        layout={},
    )
    got = run_lit(trace, "cell", {"layer": 0, "position": 1})
    assert got["shown"] == 2
    assert got["fired"] == 78


def test_fired_sums_the_cells_in_scope():
    got = run_lit(BASIC, "token", {"layer": 0, "position": 1})
    assert got["fired"] == 78 * 2, "two layers at one token"


def test_fired_is_null_when_no_cell_reported_one():
    trace = _trace(
        steps=[_step(0, [_state(0, [])]), _step(1, [_state(0, [(10, 3.0)], l0=None)])],
        layout={},
    )
    assert run_lit(trace, "cell", {"layer": 0, "position": 1})["fired"] is None


# -- 4.10 empty and partial -------------------------------------------------


def test_a_trace_without_the_sae_pass_says_that_rather_than_showing_nothing():
    trace = _trace(
        steps=[_step(0, [_state(0, [], l0=None)]), _step(1, [_state(0, [], l0=None)])],
        passes=[],
    )
    got = run_lit(trace, "trace", None)
    assert got["nodes"] == []
    assert "without the SAE pass" in got["emptyReason"]


def test_a_layer_with_no_data_is_named_not_shown_as_one_where_nothing_fired():
    """Partial coverage — the SAE pass ran on layer 0 only."""
    trace = _trace(
        steps=[
            _step(0, [_state(0, []), _state(1, [], l0=None)]),
            _step(1, [_state(0, [(10, 3.0)]), _state(1, [], l0=None)]),
        ],
        layout={"0/10": _position(0.1, 0.2, 0.3)},
    )
    got = run_lit(trace, "trace", None)
    assert got["layersWithData"] == [0]
    assert got["layersWithoutData"] == [1]


def test_no_trace_is_its_own_state():
    got = run_lit(None, "trace", None)
    assert got["nodes"] == [] and got["emptyReason"] == "no trace"


# -- 4.9 the k-d tree behind node picking -----------------------------------


def run_js(body: str) -> object:
    """Evaluate `body` against the staged lib modules and return its JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        _stage(Path(tmp))
        harness = Path(tmp) / "check.mts"
        harness.write_text(body.replace("__LIB__", json.dumps(str(Path(tmp)))[1:-1]))
        try:
            proc = subprocess.run(
                ["node", "--experimental-strip-types", str(harness)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise AssertionError(f"node failed:\n{exc.stderr}") from exc
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_tree_finds_the_same_nearest_node_as_a_brute_force_scan():
    """The index is an optimisation, so it has to agree with the thing it
    replaces — on a clumped cloud, which is what UMAP output actually is."""
    got = run_js(
        """
import { KdTree } from '__LIB__/kdtree.ts';
// Three tight clumps plus stragglers: the distribution a balanced split has
// to cope with, and the one a naive axis-cycling tree degenerates on.
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const centres = [[0.6, 0.1, -0.2], [-0.5, 0.3, 0.4], [0.0, -0.4, 0.7]];
const points = [];
for (let i = 0; i < 600; i++) {
  const c = centres[i % 3];
  points.push({ x: c[0] + (rand() - 0.5) * 0.2, y: c[1] + (rand() - 0.5) * 0.2,
                z: c[2] + (rand() - 0.5) * 0.2, id: i });
}
const tree = new KdTree(points);
const misses = [];
for (let q = 0; q < 200; q++) {
  const query = { x: rand() * 2 - 1, y: rand() * 2 - 1, z: rand() * 2 - 1 };
  const radius = 0.35;
  let best = null, bestD = radius * radius;
  for (const p of points) {
    const d = (p.x-query.x)**2 + (p.y-query.y)**2 + (p.z-query.z)**2;
    if (d < bestD) { bestD = d; best = p; }
  }
  const got = tree.nearest(query, radius);
  if ((best === null) !== (got === null)) misses.push(q);
  else if (best !== null && got !== null && got.id !== best.id) misses.push(q);
}
console.log(JSON.stringify({ misses, n: points.length }));
"""
    )
    assert got["misses"] == [], "the tree disagreed with a brute-force scan"


def test_a_query_outside_the_radius_reports_nothing():
    """Without this a pointer anywhere on screen picks *some* node, and the
    interface names a feature for a click on empty space."""
    got = run_js(
        """
import { KdTree } from '__LIB__/kdtree.ts';
const tree = new KdTree([{ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 }]);
console.log(JSON.stringify({
  far: tree.nearest({ x: 5, y: 5, z: 5 }, 0.5),
  near: tree.nearest({ x: 0.01, y: 0, z: 0 }, 0.5),
}));
"""
    )
    assert got["far"] is None
    assert got["near"] == {"x": 0, "y": 0, "z": 0}


def test_an_empty_set_is_pickable_without_throwing():
    got = run_js(
        """
import { KdTree } from '__LIB__/kdtree.ts';
console.log(JSON.stringify({ hit: new KdTree([]).nearest({ x: 0, y: 0, z: 0 }, 1) }));
"""
    )
    assert got["hit"] is None


# -- 4.3 prominence ---------------------------------------------------------


def test_prominence_is_a_pure_per_node_ratio():
    """A dim node beside a bright cluster stays dim: nothing may borrow
    brightness from a neighbour, so this is a function of one activation."""
    got = run_js(
        """
import { prominence } from '__LIB__/lit.ts';
console.log(JSON.stringify({
  bright: prominence(9, 9),
  dim: prominence(0.5, 9),
  zeroMax: prominence(3, 0),
  clamped: prominence(20, 9),
}));
"""
    )
    assert got["bright"] == 1.0
    assert got["dim"] < 0.1
    assert got["zeroMax"] == 0
    assert got["clamped"] == 1.0
