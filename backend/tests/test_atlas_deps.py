"""The atlas's build-time dependencies stay build-time.

The atlas is a precomputed table, so serving it needs no projection library at
all: `scripts/build_feature_atlas.py` imports umap-learn and nothing else does.
`umap-learn` is the expensive one — it pulls numba and llvmlite, which is a
JIT toolchain the service has no use for.

scikit-learn is deliberately *not* on the forbidden list for the service.
`transformer_lens` already imports it (and scipy) at module scope, so it was in
the service's module graph long before the atlas existed; asserting otherwise
would be asserting something that was never true. `app.schema` is the strict
case — its own docstring promises pydantic and nothing else — so it is checked
against everything.

Asserted by importing the modules in a *fresh interpreter* and inspecting
`sys.modules`: an in-process check would pass or fail depending on whatever the
rest of the test session had already imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

# What the atlas build added, and what must not reach the service with it.
BUILD_ONLY = ("umap", "numba", "llvmlite")

# app.schema promises more than that: no torch, no numeric stack at all.
SCHEMA_FORBIDDEN = BUILD_ONLY + ("sklearn", "scipy", "torch", "numpy")


def _modules_after_importing(target: str) -> set[str]:
    """Top-level module names loaded in a fresh interpreter that imports `target`."""
    code = (
        "import sys, json;"
        f"import {target};"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_schema_imports_nothing_but_pydantic():
    """The module docstring's own claim, kept honest."""
    loaded = _modules_after_importing("app.schema")
    for module in SCHEMA_FORBIDDEN:
        assert module not in loaded, f"app.schema pulled in {module}"


def test_the_service_does_not_import_a_projection_library():
    loaded = _modules_after_importing("app.service.app")
    for module in BUILD_ONLY:
        assert module not in loaded, f"app.service.app pulled in {module}"


def test_the_label_store_does_not_import_a_projection_library():
    """The layout lives in the label DB, so reading it must not need umap."""
    loaded = _modules_after_importing("app.labels")
    for module in BUILD_ONLY:
        assert module not in loaded, f"app.labels pulled in {module}"


def test_the_build_script_is_the_one_thing_that_needs_umap():
    """The complement of the checks above: if this fails, the dependency was
    dropped rather than confined."""
    source = (BACKEND / "scripts" / "build_feature_atlas.py").read_text()
    assert "umap" in source
