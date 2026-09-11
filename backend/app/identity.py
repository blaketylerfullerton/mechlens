"""Identity checks at the boundaries between features, labels and layouts."""
from __future__ import annotations

from .sae_cache import RELEASE, SAE_HOOK
from .schema import Trace


def feature_identity(trace: Trace) -> tuple[str, str]:
    record = trace.pass_record("sae")
    if record is None:
        raise ValueError("features have no SAE provenance — run the SAE pass first")
    release, width = record.params.get("release"), record.params.get("width")
    if release != RELEASE or width not in {"16k", "65k", "262k"}:
        raise ValueError(f"unsupported SAE identity: {release!r}/{width!r}")
    if record.params.get("model", trace.model) != trace.model:
        raise ValueError("SAE provenance belongs to a different model")
    if record.params.get("hook", SAE_HOOK) != SAE_HOOK:
        raise ValueError("SAE provenance belongs to a different hook")
    return str(release), str(width)
