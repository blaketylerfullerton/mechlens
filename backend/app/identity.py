"""Identity checks at the boundaries between features, labels and layouts."""
from __future__ import annotations

from .profiles import RELEASES, same_model
from .sae_cache import SAE_HOOK
from .schema import Trace


def feature_identity(trace: Trace) -> tuple[str, str]:
    record = trace.pass_record("sae")
    if record is None:
        raise ValueError("features have no SAE provenance — run the SAE pass first")
    release, width = record.params.get("release"), record.params.get("width")
    if release not in RELEASES or width not in RELEASES[release].widths:
        raise ValueError(f"unsupported SAE identity: {release!r}/{width!r}")
    if not same_model(record.params.get("model", trace.model), trace.model):
        raise ValueError("SAE provenance belongs to a different model")
    if record.params.get("hook", SAE_HOOK) != SAE_HOOK:
        raise ValueError("SAE provenance belongs to a different hook")
    return str(release), str(width)
