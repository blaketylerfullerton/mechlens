"""Every model-specific fact, in one table.

A profile names a TransformerLens model and, optionally, the SAELens release
whose SAEs read its residual stream. Everything else — the SAE id for a layer,
the hook it was trained on, its Neuronpedia source set — is looked up in
SAELens' own registry rather than written down here, so the two cannot drift.

Only TransformerLens models with SAELens releases are in scope. A model with
no release still traces, lenses, attributes and trains; the SAE, label and
atlas passes just have nothing to encode with.

Layer numbering: a trace's layer L is `blocks.L.hook_resid_post`. Some
releases (gpt2-small-res-jb) are trained on `hook_resid_pre` instead, which in
TransformerLens is the same tensor one block later — `blocks.{L+1}.hook_resid_pre`
is `blocks.L.hook_resid_post`, nothing in between. `id_templates` lists the
spellings to try, so either kind of release maps onto trace layers exactly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

# The trace's hook site. capture.RESID_HOOK must agree with it.
RESID_HOOK = "hook_resid_post"

DEFAULT_MODEL = "gpt2"


@dataclass(frozen=True)
class SAERelease:
    """One SAELens release: a residual-stream SAE per layer, in several widths."""

    release: str
    # Width name -> d_sae. The name is whatever the release calls its variants.
    widths: dict[str, int]
    # Formatted with layer, next (= layer + 1) and width; the first spelling
    # present in the release's saes_map is the SAE for that trace layer.
    id_templates: tuple[str, ...]
    # File under the data directory holding this release's labels and atlas.
    label_db: str

    @property
    def default_width(self) -> str:
        return next(iter(self.widths))

    def check_width(self, width: str | None) -> str:
        width = width or self.default_width
        if width not in self.widths:
            raise ValueError(f"{self.release} has no width {width!r}; choose from {', '.join(self.widths)}")
        return width

    def sae_id(self, layer: int, width: str | None = None) -> str:
        width = self.check_width(width)
        saes = _directory()[self.release].saes_map
        for template in self.id_templates:
            candidate = template.format(layer=layer, next=layer + 1, width=width)
            if candidate in saes:
                return candidate
        raise KeyError(f"{self.release} has no {width} SAE for layer {layer}")

    def neuronpedia_id(self, layer: int, width: str | None = None) -> tuple[str, str]:
        full = _directory()[self.release].neuronpedia_id.get(self.sae_id(layer, width))
        if full is None:
            raise KeyError(f"SAELens has no Neuronpedia id for {self.release}/{self.sae_id(layer, width)}")
        model_id, source_set = full.split("/", 1)
        return model_id, source_set


@dataclass(frozen=True)
class ModelProfile:
    name: str  # TransformerLens' cfg.model_name, which is what traces record
    load_name: str  # what HookedTransformer.from_pretrained is given
    repository: str  # Hugging Face repo, as the training selector shows it
    n_layers: int
    aliases: tuple[str, ...] = ()
    gated: bool = False
    saes: SAERelease | None = None
    # Only what TransformerLens loads; repos also carry ONNX, GGUF, TFLite.
    model_files: tuple[str, ...] = ("*.json", "*.safetensors", "*.txt", "tokenizer.model")
    names: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset((self.name, self.load_name, self.repository, *self.aliases)))


GEMMA_SCOPE = SAERelease(
    release="gemma-scope-2b-pt-res-canonical",
    widths={"16k": 16384, "65k": 65536, "262k": 262144},
    id_templates=("layer_{layer}/width_{width}/canonical",),
    label_db="neuronpedia.db",
)

GPT2_RES_JB = SAERelease(
    release="gpt2-small-res-jb",
    widths={"24k": 24576},
    id_templates=("blocks.{next}.hook_resid_pre", "blocks.{layer}.hook_resid_post"),
    label_db="labels-gpt2-small-res-jb.db",
)

PROFILES: dict[str, ModelProfile] = {
    p.name: p
    for p in (
        ModelProfile("gpt2", "gpt2", "openai-community/gpt2", n_layers=12, aliases=("gpt2-small",),
                     saes=GPT2_RES_JB),
        # BASE model, not -it: Gemma Scope SAEs are trained on base ("pt") activations.
        ModelProfile("gemma-2-2b", "gemma-2-2b", "google/gemma-2-2b", n_layers=26, gated=True, saes=GEMMA_SCOPE),
        ModelProfile("TinyStories-1M", "tiny-stories-1M", "roneneldan/TinyStories-1M", n_layers=8),
    )
}

RELEASES: dict[str, SAERelease] = {p.saes.release: p.saes for p in PROFILES.values() if p.saes}


def get_profile(name: str | None = None) -> ModelProfile:
    """By TransformerLens name, alias or Hugging Face repo; None is the default."""
    name = name or default_model()
    for profile in PROFILES.values():
        if name in profile.names:
            return profile
    raise KeyError(f"unknown model {name!r}; supported: {', '.join(PROFILES)}")


def find_profile(name: str | None) -> ModelProfile | None:
    try:
        return get_profile(name) if name else None
    except KeyError:
        return None


def default_model() -> str:
    return os.environ.get("MECHLENS_MODEL") or DEFAULT_MODEL


def default_release() -> SAERelease | None:
    return get_profile().saes


def release_for(model: str | None) -> SAERelease | None:
    """The SAE release a model's profile names, or None (unknown model, or none published)."""
    profile = find_profile(model)
    return profile.saes if profile else None


def get_release(release: str | None = None) -> SAERelease:
    if release is None:
        found = default_release()
        if found is None:
            raise ValueError(f"{default_model()} has no SAE release")
        return found
    if release not in RELEASES:
        raise ValueError(f"unsupported SAE release {release!r}")
    return RELEASES[release]


def model_for_release(release: str) -> str | None:
    """The model a release's SAEs read, as traces name it."""
    return next((p.name for p in PROFILES.values() if p.saes and p.saes.release == release), None)


def same_model(a: str | None, b: str | None) -> bool:
    """SAELens and TransformerLens spell some models differently ("gpt2-small" / "gpt2")."""
    if a == b:
        return True
    pa, pb = find_profile(a), find_profile(b)
    return pa is not None and pa is pb


_HOOK = re.compile(r"blocks\.(\d+)\.(hook_resid_post|hook_resid_pre)$")


def reads_layer(hook_name: str | None, layer: int) -> bool:
    """Whether an SAE trained at `hook_name` reads trace layer `layer`'s resid_post."""
    match = _HOOK.match(hook_name or "")
    if not match:
        return False
    block, site = int(match[1]), match[2]
    return block == layer if site == RESID_HOOK else block == layer + 1


@lru_cache(maxsize=1)
def _directory():
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    return get_pretrained_saes_directory()
