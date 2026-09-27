"""`mechlens download`: fetch model weights and SAEs ahead of time.

Other commands also download what they are missing on first use; running this
first just avoids waiting on a multi-GB fetch when the server starts.
"""
from __future__ import annotations

from .profiles import get_profile


def download(skip_saes: bool = False, model: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    from transformer_lens.loading_from_pretrained import get_official_model_name

    profile = get_profile(model)
    # The repo TransformerLens itself fetches from, so the cache entry is the
    # one `mechlens serve` reads ("gpt2", not its "openai-community/gpt2" alias).
    repo = get_official_model_name(profile.load_name)
    print(f"Downloading {profile.name} from {repo}...", flush=True)
    try:
        snapshot_download(repo, allow_patterns=list(profile.model_files))
    except GatedRepoError:
        raise SystemExit(
            f"{repo} is gated: accept the license at https://hf.co/{repo}, "
            "then run `huggingface-cli login` and try again.") from None
    if not skip_saes and profile.saes is not None:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        spec = profile.saes
        release = get_pretrained_saes_directory()[spec.release]
        # One SAE per layer at the default width, not every variant the
        # release publishes — Gemma Scope alone would otherwise be hundreds of GB.
        paths = [f"{release.saes_map[spec.sae_id(layer)]}/*" for layer in range(profile.n_layers)]
        print(f"Downloading {len(paths)} {spec.default_width} SAEs from {release.repo_id}...", flush=True)
        snapshot_download(release.repo_id, allow_patterns=paths)
    elif profile.saes is None:
        print(f"{profile.name} has no published SAE release; skipping SAEs.", flush=True)
    print("Done. `mechlens serve` will now load everything from ~/.cache/huggingface.", flush=True)
