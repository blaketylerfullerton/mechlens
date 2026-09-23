"""`mechlens download`: the one place model weights and SAEs are fetched.

Every other command runs with Hugging Face offline (see command.block_hub_downloads),
so on a fresh machine this has to be run once before `mechlens serve`.
"""
from __future__ import annotations

MODEL_REPO = "google/gemma-2-2b"
# Only what TransformerLens loads; the repo also carries a ~10GB .gguf we never use.
MODEL_FILES = ["*.json", "*.safetensors", "tokenizer.model"]


def download(skip_saes: bool = False) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    print(f"Downloading {MODEL_REPO} (~5GB)...", flush=True)
    try:
        snapshot_download(MODEL_REPO, allow_patterns=MODEL_FILES)
    except GatedRepoError:
        raise SystemExit(
            f"{MODEL_REPO} is gated: accept the license at https://hf.co/{MODEL_REPO}, "
            "then run `huggingface-cli login` and try again.") from None
    if not skip_saes:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        from .sae_cache import RELEASE
        release = get_pretrained_saes_directory()[RELEASE]
        paths = [f"{path}/*" for path in release.saes_map.values()]
        print(f"Downloading {len(paths)} Gemma Scope SAEs from {release.repo_id} (~8GB)...", flush=True)
        snapshot_download(release.repo_id, allow_patterns=paths)
    print("Done. `mechlens serve` will now load everything from ~/.cache/huggingface.", flush=True)
