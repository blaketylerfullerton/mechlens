"""SAELens public step API, streamed activations, and atomic inference checkpoints.

The service calls this on its single compute worker while holding the same lock
as tracing. No model weights are updated and no hosted logger is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .store import RunStore


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    layer: int = Field(default=12, ge=0)
    features: int = Field(default=4096, ge=16, le=65536)
    training_tokens: int = Field(default=32768, ge=32, le=100_000_000)
    batch_size: int = Field(default=256, ge=16, le=8192)
    context_size: int = Field(default=128, ge=8, le=2048)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    l1_coefficient: float = Field(default=0.1, ge=0, le=100)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    dataset: Literal["tiny-stories", "text"] = "tiny-stories"
    training_text: str = Field(default="", max_length=2_000_000)
    evaluation_text: str = Field(default="", max_length=200_000)

    @model_validator(mode="after")
    def validate_data(self):
        if self.training_tokens < self.batch_size:
            raise ValueError("Token budget must be at least one batch")
        if self.training_tokens % self.batch_size:
            raise ValueError("Token budget must be a multiple of batch size")
        if self.dataset == "text":
            if not self.training_text.strip() or not self.evaluation_text.strip():
                raise ValueError("Provide separate training and evaluation text")
            if self.training_text.strip() == self.evaluation_text.strip():
                raise ValueError("Evaluation text must differ from training text")
        return self


class Cancelled(Exception):
    pass


def check_cancel(store, run_id):
    if store.get(run_id)["status"] in {"cancelling", "cancelled", "interrupted"}:
        raise Cancelled()


def model_identity(model) -> dict:
    tokenizer = getattr(model, "tokenizer", None)
    return {
        "model": model.cfg.model_name,
        "model_revision": getattr(model.cfg, "model_revision", None),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "normalization": model.cfg.normalization_type,
        "d_in": model.cfg.d_model,
        "model_dtype": str(model.cfg.dtype),
    }


def text_sources(cfg, provenance=None):
    if cfg.dataset == "text":
        return [cfg.training_text], [cfg.evaluation_text], {
            "dataset": "user-text", "train_sha256": hashlib.sha256(cfg.training_text.encode()).hexdigest(),
            "eval_sha256": hashlib.sha256(cfg.evaluation_text.encode()).hexdigest(),
        }
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "roneneldan/TinyStories"
    revision = provenance["revision"] if provenance else HfApi().dataset_info(repo).sha
    train = load_dataset(repo, revision=revision, split="train", streaming=True)
    evaluation = load_dataset(repo, revision=revision, split="validation", streaming=True)
    return (row["text"] for row in train), (row["text"] for row in evaluation), {
        "dataset": repo, "revision": revision, "train_split": "train",
        "eval_split": "validation", "text_field": "text",
    }


def token_sequences(model, texts, context_size, check):
    """Independent documents, no padding or concatenation across boundaries.

    Long documents become context-sized chunks. The first token of every chunk
    is excluded from SAE batches and reconstruction substitution.
    """
    for text in texts:
        check()
        tokens = model.to_tokens(text, prepend_bos=True, truncate=False)
        for start in range(0, tokens.shape[1] - 1, context_size):
            chunk = tokens[:, start:start + context_size]
            if chunk.shape[1] > 1:
                yield chunk


def activation_batches(model, sequences, hook, batch_size, check):
    pending = []
    count = 0
    for tokens in sequences:
        check()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=int(hook.split('.')[1]) + 1)
            x = cache[hook][0, 1:].detach().float()
        pending.append(x)
        count += len(x)
        while count >= batch_size:
            merged = torch.cat(pending)
            yield merged[:batch_size]
            remaining = merged[batch_size:]
            pending = [remaining] if len(remaining) else []
            count = len(remaining)
    raise ValueError("Dataset exhausted before the token budget. Supply more text or reduce the budget.")


def reconstruction_metrics(x, recon, acts):
    mse = (x - recon).square().mean()
    variance = (x - x.mean(0)).square().mean()
    return {"mse": float(mse), "explained_variance": float(1 - mse / variance.clamp_min(1e-12)),
            "l0": float((acts > 0).float().sum(-1).mean())}


def checkpoint(store, run_id, sae, trainer, manifest):
    """Only register fully written immutable directories; never overwrite a good checkpoint."""
    parent = store.root / run_id
    parent.mkdir(exist_ok=True)
    name = f"checkpoint-{trainer.n_training_samples}"
    final = parent / name
    if final.exists():
        return store.get(run_id)["checkpoint"]
    temporary = parent / f".{name}-{time.time_ns()}"
    temporary.mkdir()
    sae.save_model(temporary)
    trainer.save_trainer_state(temporary)
    torch.save({"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None},
               temporary / "rng.pt")
    digest = hashlib.sha256()
    for filename in ("cfg.json", "sae_weights.safetensors"):
        with (temporary / filename).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    manifest = dict(manifest, artifact_id=digest.hexdigest(), tokens=trainer.n_training_samples,
                    resume_supported=True)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    os.replace(temporary, final)
    result = dict(path=str(final.relative_to(store.root)), **manifest)
    store.update(run_id, checkpoint=result, resume_supported=True)
    return result


def train(store: RunStore, run_id: str, model, *, sources=None, resume=False):
    from sae_lens import SAETrainer, StandardTrainingSAE, StandardTrainingSAEConfig
    from sae_lens.config import SAETrainerConfig
    from sae_lens.saes.sae import SAEMetadata

    previous = store.get(run_id)
    cfg = TrainingConfig(**previous["config"])
    check = lambda: check_cancel(store, run_id)
    check()
    if cfg.layer >= model.cfg.n_layers:
        raise ValueError(f"Layer must be below {model.cfg.n_layers}")
    if str(model.cfg.normalization_type).endswith("Pre"):
        raise ValueError("Training requires unprocessed model weights (unset MECHLENS_PROCESS_WEIGHTS)")
    device = str(model.cfg.device)
    hook = f"blocks.{cfg.layer}.hook_resid_post"
    store.update(run_id, phase="preparing", status="running")
    train_texts, eval_texts, data = sources if sources is not None else text_sources(cfg,
        previous["provenance"]["dataset"] if resume else None)
    manifest = dict(model_identity(model), run_id=run_id, layer=cfg.layer, hook=hook,
                    features=cfg.features, architecture="standard", normalize_activations="none",
                    seed=cfg.seed, dataset=data, config=cfg.model_dump(exclude={"training_text", "evaluation_text"}),
                    versions={p: version(p) for p in ("sae-lens", "torch", "transformer-lens")},
                    token_policy="independent document chunks; exclude first position of each chunk")
    store.update(run_id, provenance=manifest)
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    mps_rng = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    # Preserve the process RNG state used by trace sampling.
    cuda_devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.seed)
        sae = StandardTrainingSAE(StandardTrainingSAEConfig(
            d_in=model.cfg.d_model, d_sae=cfg.features, device=device,
            l1_coefficient=cfg.l1_coefficient,
            metadata=SAEMetadata(model_name=model.cfg.model_name, hook_name=hook)))
        batches = activation_batches(model, token_sequences(model, train_texts, cfg.context_size, check),
                                     hook, cfg.batch_size, check)
        trainer = SAETrainer(SAETrainerConfig(total_training_samples=cfg.training_tokens,
            train_batch_size_samples=cfg.batch_size, device=device, lr=cfg.learning_rate,
            lr_end=cfg.learning_rate), sae, batches)
        if resume:
            restored, saved = load_artifact(store, run_id, model)
            if saved["versions"] != manifest["versions"]:
                raise ValueError("Exact resume requires the same training package versions")
            sae.load_state_dict(restored.state_dict())
            del restored
            path = store.root / saved["path"]
            trainer.load_trainer_state(path)
            rng = torch.load(path / "rng.pt", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            if rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng["mps"] is not None:
                torch.mps.set_rng_state(rng["mps"])
        start = time.monotonic()
        starting_tokens = trainer.n_training_samples
        valid_step = False
        last_log = last_save = start
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            model.eval()
            model.requires_grad_(False)
            if resume:
                store.update(run_id, phase="replaying data")
                for _ in range(trainer.n_training_samples // cfg.batch_size):
                    check()
                    next(batches)
            store.update(run_id, phase="training")
            while trainer.n_training_samples < cfg.training_tokens:
                check()
                valid_step = False
                output = trainer.step(next(batches))
                trainer.n_training_steps += 1
                now = time.monotonic()
                done = trainer.n_training_samples
                if not math.isfinite(float(output.loss.detach())):
                    raise ValueError("Non-finite training loss; lower learning rate or inspect activations")
                valid_step = True
                if now - last_log >= 1 or done == cfg.training_tokens or trainer.n_training_steps == 1:
                    with torch.no_grad():
                        metrics = reconstruction_metrics(output.sae_in, output.sae_out, output.feature_acts)
                    elapsed = now - start
                    metrics.update(tokens=done, loss=float(output.loss.detach()), elapsed_s=elapsed,
                                   tokens_per_second=(done - starting_tokens) / max(elapsed, 1e-9),
                                   eta_s=(cfg.training_tokens - done) * elapsed / max(done - starting_tokens, 1),
                                   dead_fraction=float(trainer.dead_neurons.float().mean()),
                                   dead_window_steps=trainer.cfg.dead_feature_window,
                                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None)
                    store.metric(run_id, metrics)
                    store.update(run_id, tokens=done)
                    last_log = now
                if now - last_save >= 60 or trainer.n_training_steps == 1:
                    checkpoint(store, run_id, sae, trainer, manifest)
                    last_save = now
                del output
            check()
            trainer.set_final_sae_metadata()
            checkpoint(store, run_id, sae, trainer, manifest)
            store.update(run_id, phase="evaluating")
            evaluation = evaluate(model, sae, token_sequences(model, eval_texts, cfg.context_size, check), hook, check)
            check()
            store.update(run_id, evaluation=evaluation, phase="complete", status="completed", tokens=cfg.training_tokens)
        finally:
            # Save the latest completed optimizer step even after cancellation.
            try:
                if valid_step:
                    checkpoint(store, run_id, sae, trainer, manifest)
            finally:
                for parameter, flag in zip(model.parameters(), flags):
                    parameter.requires_grad_(flag)
                model.train(was_training)
                if mps_rng is not None:
                    torch.mps.set_rng_state(mps_rng)


@torch.no_grad()
def evaluate(model, sae, sequences, hook, check, limit=8):
    records = []
    sae.eval()
    for tokens in sequences:
        check()
        baseline, cache = model.run_with_cache(tokens, names_filter=[hook], return_type="loss")
        x = cache[hook][0, 1:].float()
        acts = sae.encode(x)
        recon = sae.decode(acts)
        def substitute(value, hook):
            result = value.clone()
            result[:, 1:] = recon.to(value.dtype).unsqueeze(0)
            return result
        altered = model.run_with_hooks(tokens, fwd_hooks=[(hook, substitute)], return_type="loss")
        records.append(dict(reconstruction_metrics(x, recon, acts), baseline_loss=float(baseline),
                            reconstruction_loss=float(altered), tokens=len(x)))
        if len(records) >= limit:
            break
    if not records:
        raise ValueError("Evaluation text produced no usable token sequences")
    total = sum(row["tokens"] for row in records)
    result = {key: sum(row[key] * row["tokens"] for row in records) / total
              for key in ("mse", "explained_variance", "l0", "baseline_loss", "reconstruction_loss")}
    return dict(result, loss_increase=result["reconstruction_loss"] - result["baseline_loss"],
                tokens=total, sequences=len(records), definition="token-weighted per-sequence metrics; first position unchanged")


def load_artifact(store, run_id, model):
    from sae_lens import SAE
    saved = store.get(run_id)["checkpoint"]
    if not saved:
        raise ValueError("This run has no saved checkpoint")
    identity = model_identity(model)
    for key in ("model", "d_in", "normalization", "model_dtype", "model_revision", "tokenizer", "tokenizer_revision"):
        if saved.get(key) != identity.get(key):
            raise ValueError(f"SAE {key} does not match the loaded model")
    if saved["hook"] != f"blocks.{saved['layer']}.hook_resid_post" or saved["normalize_activations"] != "none":
        raise ValueError("Unsupported SAE activation hook or preprocessing")
    path = (store.root / saved["path"]).resolve()
    if not path.is_relative_to(store.root.resolve()):
        raise ValueError("Artifact path is outside the training store")
    digest = hashlib.sha256()
    for name in ("cfg.json", "sae_weights.safetensors"):
        with (path / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    if digest.hexdigest() != saved["artifact_id"]:
        raise ValueError("SAE checkpoint content does not match its registered identity")
    sae = SAE.load_from_disk(path, device=str(model.cfg.device))
    if sae.cfg.d_in != identity["d_in"] or sae.cfg.d_sae != saved["features"]:
        raise ValueError("SAE dimensions do not match the manifest")
    return sae, saved
