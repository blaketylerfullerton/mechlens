"""Bounded correctness checks; quality measurements never become pass/fail claims."""
from __future__ import annotations

import torch

PROMPTS = (
    "The Golden Gate Bridge is located in",
    "Once upon a time, a little girl found",
    "def add(a, b):\n    return",
    "2 + 2 =",
)


@torch.no_grad()
def checkpoint_agreement(sae, restored, x):
    expected = sae.encode(x)
    actual = restored.encode(x)
    decoded = sae.decode(expected)
    reloaded = restored.decode(actual)
    finite = all(torch.isfinite(t).all().item() for t in (expected, actual, decoded, reloaded))
    return dict(status="passed" if finite and torch.equal(expected, actual) and torch.equal(decoded, reloaded)
                else "failed", samples=len(x),
                max_encoding_delta=float((expected - actual).abs().max()) if finite else None,
                max_reconstruction_delta=float((decoded - reloaded).abs().max()) if finite else None,
                definition="Exact encoding and decoding agreement on identical held-out activations after saving and loading.")


@torch.no_grad()
def huggingface_agreement(model, check):
    """Reference stays on CPU to avoid a second GPU-resident language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens.loading_from_pretrained import get_official_model_name

    revision = getattr(model.cfg, "model_revision", None)
    if not revision:
        return dict(status="not_run", reason="Loaded model has no resolved Hugging Face revision.")
    repository = get_official_model_name(model.cfg.model_name)
    check()
    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision, trust_remote_code=False)
    reference = AutoModelForCausalLM.from_pretrained(repository, revision=revision,
        torch_dtype=model.cfg.dtype, trust_remote_code=False).cpu().eval()
    # Numerical gates are explicit engineering tolerances, not a universal equivalence proof.
    tolerance = 1e-5 if model.cfg.dtype == torch.float32 else 5e-3
    rows = []
    try:
        for prompt in PROMPTS:
            check()
            tokens = model.to_tokens(prompt, prepend_bos=True, truncate=False)
            plain = model.to_tokens(prompt, prepend_bos=False, truncate=False)[0].tolist()
            tokenizer_matches = plain == tokenizer.encode(prompt, add_special_tokens=False)
            wrapped = model(tokens, return_type="logits")[0, -1].float().cpu()
            original = reference(input_ids=tokens.cpu()).logits[0, -1].float()
            if wrapped.shape != original.shape or not torch.isfinite(wrapped).all() or not torch.isfinite(original).all():
                rows.append(dict(prompt=prompt, status="failed", reason="Non-finite logits or vocabulary mismatch"))
                continue
            p, q = wrapped.softmax(-1), original.softmax(-1)
            delta = float((p - q).abs().max())
            tv = float((p - q).abs().sum() / 2)
            rows.append(dict(prompt=prompt, token_ids=tokens[0].tolist(), tokenizer_matches=tokenizer_matches,
                max_probability_delta=delta, total_variation=tv,
                top1_matches=int(p.argmax()) == int(q.argmax()),
                status="passed" if tokenizer_matches and delta <= tolerance and tv <= tolerance else "failed"))
    finally:
        del reference
    return dict(status="passed" if all(row["status"] == "passed" for row in rows) else "failed",
        repository=repository, revision=revision, dtype=str(model.cfg.dtype), reference_device="cpu",
        tolerance=tolerance, prompts=rows,
        definition="Final-position next-token distributions on identical token IDs. Both maximum probability difference and total variation must meet tolerance; top-1 is diagnostic.")
