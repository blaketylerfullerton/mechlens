"""Backward-pass complement to the logit lens: instead of decoding what a
layer's hidden state currently means (static projection through the final
norm + unembedding), decode what direction in that hidden state would most
increase a target token's logit (the local Jacobian, taken via backward-mode
autodiff). Per layer this surfaces which vocab directions are causally "on
the path" to the prediction, rather than which vocab entry it already looks
like.
"""

from dataclasses import dataclass

import torch


@dataclass
class JacobianLensResult:
    prompt: str
    tokens: list[str]
    target_token: str
    target_token_id: int
    # [layer] -> {grad_norm, top_aligned_tokens: [{token, score}]}, taken at the last position
    layers: list[dict]


def jacobian_lens(engine, prompt: str, target_token: str | None = None, top_k: int = 5) -> JacobianLensResult:
    """Only the last sequence position is used, on both ends: the target logit
    is read at the last position (next-token prediction), and only that
    position's gradient is kept per layer (matches the logit_lens convention
    in trace.py).
    """
    tokens = engine.tokenizer.tokenize(prompt)

    target_id = None
    if target_token is not None:
        ids = engine.tokenizer.encode(target_token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"target_token {target_token!r} does not map to a single token")
        target_id = ids[0]

    # This runs a raw forward pass on the wrapped transformers model instead
    # of going through nnsight's trace/.grad machinery: nnsight 0.7's `.grad`
    # proxy (its documented pattern for reading gradients off a non-leaf
    # tensor mid-trace — see nnsight's own
    # modeling/vllm/intervention-gaps/test_4_1.py, which exists specifically
    # to probe whether that pattern still holds) reliably raises
    # MissedProviderError against this model/version, on both a `.save()`d
    # and a live hidden-state proxy. Plain torch autograd on
    # `engine.model._model` (the actual LlamaForCausalLM nnsight wraps) has no
    # such issue, so the Jacobian is computed there directly.
    real_model = engine.model._model
    inputs = engine.tokenizer(prompt, return_tensors="pt").to(real_model.device)

    outputs = real_model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; [1:] is one entry per
    # transformer layer, matching trace()'s per-layer convention.
    hidden_states = list(outputs.hidden_states[1:])
    for hs in hidden_states:
        hs.retain_grad()

    last_logits = outputs.logits[0, -1, :]
    if target_id is None:
        target_id = int(last_logits.argmax().item())

    real_model.zero_grad(set_to_none=True)
    last_logits[target_id].backward()

    unembed = real_model.lm_head.weight.detach().float().cpu()

    layers = []
    for hs in hidden_states:
        grad = hs.grad[0, -1, :].float().cpu()
        scores = unembed @ grad
        top = torch.topk(scores, top_k)
        layers.append(
            {
                "grad_norm": grad.norm().item(),
                "top_aligned_tokens": [
                    {"token": engine.tokenizer.decode([idx]), "score": score}
                    for idx, score in zip(top.indices.tolist(), top.values.tolist())
                ],
            }
        )

    return JacobianLensResult(
        prompt=prompt,
        tokens=tokens,
        target_token=engine.tokenizer.decode([target_id]),
        target_token_id=target_id,
        layers=layers,
    )
