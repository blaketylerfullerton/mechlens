"""Backward-pass complement to the logit lens: instead of decoding what a
layer's hidden state currently means (static projection through the final
norm + unembedding), decode what direction in that hidden state would most
increase a target token's logit (the local Jacobian, taken via backward-mode
autodiff). Per (layer, position) this surfaces which vocab directions are
causally "on the path" to the prediction, rather than which vocab entry it
already looks like.
"""

from dataclasses import dataclass

import torch


@dataclass
class JacobianLensResult:
    prompt: str
    tokens: list[str]
    target_token: str
    target_token_id: int
    # [pos] -> {top_predictions: [{token, prob}]} — the model's actual
    # top-k next-token prediction at that position (real forward-pass
    # softmax, not a lens approximation), independent of layer.
    next_token_predictions: list[dict]
    # [layer] -> {positions: [pos] -> {token, grad_norm, top_aligned_tokens: [{token, score}]}}
    layers: list[dict]


def jacobian_lens(engine, prompt: str, target_token: str | None = None, top_k: int = 5) -> JacobianLensResult:
    """The target logit is always read at the last position (next-token
    prediction), but the gradient it induces is kept at every sequence
    position per layer: the attention/residual paths that feed the last
    position give it nonzero gradient w.r.t. every earlier position's
    hidden state too, not just its own.
    """
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
    # Decoded per-id (not tokenizer.tokenize(prompt) or
    # convert_ids_to_tokens, both of which leave raw BPE artifacts like
    # 'Ġ'/'Ċ' in place) so this lines up exactly with the sequence
    # positions the forward pass and gradients below are indexed by, while
    # still reading as normal text for display.
    input_ids = inputs["input_ids"][0].tolist()
    tokens = [engine.tokenizer.decode([i]) for i in input_ids]

    outputs = real_model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; [1:] is one entry per
    # transformer layer, matching trace()'s per-layer convention.
    hidden_states = list(outputs.hidden_states[1:])
    for hs in hidden_states:
        hs.retain_grad()

    last_logits = outputs.logits[0, -1, :]
    if target_id is None:
        target_id = int(last_logits.argmax().item())

    # Real forward-pass predictions, one per position — grabbed before the
    # backward pass touches anything, so this is exactly what the model
    # would have predicted next if generation had stopped at each position.
    next_token_predictions = _next_token_predictions(engine, outputs.logits[0], top_k)

    real_model.zero_grad(set_to_none=True)
    last_logits[target_id].backward()

    unembed = real_model.lm_head.weight.detach().float().cpu()  # [vocab, hidden]

    layers = []
    for hs in hidden_states:
        grad = hs.grad[0].float().cpu()  # [seq, hidden]
        scores = grad @ unembed.T  # [seq, vocab]

        positions = []
        for pos in range(grad.shape[0]):
            top = torch.topk(scores[pos], top_k)
            positions.append(
                {
                    "token": tokens[pos],
                    "grad_norm": grad[pos].norm().item(),
                    "top_aligned_tokens": [
                        {"token": engine.tokenizer.decode([idx]), "score": score}
                        for idx, score in zip(top.indices.tolist(), top.values.tolist())
                    ],
                }
            )
        layers.append({"positions": positions})

    return JacobianLensResult(
        prompt=prompt,
        tokens=tokens,
        target_token=engine.tokenizer.decode([target_id]),
        target_token_id=target_id,
        next_token_predictions=next_token_predictions,
        layers=layers,
    )


def _next_token_predictions(engine, logits, top_k: int) -> list[dict]:
    """logits: [seq, vocab]. Per position, the model's actual top-k
    next-token prediction if generation had stopped right there."""
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
    predictions = []
    for pos in range(probs.shape[0]):
        top = torch.topk(probs[pos], top_k)
        predictions.append(
            {
                "top_predictions": [
                    {"token": engine.tokenizer.decode([idx]), "prob": prob}
                    for idx, prob in zip(top.indices.tolist(), top.values.tolist())
                ]
            }
        )
    return predictions
