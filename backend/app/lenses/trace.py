"""Per-layer logit lens: hidden states, attention patterns, and "if
generation stopped here" projections through the final norm + unembedding.

Attention weights only show up on `self_attn.output` when the model is
loaded with attn_implementation="eager" — sdpa/flash attention never
materializes the full attention matrix, so eager is required (and slower)
whenever you need attention patterns, not just hidden states.
"""

from dataclasses import dataclass

import torch


@dataclass
class TraceResult:
    prompt: str
    tokens: list[str]
    predicted_next_token: str
    hidden_states: list[list[list[float]]]      # [layer][token][dim]
    attention: list[list[list[list[float]]]]    # [layer][head][query][key]
    logit_lens: list[list[dict]]                 # [layer][top-k] -> {token, prob}, taken at the last position


def trace(engine, prompt: str, top_k: int = 5) -> TraceResult:
    tokens = engine.tokenizer.tokenize(prompt)

    # Declared *before* the trace context: nnsight defers/executes the
    # `with` block's body separately from this frame, so assigning these
    # lists inside it would never propagate back out (UnboundLocalError
    # on exit). Mutating them via .append() from inside works fine since
    # it's a closure over the same list object, not a reassignment.
    hidden_states, attentions, logit_lens = [], [], []

    with engine.model.trace(prompt):
        for layer in engine.model.model.layers:
            # self_attn.output must be requested before layer.output — nnsight
            # schedules saves in the order they're referenced, and self_attn
            # runs before the parent layer finishes, so asking for it after
            # layer.output raises MissedProviderError ("out of order").
            #
            # attn_implementation="eager" (forced in Engine.__init__) means every
            # Llama/Gemma-style self_attn returns (attn_output, attn_weights, ...),
            # so index 1 is always populated here.
            attentions.append(layer.self_attn.output[1].save())

            hs = layer.output[0].save()
            hidden_states.append(hs)

            # "if generation stopped here": push this layer's residual
            # stream through the final norm + unembedding early.
            normed = engine.model.model.norm(hs)
            logit_lens.append(engine.model.lm_head(normed).save())

        final_logits = engine.model.lm_head.output.save()

    return TraceResult(
        prompt=prompt,
        tokens=tokens,
        predicted_next_token=_decode_argmax(engine, final_logits),
        hidden_states=[hs.float().cpu().tolist() for hs in hidden_states],
        # attn_weights keeps its (batch, head, query, key) shape even
        # though hidden_states above already comes back batch-less on
        # this transformers version — drop the batch=1 dim so the actual
        # output matches the [layer][head][query][key] shape promised above.
        attention=[a[0].float().cpu().tolist() for a in attentions],
        logit_lens=[_topk(engine, logits, top_k) for logits in logit_lens],
    )


def _topk(engine, logits, k: int) -> list[dict]:
    # logits can come back as (batch, seq, vocab) or (seq, vocab) depending
    # on the call site (some transformers versions drop the batch dim for
    # a manually-invoked submodule) — `...` absorbs whichever leading dims
    # are present so this always lands on the last token's vocab vector.
    probs = torch.softmax(logits[..., -1, :].float(), dim=-1)
    top = torch.topk(probs, k)
    return [
        {"token": engine.tokenizer.decode([idx]), "prob": prob}
        for idx, prob in zip(top.indices.tolist(), top.values.tolist())
    ]


def _decode_argmax(engine, logits) -> str:
    return engine.tokenizer.decode([logits[..., -1, :].argmax().item()])
