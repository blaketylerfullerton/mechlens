"""Zero one attention head's contribution before it's mixed back into the
residual stream, then generate — a quick way to see what a head was doing
by removing it.
"""


def ablate_head(engine, prompt: str, layer_idx: int, head_idx: int, max_new_tokens: int = 30) -> str:
    n_heads = engine.model.config.num_attention_heads
    head_dim = engine.model.config.hidden_size // n_heads
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

    with engine.model.generate(prompt, max_new_tokens=max_new_tokens):
        o_proj_input = engine.model.model.layers[layer_idx].self_attn.o_proj.input[0]
        # o_proj's input is (..., hidden_size) — batch/seq dims vary (e.g.
        # flattened to (seq_len, hidden) during generate on some transformers
        # versions), so index only the last dim rather than assuming a rank.
        o_proj_input[..., lo:hi] = 0
        out = engine.model.generator.output.save()

    return engine.tokenizer.decode(out[0], skip_special_tokens=True)
