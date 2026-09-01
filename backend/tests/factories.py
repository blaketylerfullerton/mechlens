"""Synthetic traces, so tests that are not about the model do not load one."""

from __future__ import annotations

import numpy as np

from app.capture import CaptureResult
from app.schema import LayerState, LogitSummary, TokenInfo, TokenStep, TopToken, Trace

N_TOKENS, N_LAYERS, D_MODEL = 4, 3, 8


def make_result(
    n_tokens: int = N_TOKENS,
    n_layers: int = N_LAYERS,
    d_model: int = D_MODEL,
    seed: int = 0,
    trace_id: str = "test0001",
) -> CaptureResult:
    rng = np.random.default_rng(seed)
    residuals = rng.standard_normal((n_tokens, n_layers, d_model), dtype=np.float32)
    n_prompt = min(2, n_tokens)

    steps = [
        TokenStep(
            step=pos,
            token=TokenInfo(
                position=pos,
                token_id=100 + pos,
                text=f" t{pos}",
                source="prompt" if pos < n_prompt else "generated",
            ),
            logits=LogitSummary(
                top_k=[TopToken(token_id=100 + pos + 1, text=" next", logit=1.0, prob=0.5)],
                entropy=1.23,
                chosen=(
                    None
                    if pos == n_tokens - 1
                    else TopToken(token_id=100 + pos + 1, text=" next", logit=1.0, prob=0.5)
                ),
            ),
            layers=[
                LayerState(layer=layer, resid_norm=float(np.linalg.norm(residuals[pos, layer])))
                for layer in range(n_layers)
            ],
        )
        for pos in range(n_tokens)
    ]

    trace = Trace(
        trace_id=trace_id,
        model="test-model",
        device="cpu",
        dtype="torch.float32",
        normalization="RMS",
        n_layers=n_layers,
        d_model=d_model,
        prompt="".join(f" t{i}" for i in range(n_prompt)),
        completion="".join(f" t{i}" for i in range(n_prompt, n_tokens)),
        n_prompt_tokens=n_prompt,
        n_generated_tokens=n_tokens - n_prompt,
        steps=steps,
    )
    return CaptureResult(trace=trace, residuals=residuals)
