"""Smoke tests for Engine, run against a tiny random Llama checkpoint.

This isn't a real language model (weights are random, so outputs are
gibberish) — it's the Llama architecture at ~4 layers / tiny hidden size,
so it exercises the exact code paths (model.model.layers, self_attn.output,
lm_head, o_proj.input) that a real Llama/Gemma checkpoint would, in a couple
seconds on CPU with a ~few-MB download.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import Engine  # noqa: E402
from app.lenses.ablate import ablate_head  # noqa: E402
from app.lenses.generate import generate  # noqa: E402
from app.lenses.jacobian import JacobianLensResult, jacobian_lens  # noqa: E402
from app.lenses.trace import TraceResult, trace  # noqa: E402

TINY_MODEL = "trl-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def engine():
    return Engine(TINY_MODEL, device_map="cpu", dtype=torch.float32)


def test_load(engine):
    assert engine.n_layers > 0
    assert engine.tokenizer is not None


def test_generate(engine):
    out = generate(engine, "Hello there", max_new_tokens=5)
    assert isinstance(out, str)
    assert out.startswith("Hello there")


def test_trace_shapes(engine):
    result = trace(engine, "Hello there", top_k=3)
    assert isinstance(result, TraceResult)

    n_layers = engine.n_layers
    assert len(result.hidden_states) == n_layers
    assert len(result.attention) == n_layers
    assert len(result.logit_lens) == n_layers

    # hidden_states[layer] is [token][dim]
    n_tokens = len(result.hidden_states[0])
    assert n_tokens > 0
    hidden_size = engine.model.config.hidden_size
    assert len(result.hidden_states[0][0]) == hidden_size

    # attention[layer] is [head][query][key] (no batch dim)
    assert result.attention[0] != [], "expected attention weights with attn_implementation='eager'"
    n_heads = engine.model.config.num_attention_heads
    assert len(result.attention[0]) == n_heads
    assert len(result.attention[0][0]) == n_tokens  # query positions
    assert len(result.attention[0][0][0]) == n_tokens  # key positions

    # logit_lens[layer] is top_k entries of {token, prob}
    assert len(result.logit_lens[0]) == 3
    assert set(result.logit_lens[0][0].keys()) == {"token", "prob"}

    assert isinstance(result.predicted_next_token, str)


def test_jacobian_lens_shapes(engine):
    result = jacobian_lens(engine, "Hello there", top_k=3)
    assert isinstance(result, JacobianLensResult)

    assert len(result.layers) == engine.n_layers
    assert isinstance(result.target_token_id, int)
    assert isinstance(result.target_token, str)

    for layer in result.layers:
        assert isinstance(layer["grad_norm"], float)
        assert layer["grad_norm"] >= 0
        assert len(layer["top_aligned_tokens"]) == 3
        assert set(layer["top_aligned_tokens"][0].keys()) == {"token", "score"}


def test_jacobian_lens_explicit_target(engine):
    # Find any low-id token this tiny tokenizer round-trips to a single id;
    # this just checks the explicit-target path resolves to that id, rather
    # than falling back to argmax.
    tokenizer = engine.tokenizer
    token_str = ids = None
    for candidate_id in range(10):
        candidate_str = tokenizer.decode([candidate_id])
        candidate_ids = tokenizer.encode(candidate_str, add_special_tokens=False)
        if len(candidate_ids) == 1:
            token_str, ids = candidate_str, candidate_ids
            break
    if token_str is None:
        pytest.skip("no low-id token round-trips to a single token for this checkpoint")

    result = jacobian_lens(engine, "Hello there", target_token=token_str, top_k=2)
    assert result.target_token_id == ids[0]


def test_ablate_head_runs(engine):
    out = ablate_head(engine, "Hello there", layer_idx=0, head_idx=0, max_new_tokens=5)
    assert isinstance(out, str)
    assert out.startswith("Hello there")
