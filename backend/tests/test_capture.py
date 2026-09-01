"""Capture-loop tests against a 1M-param model on CPU (a couple of seconds).

gemma-2-2b is too heavy to test against on every run; the loop's logic —
bookkeeping, alignment, that the incremental capture equals a single full pass —
is model-independent, so a tiny model exercises all of it.
"""

import numpy as np
import pytest
import torch

from capture import RESID_HOOK, generate_trace

TEST_MODEL = "tiny-stories-1M"
PROMPT = "Once upon a time there was a"


@pytest.fixture(scope="module")
def model():
    from transformer_lens import HookedTransformer

    try:
        m = HookedTransformer.from_pretrained(TEST_MODEL, device="cpu")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"{TEST_MODEL} unavailable: {exc}")
    m.eval()
    return m


@pytest.fixture(scope="module")
def result(model):
    return generate_trace(model, PROMPT, max_new_tokens=6, top_k=5)


def test_residual_tensor_covers_every_token_and_layer(result, model):
    trace = result.trace
    n_tokens = trace.n_prompt_tokens + trace.n_generated_tokens

    assert result.residuals.shape == (n_tokens, model.cfg.n_layers, model.cfg.d_model)
    assert result.residuals.dtype == np.float32
    assert result.hook == RESID_HOOK
    # the ref is attached on save, when there is finally a file to point at
    assert trace.residuals is None
    assert len(trace.steps) == n_tokens
    assert all(len(step.layers) == model.cfg.n_layers for step in trace.steps)
    assert np.isfinite(result.residuals).all()


def test_prompt_positions_are_captured_too(result):
    """Attribution mostly points back at the prompt, so it cannot be skipped."""
    trace = result.trace
    sources = [step.token.source for step in trace.steps]

    assert sources.count("prompt") == trace.n_prompt_tokens > 1
    assert sources.count("generated") == trace.n_generated_tokens == 6
    # prompt first, generation after — never interleaved
    assert sources == ["prompt"] * trace.n_prompt_tokens + ["generated"] * 6


def test_incremental_capture_equals_a_single_full_pass(result, model):
    """Each step re-runs the prefix and banks only the new position; causal
    attention says that must match running the finished sequence once."""
    ids = [step.token.token_id for step in result.trace.steps]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            torch.tensor([ids]), names_filter=lambda n: n.endswith(RESID_HOOK)
        )
    reference = torch.stack(
        [cache[f"blocks.{i}.{RESID_HOOK}"][0] for i in range(model.cfg.n_layers)], dim=1
    ).numpy()

    np.testing.assert_allclose(result.residuals, reference, rtol=1e-4, atol=1e-4)


def test_resid_norms_in_the_json_match_the_tensor(result):
    from_json = np.array([[l.resid_norm for l in s.layers] for s in result.trace.steps])
    np.testing.assert_allclose(
        from_json, np.linalg.norm(result.residuals, axis=-1), rtol=1e-5, atol=1e-3
    )


def test_each_position_predicts_the_token_that_follows_it(result):
    steps = result.trace.steps
    for step, nxt in zip(steps, steps[1:]):
        assert step.logits.chosen is not None
        assert step.logits.chosen.token_id == nxt.token.token_id
    # nothing was appended after the last position, so nothing was chosen there
    assert steps[-1].logits.chosen is None


def test_generated_tokens_are_the_greedy_argmax(result):
    trace = result.trace
    for step in trace.steps[trace.n_prompt_tokens - 1 : -1]:
        assert step.logits.chosen.token_id == step.logits.top_k[0].token_id


def test_logit_summaries_are_well_formed(result):
    for step in result.trace.steps:
        probs = [t.prob for t in step.logits.top_k]
        assert len(probs) == 5
        assert probs == sorted(probs, reverse=True)
        assert 0.0 < probs[0] <= 1.0
        assert 0.0 <= step.logits.entropy <= np.log(len(step.logits.top_k)) + 20


def test_completion_is_the_generated_text_only(result, model):
    trace = result.trace
    generated_ids = [s.token.token_id for s in trace.steps[trace.n_prompt_tokens :]]
    assert trace.completion == model.to_string(torch.tensor(generated_ids))
    assert trace.prompt == PROMPT
    # cfg.model_name is the resolved name, not the alias passed in
    # ("TinyStories-1M" for "tiny-stories-1M") — record what was actually loaded
    assert trace.model == model.cfg.model_name


def test_generation_is_deterministic(model, result):
    again = generate_trace(model, PROMPT, max_new_tokens=6, top_k=5)
    assert [s.token.token_id for s in again.trace.steps] == [
        s.token.token_id for s in result.trace.steps
    ]
    np.testing.assert_array_equal(again.residuals, result.residuals)
    assert again.trace.trace_id != result.trace.trace_id  # ids stay unique


def test_zero_new_tokens_still_traces_the_prompt(model):
    """The n=0 case is the one the loop is easiest to get wrong."""
    out = generate_trace(model, PROMPT, max_new_tokens=0)
    trace = out.trace

    assert trace.n_generated_tokens == 0
    assert trace.completion == ""
    assert len(trace.steps) == trace.n_prompt_tokens
    assert out.residuals.shape[0] == trace.n_prompt_tokens
    # the prompt's final position still carries a prediction, just an unused one
    assert trace.steps[-1].logits.chosen is None
    assert trace.steps[-1].logits.top_k
