"""Logit lens tests.

Two halves. The bookkeeping — which layers get filled, what the guards refuse,
what lands on the PassRecord — runs against a stub unembed, so the assertions
can be exact. The half that actually matters runs a real capture through a real
model on CPU and checks the identity the whole pass rests on: the last layer's
lens *is* the model's output, so it must reproduce it position for position.

gemma-2-2b is too heavy for a test run; the identity is model-independent, so
tiny-stories-1M exercises it. What tiny-stories cannot exercise is the softcap
(its cap is 0.0, making apply_softcap an identity), so that is tested directly
against a config with a cap set.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformer_lens.utilities.activation_functions import apply_softcap

from app.capture import RESID_HOOK, generate_trace
from app.passes import apply
from app.passes.lens import LogitLensPass
from app.schema import ResidualRef
from factories import D_MODEL, N_LAYERS, N_TOKENS, make_result

TEST_MODEL = "tiny-stories-1M"
PROMPT = "Once upon a time there was a"

D_VOCAB = 12


# --------------------------------------------------------------------------
# the real thing: a real model, and the identity that makes the lens checkable
# --------------------------------------------------------------------------


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
def lensed(model):
    result = generate_trace(model, PROMPT, max_new_tokens=6, top_k=5)
    result.trace.residuals = ResidualRef(
        path="x.npy",
        hook=RESID_HOOK,
        shape=tuple(result.residuals.shape),
        dtype="float32",
    )
    record = apply(
        LogitLensPass(model=model, verbose=False), result.trace, result.residuals
    )
    return result.trace, record


def test_final_layer_lens_reproduces_the_model_output(lensed):
    """The correctness test of phase 4.

    resid_post of the last block is exactly what forward hands to ln_final, so
    this is not an approximation — it is the model's own output recomputed. If
    the softcap were skipped or the norm reimplemented, this is what breaks.
    """
    trace, _ = lensed
    last = trace.n_layers - 1

    for step in trace.steps:
        lens = step.layers[last].logit_lens
        assert [t.token_id for t in lens.top_k] == [t.token_id for t in step.logits.top_k]
        for got, want in zip(lens.top_k, step.logits.top_k):
            assert got.prob == pytest.approx(want.prob, abs=1e-4)
        assert lens.entropy == pytest.approx(step.logits.entropy, abs=1e-3)


def test_the_agreement_stat_says_so_too(lensed):
    """So a saved trace can be judged without re-running any of this."""
    _, record = lensed
    assert record.stats["final_layer_agreement"] == 1.0
    assert record.stats["final_layer_max_prob_delta"] < 1e-4
    assert record.stats["final_layer_max_entropy_delta"] < 1e-3


def test_an_argmax_tie_counts_as_agreement_and_is_reported():
    """Under bf16 the top two logits sometimes land on the same value, and
    topk orders them however it likes — the capture and the lens index
    differently shaped tensors, so they need not pick the same one.

    That is a coin flip, not a lens bug, and a strict token-id comparison
    reports it as a failure that sends the reader hunting for a softcap
    problem. It has to count as agreement — but be visible, not absorbed.
    """
    result = _traced()
    tied = torch.zeros(N_TOKENS, D_VOCAB)
    tied[:, 2] = tied[:, 9] = 5.0  # exactly equal, so the order is arbitrary

    # Which of the two topk actually picks is itself arbitrary and can differ
    # by torch version, so ask it rather than hardcoding one — then force the
    # capture side to record the *other* one, guaranteeing the mismatch this
    # test exists to check rather than hoping the two calls happen to disagree.
    lens_pick = int(torch.topk(tied[0], 5).indices[0])
    other = 9 if lens_pick == 2 else 2

    for step in result.trace.steps:  # the capture picked the other one
        step.logits.top_k[0].token_id = other
        step.logits.top_k[0].prob = float(torch.softmax(tied[0], dim=-1)[other])
        step.logits.entropy = float(
            -(torch.softmax(tied[0], 0) * torch.log(torch.softmax(tied[0], 0))).sum()
        )

    record = apply(
        LogitLensPass(model=FakeModel(tied), verbose=False), result.trace, result.residuals
    )

    assert result.trace.steps[0].layers[N_LAYERS - 1].logit_lens.top_k[0].token_id == lens_pick
    assert record.stats["final_layer_agreement"] == 1.0
    assert record.stats["final_layer_exact_top1"] == 0.0
    assert record.stats["final_layer_argmax_ties"] == N_TOKENS


def test_a_real_disagreement_is_still_a_failure():
    """The tie rule must not swallow the bug it was written next to: a skipped
    softcap moves the top-1 probability, so the probabilities differ."""
    result = _traced()
    logits = torch.zeros(N_TOKENS, D_VOCAB)
    logits[:, 4] = 9.0  # the lens is confident about token 4...
    for step in result.trace.steps:  # ...and the model said something else
        step.logits.top_k[0].token_id = 1
        step.logits.top_k[0].prob = 0.2

    record = apply(
        LogitLensPass(model=FakeModel(logits), verbose=False), result.trace, result.residuals
    )
    assert record.stats["final_layer_agreement"] == 0.0
    assert record.stats["final_layer_argmax_ties"] == 0.0
    assert record.stats["final_layer_max_prob_delta"] > 0.1


def test_every_layer_and_position_is_filled(lensed, model):
    trace, record = lensed
    for step in trace.steps:
        assert len(step.layers) == model.cfg.n_layers
        for state in step.layers:
            assert state.logit_lens is not None
            assert len(state.logit_lens.top_k) == 5
            probs = [t.prob for t in state.logit_lens.top_k]
            assert probs == sorted(probs, reverse=True)
            assert state.logit_lens.entropy > 0
    assert record.params["n_layers_decoded"] == model.cfg.n_layers


def test_agreement_rises_toward_the_answer_with_depth(lensed):
    """The crystallisation curve, which is the point of the phase.

    Deliberately *not* asserted on entropy. Entropy is not monotonic in depth
    on a real model: early residuals sit near the embedding and decode back to
    the current token with high confidence, so entropy starts low for a reason
    unrelated to the model knowing an answer. See `echo_by_layer`.
    """
    _, record = lensed
    agreement = record.stats["top1_agreement_by_layer"]
    # Not asserted to be exactly 1.0: this curve compares token ids, so an
    # argmax tie costs it a position even where the distributions match.
    # `final_layer_agreement` is the tie-aware number, checked above.
    assert agreement[-1] >= 0.9
    assert agreement[0] < agreement[-1]
    assert 0 <= record.stats["crossover_layer"] < len(agreement)


def test_the_echo_curve_is_recorded(lensed):
    """Without it, a reader takes an early layer's 85%-confident reading of the
    token already in front of it for a prediction."""
    _, record = lensed
    echo = record.stats["echo_by_layer"]
    assert len(echo) == len(record.stats["top1_agreement_by_layer"])
    assert all(0.0 <= e <= 1.0 for e in echo)


# --------------------------------------------------------------------------
# the softcap — the one thing tiny-stories cannot exercise
# --------------------------------------------------------------------------


def test_softcap_is_applied_when_the_model_has_one():
    """Gemma caps logits at 30 *after* the unembed, and TransformerLens does it
    in forward rather than inside Unembed — so the pass has to do it itself.

    Skipping it leaves top-1 roughly right and every probability wrong, which
    is why this is asserted rather than eyeballed: the assertion below is on
    the probabilities, and it is checked that the two distributions actually
    differ, so the test cannot pass by the cap being a no-op.
    """
    cap = 3.0
    raw = _fake_logits() * 8.0  # wide enough that tanh bites
    capped = apply_softcap(raw, cap)
    assert not torch.allclose(raw, capped)

    result = _traced()
    apply(
        LogitLensPass(model=FakeModel(raw, soft_cap=cap), layers=[0], verbose=False),
        result.trace,
        result.residuals,
    )

    lens = result.trace.steps[1].layers[0].logit_lens
    want = torch.softmax(capped[1], dim=-1)
    uncapped = torch.softmax(raw[1], dim=-1)

    assert lens.top_k[0].prob == pytest.approx(float(want.max()), abs=1e-5)
    # and the uncapped distribution is a materially different answer, so the
    # assertion above is actually discriminating
    assert abs(float(uncapped.max()) - float(want.max())) > 1e-3


def test_softcap_is_recorded_even_when_the_model_has_none():
    """'Was the cap applied?' is the first question to ask of a lens whose
    probabilities look wrong, so the trace has to be able to answer it."""
    result = _traced()
    record = apply(
        LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
        result.trace,
        result.residuals,
    )
    assert record.params["output_logits_soft_cap"] == 0.0


# --------------------------------------------------------------------------
# bookkeeping, against a stub
# --------------------------------------------------------------------------


class FakeModel:
    """Just enough HookedTransformer for the pass: a norm, an unembed, a cfg.

    `logits[pos]` is dictated by the test, so what lands in the trace can be
    asserted exactly rather than approximately.
    """

    def __init__(self, logits: torch.Tensor, soft_cap: float = 0.0):
        self._logits = logits
        self.cfg = type(
            "cfg",
            (),
            {
                "n_layers": N_LAYERS,
                "d_model": D_MODEL,
                "device": "cpu",
                "model_name": "test-model",
                "output_logits_soft_cap": soft_cap,
            },
        )()
        self.tokenizer = type("tok", (), {"decode": staticmethod(lambda ids: f"<{ids[0]}>")})()

    def ln_final(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits[: x.shape[0]]


def _traced(**ref_kwargs):
    result = make_result()
    defaults = dict(
        path="x.npy", hook=RESID_HOOK, shape=(N_TOKENS, N_LAYERS, D_MODEL), dtype="float32"
    )
    result.trace.residuals = ResidualRef(**{**defaults, **ref_kwargs})
    return result


def _fake_logits(seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(
        np.random.default_rng(seed).standard_normal((N_TOKENS, D_VOCAB), dtype=np.float32)
    )


def test_only_the_requested_layers_are_decoded():
    result = _traced()
    apply(
        LogitLensPass(model=FakeModel(_fake_logits()), layers=[1], verbose=False),
        result.trace,
        result.residuals,
    )
    step = result.trace.steps[0]
    assert step.layers[1].logit_lens is not None
    assert step.layers[0].logit_lens is None
    assert step.layers[2].logit_lens is None


def test_partial_layers_skip_the_correctness_check_rather_than_faking_it():
    """A check that quietly ran against layer 1 would report a pass that means
    nothing — the identity only holds at the last layer."""
    result = _traced()
    record = apply(
        LogitLensPass(model=FakeModel(_fake_logits()), layers=[0, 1], verbose=False),
        result.trace,
        result.residuals,
    )
    assert "final_layer_agreement" not in record.stats
    assert record.stats["top1_agreement_by_layer"]  # the curve is still recorded


def test_crossover_is_the_first_layer_that_agrees_with_the_answer():
    """The 'done when' of phase 4, as a number: the depth where the answer sets."""
    result = _traced()
    answer = 3
    for step in result.trace.steps:
        step.logits.top_k[0].token_id = answer

    early = torch.zeros(N_TOKENS, D_VOCAB)
    early[:, 7] = 5.0  # a confident wrong guess
    late = torch.zeros(N_TOKENS, D_VOCAB)
    late[:, answer] = 5.0

    class Staged(FakeModel):
        """Layers 0 and 1 guess wrong; layer 2 lands on the answer."""

        def __init__(self):
            super().__init__(early)
            self.decoded = 0

        def unembed(self, x):
            self.decoded += 1
            return late if self.decoded == N_LAYERS else early

    record = apply(LogitLensPass(model=Staged(), verbose=False), result.trace, result.residuals)

    assert record.stats["top1_agreement_by_layer"] == [0.0, 0.0, 1.0]
    assert record.stats["crossover_layer"] == 2.0
    assert record.stats["final_layer_agreement"] == 1.0


def test_crossover_is_negative_when_the_answer_never_forms():
    result = _traced()
    for step in result.trace.steps:
        step.logits.top_k[0].token_id = 3

    logits = torch.zeros(N_TOKENS, D_VOCAB)
    logits[:, 7] = 5.0
    record = apply(
        LogitLensPass(model=FakeModel(logits), verbose=False), result.trace, result.residuals
    )
    assert record.stats["crossover_layer"] == -1.0
    assert record.stats["final_layer_agreement"] == 0.0


def test_rerunning_the_pass_replaces_its_record():
    result = _traced()
    apply(LogitLensPass(model=FakeModel(_fake_logits()), top_k=2, verbose=False),
          result.trace, result.residuals)
    apply(LogitLensPass(model=FakeModel(_fake_logits()), top_k=4, verbose=False),
          result.trace, result.residuals)

    assert [p.name for p in result.trace.passes] == ["lens"]
    assert result.trace.pass_record("lens").params["top_k"] == 4


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------


def test_wrong_hook_site_is_refused():
    """resid_pre would still decode to something readable — and the last
    layer's identity, the only thing that proves the lens right, would be gone."""
    result = _traced(hook="hook_resid_pre")
    with pytest.raises(ValueError, match="needs hook_resid_post"):
        apply(LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
              result.trace, result.residuals)


def test_trace_without_residuals_is_refused():
    result = make_result()
    with pytest.raises(ValueError, match="no residuals"):
        apply(LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
              result.trace, result.residuals)


def test_a_different_model_is_refused():
    result = _traced()
    result.trace.model = "gemma-2-2b"
    with pytest.raises(ValueError, match="different unembed"):
        apply(LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
              result.trace, result.residuals)


def test_mismatched_shape_is_refused():
    result = _traced()
    result.trace.d_model = D_MODEL + 1
    with pytest.raises(ValueError, match=f"{N_LAYERS}x{D_MODEL + 1}"):
        apply(LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
              result.trace, result.residuals)


def test_folded_layernorm_is_accepted_unlike_the_sae_pass():
    """Folding is RMSNormPre(x) @ W_U_folded == RMSNorm(x) @ W_U — the same
    arithmetic. The SAE pass must refuse it; the lens has no reason to."""
    result = _traced()
    result.trace.normalization = "RMSPre"
    apply(LogitLensPass(model=FakeModel(_fake_logits()), verbose=False),
          result.trace, result.residuals)
    assert result.trace.steps[0].layers[0].logit_lens is not None
