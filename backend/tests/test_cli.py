import pytest

from app.cli import parse_layers


def test_layer_spec_accepts_ranges_lists_and_both():
    assert parse_layers("0-3", 26) == [0, 1, 2, 3]
    assert parse_layers("5,20,25", 26) == [5, 20, 25]
    assert parse_layers("0-2,20,25", 26) == [0, 1, 2, 20, 25]


def test_layer_spec_is_sorted_and_deduplicated():
    assert parse_layers("20,0-2,20,1", 26) == [0, 1, 2, 20]


def test_no_spec_means_every_layer():
    assert parse_layers(None, 26) is None
    assert parse_layers("", 26) is None


def test_out_of_range_layers_are_rejected():
    """Silently encoding 25 layers when you asked for 30 would be worse."""
    with pytest.raises(SystemExit, match=r"\[26, 30\]"):
        parse_layers("0,26,30", 26)


def test_summary_defaults_to_a_layer_that_was_actually_encoded(capsys):
    """`enrich --layers 12` leaves every other layer featureless.

    Defaulting the summary to n_layers//2 then reports an empty `l0=None`
    state, which reads as "the pass did nothing" on a pass that worked fine.
    """
    from app.cli import print_sae_summary
    from app.schema import Feature, PassRecord

    from factories import make_result

    trace = make_result().trace  # 4 tokens x 3 layers
    encoded = 2  # not n_layers // 2
    for step in trace.steps:
        step.layers[encoded].features = [Feature(index=7, activation=1.5)]
        step.layers[encoded].l0 = 1
    trace.passes = [
        PassRecord(
            name="sae",
            params={"release": "test", "width": "16k"},
            stats={"l0_mean": 80.0, "explained_variance_mean": 0.9},
        )
    ]

    print_sae_summary(trace)
    out = capsys.readouterr().out
    assert f"layer {encoded}," in out
    assert "l0=None" not in out
