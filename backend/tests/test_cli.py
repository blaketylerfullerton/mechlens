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
