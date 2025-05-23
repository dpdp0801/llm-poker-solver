import pytest

from llm_poker_solver.preflop import PreflopChart, PreflopLookup, canonize_hand


def test_rfi_lookup():
    lookup = PreflopLookup()
    ranges = lookup.get_ranges("UTG raise")
    assert "66+" in ranges["hero"]


def test_canonize_hand():
    assert canonize_hand("AhKh") == "AKs"
    assert canonize_hand("AdKd") == "AKs"
    assert canonize_hand("AsKd") == "AKo"


def test_expand_plus_non_pair():
    from llm_poker_solver.preflop import expand_range

    hands = expand_range("66+, AQs+")
    assert "AKs" in hands
    assert "AAs" not in hands


def test_get_ranges_with_positions():
    lookup = PreflopLookup()
    res = lookup.get_ranges("CO raise, BTN call", hero_position="BTN")
    assert "33-TT" in res["hero"]
    assert "44+" in res["villain"]

    res2 = lookup.get_ranges("UTG raise, BTN 3bet, UTG call", hero_position="UTG")
    from llm_poker_solver.preflop import expand_range

    villain_range = expand_range(res2["villain"])
    assert "QQ" in villain_range  # BTN 3bet range should include premium pairs
