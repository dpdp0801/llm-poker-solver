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
