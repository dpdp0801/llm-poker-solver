#!/usr/bin/env python3
"""Print a series of manual checks for `preflop.py`.

This script exercises the public API of ``PreflopLookup`` along with helper
functions. The goal is to make it easy to visually verify that range lookups
and recommendations behave as expected.
"""

import os
import sys

# Ensure src package is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "llm_poker_solver"))

from preflop import PreflopLookup, canonize_hand, expand_range


# ---------------------------------------------------------------------------
# helper printing functions
# ---------------------------------------------------------------------------


def run_get_ranges_tests(lookup: PreflopLookup) -> None:
    """Print ranges for a variety of action sequences."""

    tests = [
        ("UTG raise", None),
        ("CO raise, BTN call", None),
        ("CO raise, BTN call", "CO"),
        ("CO raise, BTN 3bet", None),
        ("CO raise, BTN 3bet", "CO"),
        ("CO raise, BTN 3bet, CO call", "CO"),
        ("BTN raise, SB 3bet, BTN 4bet", None),
        ("BTN raise, SB 3bet, BTN 4bet", "SB"),
    ]

    print("=== get_ranges ===")
    for action, hero in tests:
        res = lookup.get_ranges(action, hero_position=hero)
        print(f"Action: {action} | hero={hero or 'default'}")
        print(f"  hero range   : {res.get('hero')}")
        print(f"  villain range: {res.get('villain')}")
        print("-" * 60)
    print()


def run_recommend_tests(lookup: PreflopLookup) -> None:
    """Print recommended actions for specific hands."""

    tests = [
        ("CO raise, BTN 3bet", "AhKs", None),
        ("CO raise, BTN 3bet", "AhKs", "CO"),
        ("UTG raise, BTN 3bet", "9c9d", "UTG"),
    ]

    print("=== recommend ===")
    for action, hand, hero in tests:
        rec = lookup.recommend(action, hand, hero_position=hero)
        print(f"Action: {action} | hand={hand} | hero={hero or 'default'} -> {rec}")
    print()


def run_utils_tests() -> None:
    """Show helper function behaviour."""

    print("=== expand_range ===")
    for text in ["AQs+", "KTo+", "TT+"]:
        print(f"{text}: {sorted(expand_range(text))}")

    print("\n=== canonize_hand ===")
    for hand in ["AhKs", "AdKd", "AsKd"]:
        print(f"{hand} -> {canonize_hand(hand)}")
    print()


def main() -> None:
    lookup = PreflopLookup()
    run_get_ranges_tests(lookup)
    run_recommend_tests(lookup)
    run_utils_tests()


if __name__ == "__main__":
    main()
