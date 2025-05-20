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

from preflop import PreflopLookup


def run_recommend_tests(lookup: PreflopLookup) -> None:
    """Print recommended actions for specific hands."""

    tests = [
        ("UTG raise, MP call", "JTs", "MP"),
        ("UTG raise, MP call", "2c2d", "MP"),
        ("UTG raise, MP 3bet", "AhKd", "MP"),
        ("UTG raise, MP 3bet", "7h6h", "MP"),
        ("MP raise, CO call", "6c5c", "CO"),
        ("MP raise, CO call", "Ac2c", "CO"),
        ("MP raise, CO 3bet", "AsJd", "CO"),
        ("MP raise, CO 3bet", "9h8h", "CO"),
        ("CO raise, BTN call", "AdJs", "BTN"),
        ("CO raise, BTN call", "5c4c", "BTN"),
        ("CO raise, BTN 3bet", "AhKd", "BTN"),
        ("CO raise, BTN 3bet", "J9s", "BTN"),
        ("BTN raise, SB call", "9d8d", "SB"),
        ("BTN raise, SB call", "Q5s", "SB"),
        ("BTN raise, SB 3bet", "AhKd", "SB"),
        ("BTN raise, SB 3bet", "JTs", "SB"),
        ("CO raise, BTN 3bet", "AhKs", "CO"),
        ("CO raise, BTN 3bet", "6c6d", "CO"),
        ("UTG raise, BTN 3bet", "9c9d", "UTG"),
        ("UTG raise, BTN 3bet", "AsKs", "UTG"),
        ("BTN raise, SB 3bet, BTN 4bet", "AsKs", "BTN"),
        ("UTG raise, BTN 3bet, UTG 4bet, BTN allin", "KcKd", "UTG"),
    ]

    print("=== recommend ===")
    for action, hand, hero in tests:
        rec = lookup.recommend(action, hand, hero_position=hero)
        print(f"Action: {action} | hand={hand} | hero={hero or 'default'} -> {rec}")
    print()


def main() -> None:
    lookup = PreflopLookup()
    run_recommend_tests(lookup)


if __name__ == "__main__":
    main()
