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
        # Simple RFI scenarios with various hand strengths
        ("UTG raise", "AhAd", None),
        ("UTG raise", "7c2d", None),
        ("CO raise", "QhTd", None),  # Changed from "QTo" to use specific cards
        ("CO raise", "9c2c", None),
        ("BTN raise", "KhJd", None),  # Changed from "KJo" to use specific cards
        ("BTN raise", "8c2c", None),
        ("SB raise", "Ah4d", None),  # Changed from "A4o" to use specific cards
        ("SB raise", "Qh2d", None),  # Changed from "Q2o" to use specific cards
        
        # Facing a raise - call
        ("UTG raise, MP call", "JhTs", "MP"),  # Changed from "JTs" to use specific cards
        ("UTG raise, MP call", "2c2d", "MP"),
        
        # Facing a raise - 3bet
        ("UTG raise, MP 3bet", "AhKd", "MP"),
        ("UTG raise, MP 3bet", "7h6h", "MP"),
        
        # Flatting scenarios
        ("MP raise, CO call", "6c5c", "CO"),
        ("MP raise, CO call", "Ac2c", "CO"),
        
        # 3betting scenarios
        ("MP raise, CO 3bet", "AsJd", "CO"),
        ("MP raise, CO 3bet", "9h8h", "CO"),
        
        # BTN vs CO scenarios
        ("CO raise, BTN call", "AdJs", "BTN"),
        ("CO raise, BTN call", "5c4c", "BTN"),
        ("CO raise, BTN 3bet", "AhKd", "BTN"),
        ("CO raise, BTN 3bet", "Jh9s", "BTN"),  # Changed from "J9s" to use specific cards
        
        # SB defending scenarios
        ("BTN raise, SB call", "9d8d", "SB"),
        ("BTN raise, SB call", "Qh5s", "SB"),  # Changed from "Q5s" to use specific cards
        ("BTN raise, SB 3bet", "AhKd", "SB"),
        ("BTN raise, SB 3bet", "JhTs", "SB"),  # Changed from "JTs" to use specific cards
        
        # Facing 3bets
        ("CO raise, BTN 3bet", "AhKs", "CO"),
        ("CO raise, BTN 3bet", "6c6d", "CO"),
        ("UTG raise, BTN 3bet", "9c9d", "UTG"),
        ("UTG raise, BTN 3bet", "AsKs", "UTG"),
        
        # 4bet and allin scenarios
        ("BTN raise, SB 3bet, BTN 4bet", "AsKs", "BTN"),
        ("UTG raise, BTN 3bet, UTG 4bet, BTN allin", "KcKd", "UTG"),
    ]

    print("=== recommend ===")
    for action, hand, hero in tests:
        try:
            rec = lookup.recommend(action, hand, hero_position=hero)
            print(f"Action: {action} | hand={hand} | hero={hero or 'default'} -> {rec}")
        except ValueError as e:
            print(f"Error: {action} | hand={hand} | hero={hero or 'default'} -> {e}")
    print()


def main() -> None:
    lookup = PreflopLookup()
    run_recommend_tests(lookup)


if __name__ == "__main__":
    main()
