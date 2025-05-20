#!/usr/bin/env python3
"""Show recommendations for a variety of preflop scenarios."""

import os
import sys

# Ensure src package is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "llm_poker_solver"))

from preflop import PreflopLookup


def main() -> None:
    lookup = PreflopLookup()

    tests = [
        ("UTG raise", "AhKs", "UTG"),
        ("UTG raise", "7c7d", "UTG"),
        ("CO raise, BTN 3bet", "AhKs", None),
        ("CO raise, BTN 3bet", "9c9d", None),
        ("CO raise, BTN 3bet", "AhKs", "CO"),
        ("CO raise, BTN 3bet", "9c9d", "CO"),
        ("UTG raise, CO 3bet", "AhKs", "UTG"),
        ("UTG raise, CO 3bet", "9c9d", "UTG"),
        ("UTG raise, CO 3bet", "AhKs", "CO"),
        ("UTG raise, CO 3bet", "9c9d", "CO"),
        ("BTN raise, SB 3bet", "AhKs", "BTN"),
        ("BTN raise, SB 3bet", "9c9d", "BTN"),
        ("BTN raise, SB 3bet", "AhKs", "SB"),
        ("BTN raise, SB 3bet", "9c9d", "SB"),
        ("SB raise, BB 3bet", "AhKs", "SB"),
        ("SB raise, BB 3bet", "9c9d", "SB"),
        ("SB raise, BB 3bet", "AhKs", "BB"),
        ("SB raise, BB 3bet", "9c9d", "BB"),
        ("CO raise, BTN 3bet, CO 4bet", "AhKs", "CO"),
        ("CO raise, BTN 3bet, CO 4bet", "9c9d", "CO"),
        ("CO raise, BTN 3bet, CO 4bet", "AhKs", "BTN"),
        ("UTG raise, CO 3bet, UTG 4bet", "AhKs", "UTG"),
        ("UTG raise, CO 3bet, UTG 4bet", "9c9d", "UTG"),
        ("UTG raise, CO 3bet, UTG 4bet", "AhKs", "CO"),
        ("CO raise, BTN 3bet, CO 4bet, BTN allin", "AhKs", "CO"),
        ("CO raise, BTN 3bet, CO 4bet, BTN allin", "9c9d", "CO"),
        ("UTG raise, CO 3bet, UTG 4bet, CO allin", "AhKs", "UTG"),
        ("UTG raise, CO 3bet, UTG 4bet, CO allin", "9c9d", "UTG"),
        ("MP raise, CO call", "AhKs", "CO"),
        ("CO raise, SB call", "7c6c", "SB"),
    ]

    print("=== recommend ===")
    for action, hand, hero in tests:
        rec = lookup.recommend(action, hand, hero_position=hero)
        print(f"Action: {action} | hand={hand} | hero={hero or 'default'} -> {rec}")


if __name__ == "__main__":
    main()
