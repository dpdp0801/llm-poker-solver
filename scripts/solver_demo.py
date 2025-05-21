#!/usr/bin/env python3
"""Interactive demo for TexasSolver with preflop ranges."""

import os
import sys
import json
import subprocess

# add src package
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'llm_poker_solver'))

import preflop


POSITION_FALLBACK = {
    "EP": "UTG",
    "MP": "LJ",
}

def _seat(pos: str) -> str:
    return POSITION_FALLBACK.get(pos.upper(), pos.upper())


def format_board(cards: str) -> str:
    cards = cards.strip()
    if len(cards) != 6:
        raise ValueError('Board cards should be like "QsJh2c"')
    return f"{cards[0:2]},{cards[2:4]},{cards[4:6]}"


def main() -> None:
    lookup = preflop.PreflopLookup()

    action = input("Enter preflop action (e.g. 'UTG raise, BTN call'): ").strip()
    hero_position = input("Enter hero position (default last actor): ").strip() or None
    flop = input("Enter flop cards (e.g. 'QsJh2c'): ").strip()

    ranges = lookup.get_ranges(action, hero_position=hero_position)
    hero_range = ranges.get('hero')
    villain_range = ranges.get('villain')

    print("\n=== Preflop ranges ===")
    print(f"Hero: {hero_range}")
    print(f"Villain: {villain_range}")

    if not hero_range or not villain_range:
        print("Could not determine ranges for both players.")
        return

    hero_combos = ','.join(sorted(preflop.expand_range(hero_range)))
    villain_combos = ','.join(sorted(preflop.expand_range(villain_range)))

    acts = preflop.parse_action_string(action)
    if hero_position is None:
        hero_position = acts[-1][0]
    hero_index = None
    for i in range(len(acts) - 1, -1, -1):
        if acts[i][0].upper() == hero_position.upper():
            hero_index = i
            break
    if hero_index is None:
        hero_index = len(acts) - 1

    if hero_index == len(acts) - 1:
        if hero_index == 0:
            villain_pos = acts[1][0] if len(acts) > 1 else hero_position
        else:
            villain_pos = acts[hero_index - 1][0]
    else:
        villain_pos = acts[-1][0]

    hero_seat = _seat(hero_position)
    villain_seat = _seat(villain_pos)
    hero_ip = preflop.POSTFLOP_ORDER.index(hero_seat) > preflop.POSTFLOP_ORDER.index(villain_seat)

    ip_range = hero_combos if hero_ip else villain_combos
    oop_range = villain_combos if hero_ip else hero_combos

    board_text = format_board(flop)

    commands = [
        'set_pot 100',
        'set_effective_stack 200',
        f'set_board {board_text}',
        f'set_range_ip {ip_range}',
        f'set_range_oop {oop_range}',
        'build_tree',
        'start_solve',
        'dump_result result.json',
    ]

    print("\n=== TexasSolver commands ===")
    for c in commands:
        print(c)

    solver_path = os.path.join('external', 'TexasSolver', 'build', 'console_solver')
    if os.path.exists(solver_path):
        print("\nRunning solver...\n")
        proc = subprocess.run([solver_path], input='\n'.join(commands), text=True, capture_output=True)
        print(proc.stdout)
        if os.path.exists('result.json'):
            with open('result.json') as f:
                data = json.load(f)
            print("\n=== Strategy Output ===")
            print(json.dumps(data, indent=2)[:1000])
    else:
        print(f"Solver binary not found at {solver_path}.")


if __name__ == '__main__':
    main()
