#!/usr/bin/env python3
"""Interactive demo linking preflop ranges with TexasSolver."""

import json
import os
import subprocess
import sys

# Ensure the library is importable when running from the repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "llm_poker_solver"))
from preflop import PreflopLookup, expand_range


def build_commands(hero_combos, villain_combos, board: str):
    board_csv = ",".join(board[i : i + 2] for i in range(0, len(board), 2))
    return [
        "set_pot 100",
        "set_effective_stack 200",
        f"set_board {board_csv}",
        f"set_range_ip {','.join(sorted(villain_combos))}",
        f"set_range_oop {','.join(sorted(hero_combos))}",
        "build_tree",
        "start_solve",
        "dump_result result.json",
    ]


def run_solver(cmds):
    bin_path = os.path.join(
        os.path.dirname(__file__), "..", "external", "TexasSolver", "build", "console_solver"
    )
    if not os.path.exists(bin_path):
        print("TexasSolver binary not found; expected at", bin_path)
        return None
    proc = subprocess.Popen(
        [bin_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    proc.stdin.write("\n".join(cmds) + "\n")
    proc.stdin.flush()
    out, _ = proc.communicate()
    print("\n<<<START SOLVING>>>")
    print(out)
    try:
        with open("result.json") as f:
            return json.load(f)
    except Exception as e:
        print("Could not load result.json:", e)
        return None


def print_root_strategy(data):
    if not data:
        return
    strat = data.get("strategy", {})
    actions = strat.get("actions", [])
    table = {}
    for hand, probs in strat.get("strategy", {}).items():
        for act, p in zip(actions, probs):
            table[act] = table.get(act, 0.0) + p
    if not table:
        return
    print("\n=== Root Strategy Summary ===")
    total = sum(table.values())
    for act in actions:
        prob = table.get(act, 0.0) / total if total else 0.0
        print(f"{act}: {prob:.2f}")


def main():
    action = input("Enter preflop action (e.g. 'UTG raise, BTN call'): ")
    hero_pos = input("Enter hero position (default last actor): ").strip() or None
    flop = input("Enter flop cards (e.g. 'QsJh2c'): ")

    lookup = PreflopLookup()
    ranges = lookup.get_ranges(action, hero_position=hero_pos)
    hero_range = ranges.get("hero", "")
    villain_range = ranges.get("villain", "")

    print("\n=== Preflop ranges ===")
    print("Hero:", hero_range)
    print("Villain:", villain_range)

    hero_combos = expand_range(hero_range) if hero_range else []
    villain_combos = expand_range(villain_range) if villain_range else []

    cmds = build_commands(hero_combos, villain_combos, flop)
    print("\n=== TexasSolver commands ===")
    for c in cmds:
        print(c)

    result = run_solver(cmds)
    print_root_strategy(result)


if __name__ == "__main__":
    main()
