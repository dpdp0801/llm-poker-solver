#!/usr/bin/env python3
"""Advanced interactive demo for TexasSolver with improved game tree navigation.

This script provides a more sophisticated interface to the TexasSolver,
allowing for analysis of specific decision points in the game tree and
getting optimal strategies for specific hands.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add src package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'llm_poker_solver'))

from preflop import (
    PreflopLookup,
    expand_range,
    parse_action_string,
    compute_pot_and_effective_stack,
)
from texas_solver import TexasSolverBridge, SolverConfig
from utils import (
    analyze_board_texture, suggest_bet_sizes, format_hand_for_display,
    generate_realistic_strategy, calculate_hand_strength
)


def print_section(title: str) -> None:
    """Print a section title with formatting."""
    print(f"\n{'=' * 5} {title} {'=' * 5}")


def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with a default value."""
    result = input(f"{prompt} [{default}]: ").strip()
    return result if result else default


def format_action_frequencies(frequencies: Dict[str, float], threshold: float = 0.01) -> str:
    """Format action frequencies as a string."""
    return " | ".join(
        f"{action}: {freq:.1%}" 
        for action, freq in sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
        if freq >= threshold
    )


def main() -> None:
    """Run the advanced solver demo."""
    print_section("Advanced Poker Solver Analysis")
    
    # Initialize the preflop lookup
    lookup = PreflopLookup()
    
    # Get preflop action and positions
    action = get_user_input("Enter preflop action (e.g. 'UTG raise, BTN call')", "CO raise, BTN call")
    hero_position = get_user_input("Enter hero position (default last actor)", "")
    
    # Get ranges from preflop lookup
    try:
        ranges = lookup.get_ranges(action, hero_position=hero_position)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    
    hero_range = ranges.get('hero')
    villain_range = ranges.get('villain')
    
    print_section("Preflop Ranges")
    print(f"Hero: {hero_range}")
    print(f"Villain: {villain_range}")
    
    if not hero_range or not villain_range:
        print("Could not determine ranges for both players.")
        return
    
    # Get flop cards
    flop = get_user_input("Enter flop cards (e.g. 'Ah7d2c')", "Ah7d2c")
    if len(flop) < 6 or len(flop) % 2 != 0:
        print("Invalid flop format. Please use format like 'Ah7d2c'")
        return
    
    # Analyze board texture using utility function
    texture = analyze_board_texture(flop)
    print_section("Board Texture Analysis")
    print(f"Texture: {texture['texture']}")
    for key, value in texture.items():
        if key != "texture":
            print(f"{key}: {value}")
    
    # Suggest bet sizes based on board texture using utility function
    bet_sizes = suggest_bet_sizes(texture)
    print_section("Recommended Bet Sizes")
    for pos, sizes in bet_sizes.items():
        print(f"{pos}: {sizes}")
    
    # Ask user if they want to use these bet sizes
    use_suggested = get_user_input("Use suggested bet sizes? (y/n)", "y").lower() == "y"
    
    if not use_suggested:
        default_sizes = "25,33,50,66,75,100,150"
        ip_flop_sizes = get_user_input(
            "Enter IP flop bet sizes (comma-separated percentages)", default_sizes
        )
        oop_flop_sizes = get_user_input(
            "Enter OOP flop bet sizes (comma-separated percentages)", default_sizes
        )
        bet_sizes = {
            "ip_flop": [float(x.strip()) for x in ip_flop_sizes.split(",")],
            "oop_flop": [float(x.strip()) for x in oop_flop_sizes.split(",")]
        }
    
    # Prepare solver configuration
    hero_combos = ','.join(sorted(expand_range(hero_range)))
    villain_combos = ','.join(sorted(expand_range(villain_range)))
    pot, eff_stack = compute_pot_and_effective_stack(action)
    
    # Determine IP and OOP positions and ranges
    acts = parse_action_string(action)
    if hero_position is None:
        hero_position = acts[-1][0]
    hero_position = hero_position.upper()
    
    # Find the hero index
    hero_index = None
    for i in range(len(acts) - 1, -1, -1):
        if acts[i][0].upper() == hero_position:
            hero_index = i
            break
    
    if hero_index is None:
        hero_index = len(acts) - 1
    
    # Find the villain position
    if hero_index == len(acts) - 1:
        if hero_index == 0:
            villain_pos = acts[1][0] if len(acts) > 1 else hero_position
        else:
            villain_pos = acts[hero_index - 1][0]
    else:
        villain_pos = acts[-1][0]
    
    # Map positions to postflop order
    positions = {"UTG": 2, "UTG+1": 3, "LJ": 4, "HJ": 5, "CO": 6, "BTN": 7, "SB": 0, "BB": 1}
    hero_pos_index = positions.get(hero_position.upper(), 0)
    villain_pos_index = positions.get(villain_pos.upper(), 0)
    
    # Determine who is in position postflop
    hero_is_ip = hero_pos_index > villain_pos_index
    
    range_ip = hero_combos if hero_is_ip else villain_combos
    range_oop = villain_combos if hero_is_ip else hero_combos
    
    print_section("Position Info")
    print(f"Hero position: {hero_position} ({'IP' if hero_is_ip else 'OOP'} postflop)")
    print(f"Villain position: {villain_pos} ({'IP' if not hero_is_ip else 'OOP'} postflop)")
    
    # Show formatted board
    print_section("Board")
    formatted_board = " ".join(format_hand_for_display(flop[i:i+2]) for i in range(0, len(flop), 2))
    print(formatted_board)
    
    # Configure the solver
    config = SolverConfig(
        iterations=5000,
        accuracy=0.0001,
        board=flop,
        pot_sizes=[pot],
        effective_stack=eff_stack,
        range_ip=range_ip,
        range_oop=range_oop,
        bet_sizes={
            "ip_flop": bet_sizes["ip_flop"],
            "oop_flop": bet_sizes["oop_flop"]
        },
    )
    
    # Create solver bridge
    solver = TexasSolverBridge()
    solver.set_config(config)
    
    # Check if the solver binary exists
    if not os.path.exists(solver.solver_path):
        print(f"Solver binary not found at {solver.solver_path}")
        print("Please build the solver first:")
        print("cd external/TexasSolver && mkdir -p build && cd build && cmake .. && make")
        return
    
    # Ask user if they want to run the solver
    run_solver = get_user_input("Run the solver? (y/n)", "y").lower() == "y"
    
    if not run_solver:
        print("Exiting without running solver.")
        return
    
    # Run the solver
    print_section("Running Solver")
    print("This may take a few minutes depending on the complexity...")
    print("The solver will display its progress below. If you don't see any output,")
    print("the solver binary might not be working properly.")
    print("You should see solver commands and iterations progress.")
    
    try:
        result = solver.run_solver("solver_results.json")
        print("\nSolver completed successfully!")
        print(f"Check solver_results.json for full results")
    except Exception as e:
        print(f"\nError running solver: {e}")
        return
    
    # Analyze the results
    print_section("Solver Results")
    
    # Get EV
    ev = solver.get_ev()
    print(f"Expected Value: {ev:.3f} big blinds")
    
    # Analyze strategies
    ip_strategy = solver.analyze_strategy("ip", "flop")
    oop_strategy = solver.analyze_strategy("oop", "flop")
    
    print_section("IP Strategy")
    if ip_strategy:
        print(format_action_frequencies(ip_strategy))
    else:
        print("No IP strategy found in solver results")
    
    print_section("OOP Strategy")
    if oop_strategy:
        print(format_action_frequencies(oop_strategy))
    else:
        print("No OOP strategy found in solver results")
    
    # Ask if user wants to navigate the game tree
    navigate_tree = get_user_input("Navigate game tree? (y/n)", "n").lower() == "y"
    
    if navigate_tree:
        current_node = "flop"
        while True:
            print_section(f"Current Node: {current_node}")
            
            # Show IP strategy at this node
            ip_strat = solver.analyze_strategy("ip", current_node)
            if ip_strat:
                print(f"IP Strategy: {format_action_frequencies(ip_strat)}")
            else:
                print("IP: No strategy at this node")
                
            # Show OOP strategy at this node
            oop_strat = solver.analyze_strategy("oop", current_node)
            if oop_strat:
                print(f"OOP Strategy: {format_action_frequencies(oop_strat)}")
            else:
                print("OOP: No strategy at this node")
                
            # Get next action
            next_action = get_user_input("Enter action to navigate (e.g. 'check', 'bet', or 'back' to go up)", "")
            if not next_action:
                break
                
            if next_action.lower() == "back":
                # Go up one level
                if "." in current_node:
                    current_node = ".".join(current_node.split(".")[:-1])
                continue
                
            # Navigate to next node
            next_node = current_node + "." + next_action if current_node else next_action
            
            # Check if there's anything at that node
            if solver.analyze_strategy("ip", next_node) or solver.analyze_strategy("oop", next_node):
                current_node = next_node
            else:
                print(f"No strategy found at node: {next_node}")
    
    # Ask if user wants to analyze specific hands
    analyze_hands = get_user_input("Analyze specific hands? (y/n)", "y").lower() == "y"
    
    if analyze_hands:
        while True:
            hand = get_user_input("Enter hand to analyze (e.g. 'AcKd') or q to quit", "")
            if not hand or hand.lower() == 'q':
                break
                
            pos = get_user_input("Position (ip/oop)", "ip")
            node = get_user_input("Node (default: flop)", "flop")
            
            try:
                # Get solver's strategy
                hand_strategy = solver.get_optimal_play(pos, node, hand)
                
                # Calculate realistic strategy based on hand strength
                hand_strength = calculate_hand_strength(hand, flop)
                is_ip = pos.lower() == "ip"
                realistic_strategy = generate_realistic_strategy(hand_strength, texture, is_ip)
                
                # Display hand information
                print_section(f"Analysis for {format_hand_for_display(hand)}")
                print(f"Position: {pos.upper()}")
                print(f"Hand strength: {hand_strength:.2f} (0=weak, 1=strong)")
                
                # Display solver strategy
                print("\nSolver Strategy:")
                if hand_strategy:
                    for action, prob in sorted(hand_strategy.items(), key=lambda x: (-x[1], x[0])):
                        if prob > 0.01:
                            print(f"  {action}: {prob:.1%}")
                else:
                    print("  No specific strategy found in solver results")
                
                # Display realistic strategy
                print("\nRealistic GTO Strategy:")
                for action, prob in sorted(realistic_strategy.items(), key=lambda x: (-x[1], x[0])):
                    if prob > 0.01:
                        print(f"  {action}: {prob:.1%}")
                
                # Give a recommendation
                print("\nRecommendation:")
                if "BET" in realistic_strategy and realistic_strategy["BET"] > 0.7:
                    print("  Strong betting spot")
                elif "BET" in realistic_strategy and realistic_strategy["BET"] > 0.3:
                    print("  Mixed strategy spot - occasionally bet for balance")
                else:
                    print("  Mostly checking spot")
                
            except Exception as e:
                print(f"Error analyzing hand: {e}")
    
    print_section("Analysis Complete")


if __name__ == "__main__":
    main() 