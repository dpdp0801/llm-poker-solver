#!/usr/bin/env python3
"""
Solver Input Generator

Generates random preflop scenarios, calculates pot sizes, generates flops,
analyzes textures, and creates appropriate filenames for solver input.
"""

import os
import sys
import random
from typing import List, Tuple, Dict, Any

# Add src package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'llm_poker_solver'))

from preflop_scenario_generator import (
    generate_preflop_scenario,
    get_ranges_with_frequencies,
    parse_scenario,
    determine_ip_oop,
    PREFLOP_POSITIONS,
    POSITION_GROUPS
)

# Starting stack size
STARTING_STACK = 100  # bb

# Blind sizes
SB_SIZE = 0.5  # bb
BB_SIZE = 1.0  # bb

# Raise sizes
INITIAL_RAISE_SIZE = 2.5  # bb
IP_3BET_MULTIPLIER = 3  # 3x the original raise
OOP_3BET_SIZE = 11  # bb
IP_4BET_SIZE = 25  # bb  
OOP_4BET_SIZE = 22  # bb

# Card definitions
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}


def generate_deck() -> List[str]:
    """Generate a standard 52-card deck."""
    return [rank + suit for rank in RANKS for suit in SUITS]


def generate_flop() -> List[str]:
    """Generate 3 random flop cards."""
    deck = generate_deck()
    return random.sample(deck, 3)


def analyze_suits(flop: List[str]) -> str:
    """Analyze suit distribution of flop.
    
    Returns
    -------
    str
        'mono' for monotone, 'tt' for twotone, 'rb' for rainbow
    """
    suits = [card[1] for card in flop]
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    max_count = max(suit_counts.values())
    
    if max_count == 3:
        return 'mono'
    elif max_count == 2:
        return 'tt'
    else:
        return 'rb'


def analyze_pairing(flop: List[str]) -> str:
    """Analyze pairing structure of flop.
    
    Returns
    -------
    str
        'np' for unpaired, 
        'lowpair'/'midpair'/'broadpair'/'acepair' for paired,
        'trips' for trips
    """
    ranks = [card[0] for card in flop]
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    max_count = max(rank_counts.values())
    
    if max_count == 1:
        return 'np'
    
    # Find the paired/trips rank
    paired_rank = None
    for rank, count in rank_counts.items():
        if count == max_count:
            paired_rank = rank
            break
    
    # Categorize the rank
    if paired_rank == 'A':
        rank_category = 'ace'
    elif paired_rank in ['K', 'Q', 'J']:
        rank_category = 'broad'
    elif paired_rank in ['T', '9', '8']:
        rank_category = 'mid'
    else:  # 2-7
        rank_category = 'low'
    
    if max_count == 3:
        return 'trips'  # Just "trips" since hirank shows the rank
    elif max_count == 2:
        return f'{rank_category}pair'
    else:
        return 'np'


def analyze_hirank(flop: List[str]) -> str:
    """Analyze high card structure of flop.
    
    Returns
    -------
    str
        'ah' for ace high, 'bh' for broadway high, 'mh' for mid-high, 'low' for low
    """
    ranks = [card[0] for card in flop]
    rank_values = [RANK_VALUES[rank] for rank in ranks]
    max_rank_value = max(rank_values)
    max_rank = RANKS[max_rank_value]
    
    if max_rank == 'A':
        return 'ah'
    elif max_rank in ['K', 'Q', 'J']:
        return 'bh'
    elif max_rank in ['T', '9', '8']:
        return 'mh'
    else:
        return 'low'


def analyze_connectivity(flop: List[str]) -> str:
    """Analyze straight potential of flop using the new algorithm.
    
    Parameters
    ----------
    flop : List[str]
        List of three cards, e.g. ["9s","7h","6d"]
    
    Returns
    -------
    str
        connectivity ∈ {"high","semi","dry"}
    """
    # A) Extract numeric ranks 2..14 (T=10, J=11, Q=12, K=13, A=14)
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    ranks = [rank_map[card[0]] for card in flop]
    
    # B) If A is present, also consider A=1 (wheel) and keep whichever 
    #    representation gives the smaller span
    if 14 in ranks:  # A is present
        # Try with A=14
        ranks_high = ranks[:]
        
        # Try with A=1
        ranks_low = [1 if r == 14 else r for r in ranks]
        
        # Calculate spans for both representations
        span_high = max(ranks_high) - min(ranks_high)
        span_low = max(ranks_low) - min(ranks_low)
        
        # Keep whichever gives smaller span
        if span_low < span_high:
            ranks = ranks_low
    
    # C) Remove duplicates (paired/trips keep one copy). Let R be the
    #    sorted list of unique ranks (length 1-3).
    R = sorted(list(set(ranks)))
    
    # D) Apply rules based on |R|
    if len(R) == 1:
        # trips → "dry"
        tag = "dry"
    
    elif len(R) == 2:
        # paired: gap = R[1]-R[0]
        gap = R[1] - R[0]
        if gap <= 1:
            tag = "semi"
        else:
            tag = "dry"
    
    else:  # len(R) == 3
        # unpaired: gaps g1 = R[1]-R[0], g2 = R[2]-R[1]
        g1 = R[1] - R[0]
        g2 = R[2] - R[1]
        span = g1 + g2
        
        # Check for direct connections first
        if g1 == 1 or g2 == 1:
            # Special case: complete straights (span <= 2) start as high
            if span <= 2:
                tag = "high"  # Complete straights start as high
                # But still apply edge penalty demotion
                if min(R) <= 3 or max(R) >= 13:
                    tag = "semi"  # Demote by 1 level due to edge penalty
            else:
                # At least one direct connection, check edge penalty first
                has_edge_penalty = False
                
                if g1 == 1:  # R[0] and R[1] are connected
                    connected_pair = [R[0], R[1]]
                    if min(connected_pair) <= 3 or max(connected_pair) >= 13:
                        has_edge_penalty = True
                
                if g2 == 1:  # R[1] and R[2] are connected
                    connected_pair = [R[1], R[2]]
                    if min(connected_pair) <= 3 or max(connected_pair) >= 13:
                        has_edge_penalty = True
                
                # If any connected pair has edge penalty, becomes dry
                if has_edge_penalty:
                    tag = "dry"
                else:
                    # No edge penalty, use span logic but ensure at least semi
                    if span <= 4:
                        tag = "high"
                    else:
                        tag = "semi"
        
        else:
            # No direct connections, use original span-based logic
            if span <= 4:
                tag = "high"
            elif span <= 5 and (g1 <= 2 or g2 <= 2):
                tag = "semi"
            else:
                tag = "dry"
    
    # E) Edge-penalty: if min(R) ≤ 3 or max(R) ≥ 13
    #    and current tag ≠ "dry": demote one level
    #    ("high"→"semi", "semi"→"dry")
    #    Skip this for unpaired cases with direct connections (already handled above)
    skip_general_edge_penalty = (len(R) == 3 and 
                                ((R[1] - R[0]) == 1 or (R[2] - R[1]) == 1))
    
    if not skip_general_edge_penalty and (min(R) <= 3 or max(R) >= 13) and tag != "dry":
        if tag == "high":
            tag = "semi"
        elif tag == "semi":
            tag = "dry"
    
    return tag


def format_flop(flop: List[str]) -> str:
    """Format flop cards as a string, sorted by rank (high to low)."""
    # Sort cards by rank value (high to low)
    sorted_flop = sorted(flop, key=lambda card: RANK_VALUES[card[0]], reverse=True)
    return ''.join(sorted_flop)


def determine_raiser_caller_names(scenario: str, range_data: Dict[str, Any]) -> Tuple[str, str, str]:
    """Determine raiser and caller names for filename, plus pot type.
    
    Returns
    -------
    Tuple[str, str, str]
        (raiser_name, caller_name, pot_type)
    """
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        return "ERROR", "ERROR", "ERROR"
    
    # Find the last raiser and the caller
    last_raiser = None
    caller = None
    
    for position, action in reversed(actions):
        if action == "call" and caller is None:
            caller = position
        elif action in ["raise", "3bet", "4bet"] and last_raiser is None:
            last_raiser = position
    
    if not last_raiser or not caller:
        return "ERROR", "ERROR", "ERROR"
    
    # Determine pot type
    if len(actions) == 2:
        pot_type = "SRP"
    elif any(action == "4bet" for _, action in actions):
        pot_type = "4bet"
    elif any(action == "3bet" for _, action in actions):
        pot_type = "3bet"
    else:
        pot_type = "SRP"
    
    # Determine naming based on pot type
    if pot_type == "SRP":
        # For SRP: use actual positions, but check if caller maps to position group
        raiser_name = last_raiser
        
        # Check if we need to use position group for caller
        # Look at what ranges were extracted to determine grouping
        basic_ranges = range_data.get('basic_ranges', {})
        caller_name = caller
        
        # If caller range came from a grouped block, use the group name
        for pos, (block_header, _) in basic_ranges.items():
            if pos == caller and "raise," in block_header:
                # Extract the raiser group from block header
                parts = block_header.split(", ")
                if len(parts) >= 4:
                    raiser_group = parts[3]
                    # Map caller to group if they're not the exact position in the block
                    caller_group = POSITION_GROUPS.get(caller, caller)
                    if raiser_group in ["EP", "MP"] and caller_group != caller:
                        caller_name = caller_group
                        break
    
    elif pot_type == "3bet":
        # For 3bet: raiser = actual position, caller = position group
        raiser_name = last_raiser
        caller_name = POSITION_GROUPS.get(caller, caller)
        
    else:  # 4bet
        # For 4bet: raiser = position group, caller = IP/OOP
        raiser_name = POSITION_GROUPS.get(last_raiser, last_raiser)
        
        # For 4bet caller, determine IP/OOP based on the original raiser vs caller
        original_raiser = None
        for pos, action in actions:
            if action == "raise":
                original_raiser = pos
                break
        
        if original_raiser:
            ip_pos, oop_pos = determine_ip_oop(caller, original_raiser)
            caller_name = "IP" if caller == ip_pos else "OOP"
        else:
            caller_name = "OOP"  # fallback
    
    return raiser_name, caller_name, pot_type


def generate_filename(scenario: str, flop: List[str], range_data: Dict[str, Any]) -> str:
    """Generate filename for solver input.
    
    Returns
    -------
    str
        Filename in format: flopCards_raiser_caller_potType_suits_pairing_hirank_connectivity.json
    """
    # Get raiser/caller names and pot type
    raiser_name, caller_name, pot_type = determine_raiser_caller_names(scenario, range_data)
    
    # Format flop cards (sorted by rank, high to low)
    flop_string = format_flop(flop)
    
    # Analyze flop texture
    suits = analyze_suits(flop)
    pairing = analyze_pairing(flop)
    hirank = analyze_hirank(flop)
    connectivity = analyze_connectivity(flop)
    
    # Construct filename with flop cards at the beginning
    filename = f"{flop_string}_{raiser_name}_{caller_name}_{pot_type}_{suits}_{pairing}_{hirank}_{connectivity}.json"
    
    return filename


def generate_complete_scenario() -> Dict[str, Any]:
    """Generate a complete scenario with preflop action, ranges, flop, and filename.
    
    Returns
    -------
    Dict[str, Any]
        Complete scenario information
    """
    # Generate preflop scenario
    scenario = generate_preflop_scenario()
    
    # Get ranges and frequencies
    range_data = get_ranges_with_frequencies(scenario)
    
    # Expand any ranges in basic_ranges that contain "+" notation
    if 'basic_ranges' in range_data:
        expanded_basic_ranges = {}
        for pos, (block_header, range_str) in range_data['basic_ranges'].items():
            expanded_range_str = expand_range_notation(range_str)
            expanded_basic_ranges[pos] = (block_header, expanded_range_str)
        range_data['basic_ranges'] = expanded_basic_ranges
    
    # Calculate pot and effective stack
    pot_info = calculate_pot_and_stacks(scenario)
    
    # Generate flop
    flop = generate_flop()
    
    # Generate filename
    filename = generate_filename(scenario, flop, range_data)
    
    return {
        'scenario': scenario,
        'range_data': range_data,
        'pot_info': pot_info,
        'flop': flop,
        'flop_string': format_flop(flop),
        'filename': filename,
        'texture': {
            'suits': analyze_suits(flop),
            'pairing': analyze_pairing(flop),
            'hirank': analyze_hirank(flop),
            'connectivity': analyze_connectivity(flop)
        }
    }


def calculate_pot_and_stacks(scenario: str) -> Dict[str, Any]:
    """Calculate pot size and effective stack for a given scenario.
    
    Parameters
    ----------
    scenario : str
        The preflop scenario string
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing pot size, effective stack, and detailed action breakdown
    """
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        return {"error": "Invalid scenario - need at least 2 actions"}
    
    # Track money contributed by each position
    contributions = {pos: 0 for pos in PREFLOP_POSITIONS + ["BB"]}
    
    # Start with blinds
    contributions["SB"] = SB_SIZE
    contributions["BB"] = BB_SIZE
    
    # Track the detailed action sequence with bet sizes
    detailed_actions = []
    
    # Process each action
    current_bet = 0
    
    for i, (position, action) in enumerate(actions):
        if action == "raise":
            if i == 0:  # First raise (initial raise)
                bet_size = INITIAL_RAISE_SIZE
                contributions[position] = bet_size
                current_bet = bet_size
                detailed_actions.append(f"{position} raise to {bet_size}bb")
            else:
                # This shouldn't happen in our scenario format, but handle gracefully
                detailed_actions.append(f"{position} raise")
                
        elif action == "call":
            # Call the current bet
            contributions[position] = current_bet
            detailed_actions.append(f"{position} call")
            
        elif action == "3bet":
            # Determine if this is IP or OOP 3bet
            original_raiser = actions[0][0]  # First action should be the raise
            ip_pos, oop_pos = determine_ip_oop(position, original_raiser)
            
            if position == ip_pos:
                # IP 3bet: 3x the original raise
                bet_size = INITIAL_RAISE_SIZE * IP_3BET_MULTIPLIER
            else:
                # OOP 3bet: fixed size
                bet_size = OOP_3BET_SIZE
            
            contributions[position] = bet_size
            current_bet = bet_size
            detailed_actions.append(f"{position} 3bet to {bet_size}bb")
            
        elif action == "4bet":
            # Determine if this is IP or OOP 4bet
            # Find the 3better
            threebetter = None
            for j in range(i):
                if actions[j][1] == "3bet":
                    threebetter = actions[j][0]
                    break
            
            if threebetter:
                ip_pos, oop_pos = determine_ip_oop(position, threebetter)
                
                if position == ip_pos:
                    bet_size = IP_4BET_SIZE
                else:
                    bet_size = OOP_4BET_SIZE
            else:
                # Fallback
                bet_size = IP_4BET_SIZE
                
            contributions[position] = bet_size
            current_bet = bet_size
            detailed_actions.append(f"{position} 4bet to {bet_size}bb")
    
    # Calculate total pot
    total_pot = sum(contributions.values())
    
    # Find the two players involved in the final action
    final_positions = []
    for position, action in actions:
        if position not in final_positions:
            final_positions.append(position)
    
    # Calculate effective stack (remaining stack of the player with less money left)
    if len(final_positions) >= 2:
        remaining_stacks = []
        for pos in final_positions:
            remaining = STARTING_STACK - contributions[pos]
            remaining_stacks.append(remaining)
        effective_stack = min(remaining_stacks)
    else:
        effective_stack = STARTING_STACK
    
    return {
        "scenario": scenario,
        "detailed_actions": detailed_actions,
        "contributions": contributions,
        "total_pot": total_pot,
        "effective_stack": effective_stack,
        "final_positions": final_positions
    }


def format_detailed_scenario(pot_info: Dict[str, Any]) -> str:
    """Format the detailed scenario with pot and stack information.
    
    Parameters
    ----------
    pot_info : Dict[str, Any]
        Dictionary from calculate_pot_and_stacks
        
    Returns
    -------
    str
        Formatted string with detailed action and pot/stack info
    """
    if "error" in pot_info:
        return pot_info["error"]
    
    # Join the detailed actions
    action_str = ", ".join(pot_info["detailed_actions"])
    
    # Format the final summary
    pot_size = pot_info["total_pot"]
    effective_stack = pot_info["effective_stack"]
    
    return f"{action_str}. Pot size is {pot_size}bb, and effective stack is {effective_stack}bb"


def display_ranges_simple(range_data: Dict[str, Any], scenario: str) -> None:
    """Display ranges in simple IP/OOP format.
    
    Parameters
    ----------
    range_data : Dict[str, Any]
        Range data from get_ranges_with_frequencies
    scenario : str
        The scenario string to determine IP/OOP
    """
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        print("Error: Need at least 2 actions")
        return
    
    # Get the two main players from the scenario itself
    acting_pos = range_data.get('acting_position', '')
    prev_acting_pos = range_data.get('prev_acting_position', '')
    
    # If range data is missing, extract positions from scenario
    if not acting_pos or not prev_acting_pos:
        # Extract the last two players from actions
        unique_positions = []
        for pos, _ in actions:
            if pos not in unique_positions:
                unique_positions.append(pos)
        
        if len(unique_positions) >= 2:
            prev_acting_pos = unique_positions[0]  # First raiser
            acting_pos = unique_positions[-1]     # Last actor (usually caller)
        else:
            print("Error: Cannot determine positions")
            return
    
    # Determine IP/OOP
    ip_pos, oop_pos = determine_ip_oop(acting_pos, prev_acting_pos)
    
    # Get ranges if available
    acting_frequencies = range_data.get('frequencies', {})
    prev_frequencies = range_data.get('prev_frequencies', {})
    acting_primary_action = range_data.get('primary_action', 'call')
    prev_primary_action = range_data.get('prev_primary_action', 'RFI')
    
    # Get basic_ranges for fallback
    basic_ranges = range_data.get('basic_ranges', {})
    
    # Handle missing range data
    if not acting_frequencies and not prev_frequencies:
        print(f"{ip_pos} (IP) range: [preflop chart not found]")
        print(f"{oop_pos} (OOP) range: [preflop chart not found]")
        return
    
    # Format ranges - use actual ranges from basic_ranges when needed
    if acting_frequencies:
        acting_range = format_range_simple(acting_frequencies, acting_primary_action)
    elif acting_pos in basic_ranges:
        _, range_str = basic_ranges[acting_pos]
        acting_range = expand_range_notation(range_str)
    else:
        acting_range = "[no range]"
    
    if prev_frequencies:
        prev_range = format_range_simple(prev_frequencies, prev_primary_action)
    elif prev_acting_pos in basic_ranges:
        _, range_str = basic_ranges[prev_acting_pos]
        prev_range = expand_range_notation(range_str)
    else:
        prev_range = "[no range]"
    
    if acting_pos == ip_pos:
        print(f"{ip_pos} (IP) range: {acting_range}")
        print(f"{oop_pos} (OOP) range: {prev_range}")
    else:
        print(f"{ip_pos} (IP) range: {prev_range}")
        print(f"{oop_pos} (OOP) range: {acting_range}")


def expand_range_notation(range_str: str) -> str:
    """Expand range notation like '22+, A2s+' into individual hands.
    
    Parameters
    ----------
    range_str : str
        Range string that may contain + notation
        
    Returns
    -------
    str
        Expanded range string with individual hands
    """
    if not range_str or range_str in ["no range", "full RFI range"]:
        return range_str
    
    try:
        # Import expand_range from local preflop module
        from preflop import expand_range
        
        # If the range string contains "+", it needs expansion
        if "+" in range_str:
            expanded_hands = expand_range(range_str)
            return ",".join(sorted(expanded_hands))
        else:
            # Already individual hands, just return as is
            return range_str
    except Exception as e:
        print(f"Warning: Could not expand range '{range_str}': {e}")
        return range_str


def format_range_simple(frequencies: Dict[str, float], primary_action: str) -> str:
    """Format frequency range in a simple way."""
    if not frequencies:
        return "no range"
    
    # Count hands with different frequencies
    full_freq_hands = []
    mixed_freq_hands = []
    
    for hand in sorted(frequencies.keys()):
        if frequencies[hand] == 1.0:
            full_freq_hands.append(hand)
        else:
            mixed_freq_hands.append(f"{hand}:{frequencies[hand]}")
    
    # Create display - always show actual hands
    all_hands = mixed_freq_hands + full_freq_hands
    return ",".join(all_hands)


def generate_and_analyze_scenario() -> None:
    """Generate a random scenario and provide simple analysis."""
    # Generate complete scenario with flop and filename
    complete_data = generate_complete_scenario()
    
    scenario = complete_data['scenario']
    range_data = complete_data['range_data']
    pot_info = complete_data['pot_info']
    flop_string = complete_data['flop_string']
    filename = complete_data['filename']
    
    print(f"Generated scenario: {scenario}")
    
    # Display ranges in IP/OOP format
    display_ranges_simple(range_data, scenario)
    
    # Display pot calculation (simplified)
    detailed_scenario = format_detailed_scenario(pot_info)
    print(f"Pot calculation: {detailed_scenario}")
    
    # Display flop
    print(f"Flop: {flop_string}")
    
    # Display filename
    print(f"Solver input filename: {filename}")


def main():
    """Main function to run the solver input generator."""
    print("Solver Input Generator")
    print("-" * 25)
    
    # Generate and analyze one scenario
    generate_and_analyze_scenario()
    
    # Interactive mode
    while True:
        choice = input("\nGenerate another scenario? (y/n/q to quit) [y]: ").lower()
        if choice in ['q', 'quit']:
            break
        elif choice in ['n', 'no']:
            custom_scenario = input("Enter custom scenario: ").strip()
            if custom_scenario:
                print(f"Generated scenario: {custom_scenario}")
                
                # Get ranges and frequencies
                range_data = get_ranges_with_frequencies(custom_scenario)
                display_ranges_simple(range_data, custom_scenario)
                
                # Calculate pot and effective stack
                pot_info = calculate_pot_and_stacks(custom_scenario)
                detailed_scenario = format_detailed_scenario(pot_info)
                print(f"Pot calculation: {detailed_scenario}")
                
                # Generate flop and filename for custom scenario
                flop = generate_flop()
                flop_string = format_flop(flop)
                filename = generate_filename(custom_scenario, flop, range_data)
                
                print(f"Flop: {flop_string}")
                print(f"Solver input filename: {filename}")
        else:
            generate_and_analyze_scenario()


if __name__ == "__main__":
    main() 