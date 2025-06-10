#!/usr/bin/env python3
"""
Preflop Scenario Generator

Randomly generates poker preflop action sequences and extracts ranges.
"""

import os
import sys
import random
from typing import List, Tuple, Optional, Dict, Any

# Import from local preflop module (no path addition needed)
from preflop import PreflopLookup, expand_range


# Preflop position order (UTG to SB)
PREFLOP_POSITIONS = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB"]

# Postflop position order (early to late position)
POSTFLOP_POSITIONS = ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"]

# Position group mappings for preflop chart
POSITION_GROUPS = {
    "UTG": "EP",
    "UTG+1": "EP", 
    "LJ": "MP",
    "HJ": "MP",
    "CO": "CO",
    "BTN": "BTN",
    "SB": "SB",
    "BB": "BB"
}


def get_positions_after(raiser_pos: str) -> List[str]:
    """Get all positions that act after the raiser in preflop action.
    
    Excludes players from the same position group as the raiser.
    For example, if UTG raises, UTG+1 won't be included.
    If LJ raises, HJ won't be included.
    
    Parameters
    ----------
    raiser_pos : str
        The position of the initial raiser
        
    Returns
    -------
    List[str]
        List of positions that can act after the raiser
    """
    try:
        raiser_index = PREFLOP_POSITIONS.index(raiser_pos)
        # Get all positions after the raiser, plus BB (who always acts after preflop raises)
        positions_after = PREFLOP_POSITIONS[raiser_index + 1:] + ["BB"]
        
        # Remove duplicates and filter out players from the same position group
        raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
        filtered_positions = []
        
        for pos in positions_after:
            # Skip if this position is in the same group as the raiser
            if POSITION_GROUPS.get(pos, pos) == raiser_group and pos != raiser_pos:
                continue
            # Avoid duplicates (BB might already be in the list)
            if pos not in filtered_positions:
                filtered_positions.append(pos)
        
        return filtered_positions
    except ValueError:
        return ["BB"]  # If raiser position not found, only BB can act


def choose_action(is_facing_4bet: bool = False, is_facing_3bet: bool = False) -> str:
    """Choose an action based on probability weights.
    
    Parameters
    ----------
    is_facing_4bet : bool
        If True, always returns 'call' (100% call rate)
    is_facing_3bet : bool
        If True, uses 85% call / 15% 4bet frequencies
        
    Returns
    -------
    str
        The chosen action: 'call', '3bet', or '4bet'
    """
    if is_facing_4bet:
        return "call"  # 100% call rate vs 4bet
    
    if is_facing_3bet:
        # 85% call, 15% 4bet when facing 3bet
        actions = ["call"] * 85 + ["4bet"] * 15
        return random.choice(actions)
    
    # Initial response to raise: 75% call, 25% 3bet
    actions = ["call"] * 75 + ["3bet"] * 25
    return random.choice(actions)


def generate_preflop_scenario() -> str:
    """Generate a random preflop scenario with specific action frequencies.
    
    Frequencies:
    - Initial response to raise: 75% call, 25% 3bet
    - Response to 3bet: 85% call, 15% 4bet  
    - Response to 4bet: 100% call
    
    Returns
    -------
    str
        A preflop action string like "UTG raise, BB call" or "UTG raise, BTN 3bet, UTG call"
    """
    # Step 1: Choose random raiser from UTG to SB
    raiser = random.choice(PREFLOP_POSITIONS)
    
    # Step 2: Choose random position that acts after the raiser
    positions_after = get_positions_after(raiser)
    if not positions_after:
        # Edge case - if no positions after, just return the raise
        return f"{raiser} raise"
    
    responder = random.choice(positions_after)
    
    # Start building the action sequence
    actions = [(raiser, "raise")]
    
    # Step 3: Choose first action (75% call, 25% 3bet)
    first_action = choose_action()
    
    if first_action == "call":
        actions.append((responder, "call"))
    else:  # 3bet
        actions.append((responder, "3bet"))
        
        # Step 4: Original raiser responds to 3bet (85% call, 15% 4bet)
        response_to_3bet = choose_action(is_facing_3bet=True)
        
        if response_to_3bet == "call":
            actions.append((raiser, "call"))
        else:  # 4bet
            actions.append((raiser, "4bet"))
            
            # Step 5: Responder faces 4bet - always calls (100% call rate)
            final_action = choose_action(is_facing_4bet=True)
            actions.append((responder, final_action))
    
    # Convert actions to readable string
    action_strings = []
    for position, action in actions:
        action_strings.append(f"{position} {action}")
    
    return ", ".join(action_strings)


def parse_scenario(scenario: str) -> List[Tuple[str, str]]:
    """Parse a scenario string into a list of (position, action) tuples.
    
    Parameters
    ----------
    scenario : str
        Scenario string like "UTG raise, BB call"
        
    Returns
    -------
    List[Tuple[str, str]]
        List of (position, action) tuples
    """
    actions = []
    parts = scenario.split(", ")
    for part in parts:
        tokens = part.strip().split()
        if len(tokens) >= 2:
            position = tokens[0]
            action = tokens[1]
            actions.append((position, action))
    return actions


def determine_ip_oop(pos1: str, pos2: str) -> Tuple[str, str]:
    """Determine which position is IP and OOP postflop.
    
    Parameters
    ----------
    pos1 : str
        First position
    pos2 : str
        Second position
        
    Returns
    -------
    Tuple[str, str]
        (ip_position, oop_position)
    """
    try:
        pos1_index = POSTFLOP_POSITIONS.index(pos1)
        pos2_index = POSTFLOP_POSITIONS.index(pos2)
        
        if pos1_index < pos2_index:
            return pos2, pos1  # pos2 is IP, pos1 is OOP
        else:
            return pos1, pos2  # pos1 is IP, pos2 is OOP
    except ValueError:
        # Default fallback
        return pos1, pos2


def load_preflop_chart() -> Dict[str, Dict[str, str]]:
    """Load the preflop chart from file.
    
    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping block headers to position ranges
    """
    chart_path = os.path.join(os.path.dirname(__file__), "preflop_chart.txt")
    if not os.path.exists(chart_path):
        print(f"Warning: Preflop chart not found at {chart_path}")
        return {}
    
    chart = {}
    current_block = None
    
    with open(chart_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Cash, 100bb, 8-max"):
                # This is a block header
                current_block = line
                chart[current_block] = {}
            elif current_block and ":" in line:
                # This is a position range line (e.g., "BB: range")
                pos, range_str = line.split(":", 1)
                chart[current_block][pos.strip()] = range_str.strip()
            elif current_block and line and not line.startswith("Cash"):
                # This is a direct range line without position prefix
                # Used for blocks like 4bet, allin where ranges apply universally
                chart[current_block]["ALL"] = line.strip()
    
    return chart


def get_ranges_from_chart(scenario: str) -> Dict[str, Tuple[str, str]]:
    """Get ranges for both players from the preflop chart.
    
    Parameters
    ----------
    scenario : str
        The preflop scenario string
        
    Returns
    -------
    Dict[str, Tuple[str, str]]
        Dictionary with player positions as keys, values are (block_header, range_string) tuples
    """
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        return {}
    
    chart = load_preflop_chart()
    if not chart:
        return {}
    
    results = {}
    
    # Check if this is a single raised pot (only 2 actions total)
    if len(actions) == 2:
        # Single raised pot: first action is raise, second is call
        raiser_pos, raiser_action = actions[0]
        caller_pos, caller_action = actions[1]
        
        if raiser_action == "raise" and caller_action == "call":
            # Raiser gets RFI ranges
            rfi_block = "Cash, 100bb, 8-max, RFI"
            if rfi_block in chart and raiser_pos in chart[rfi_block]:
                results[raiser_pos] = (rfi_block, chart[rfi_block][raiser_pos])
            
            # Caller gets calling ranges
            raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
            call_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, call"
            if call_block in chart:
                # Try exact position first, then position group
                if caller_pos in chart[call_block]:
                    results[caller_pos] = (call_block, chart[call_block][caller_pos])
                else:
                    caller_group = POSITION_GROUPS.get(caller_pos, caller_pos)
                    if caller_group in chart[call_block]:
                        results[caller_pos] = (call_block, chart[call_block][caller_group])
    
    else:
        # Multi-way pot: look at last 2 actions
        last_action = actions[-1]
        second_last_action = actions[-2]
        
        acting_pos, acting_action = last_action
        previous_pos, previous_action = second_last_action
        
        # Determine IP/OOP
        ip_pos, oop_pos = determine_ip_oop(acting_pos, previous_pos)
        
        # Handle the acting player's range (the last action)
        if previous_action == "3bet" and acting_action == "call":
            # Calling a 3bet
            if previous_pos == ip_pos:
                # IP 3bet, we're calling
                call_block = f"Cash, 100bb, 8-max, 3bet, IP, call"
            else:
                # OOP 3bet, we're calling  
                call_block = f"Cash, 100bb, 8-max, 3bet, OOP, call"
            
            if call_block in chart:
                # Try exact position first, then position group
                if acting_pos in chart[call_block]:
                    results[acting_pos] = (call_block, chart[call_block][acting_pos])
                else:
                    acting_group = POSITION_GROUPS.get(acting_pos, acting_pos)
                    if acting_group in chart[call_block]:
                        results[acting_pos] = (call_block, chart[call_block][acting_group])
        
        elif previous_action == "4bet" and acting_action == "call":
            # Calling a 4bet
            if previous_pos == ip_pos:
                call_block = f"Cash, 100bb, 8-max, 4bet, IP, call"
            else:
                call_block = f"Cash, 100bb, 8-max, 4bet, OOP, call"
            
            if call_block in chart:
                # Try exact position first, then position group
                if acting_pos in chart[call_block]:
                    results[acting_pos] = (call_block, chart[call_block][acting_pos])
                else:
                    acting_group = POSITION_GROUPS.get(acting_pos, acting_pos)
                    if acting_group in chart[call_block]:
                        results[acting_pos] = (call_block, chart[call_block][acting_group])
        
        elif previous_action == "raise" and acting_action == "3bet":
            # 3betting a raise
            # Find the original raiser to determine position group
            original_raiser = None
            for pos, action in actions:
                if action == "raise":
                    original_raiser = pos
                    break
            
            if original_raiser:
                raiser_group = POSITION_GROUPS.get(original_raiser, original_raiser)
                threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
                if threbet_block in chart and acting_pos in chart[threbet_block]:
                    results[acting_pos] = (threbet_block, chart[threbet_block][acting_pos])
        
        elif previous_action == "3bet" and acting_action == "4bet":
            # 4betting a 3bet
            if previous_pos == ip_pos:
                fourbet_block = f"Cash, 100bb, 8-max, 3bet, IP, 4bet"
            else:
                fourbet_block = f"Cash, 100bb, 8-max, 3bet, OOP, 4bet"
            
            if fourbet_block in chart:
                # Try exact position first, then position group
                if acting_pos in chart[fourbet_block]:
                    results[acting_pos] = (fourbet_block, chart[fourbet_block][acting_pos])
                else:
                    acting_group = POSITION_GROUPS.get(acting_pos, acting_pos)
                    if acting_group in chart[fourbet_block]:
                        results[acting_pos] = (fourbet_block, chart[fourbet_block][acting_group])
        
        # Handle the previous player's range (the second-to-last action)
        if previous_action == "3bet":
            # Find who they were 3betting against
            threebetter_pos = previous_pos
            original_raiser = None
            for pos, action in actions[:-1]:  # Exclude the last action
                if action == "raise":
                    original_raiser = pos
                    break
            
            if original_raiser:
                raiser_group = POSITION_GROUPS.get(original_raiser, original_raiser)
                threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
                if threbet_block in chart and threebetter_pos in chart[threbet_block]:
                    results[threebetter_pos] = (threbet_block, chart[threbet_block][threebetter_pos])
        
        elif previous_action == "4bet":
            # 4betting a 3bet - find the appropriate block
            fourbetter_pos = previous_pos
            
            # Look for the 3bet action before this 4bet
            threebetter_pos = None
            for i in range(len(actions) - 2, -1, -1):
                if actions[i][1] == "3bet":
                    threebetter_pos = actions[i][0]
                    break
            
            if threebetter_pos:
                # Determine IP/OOP between 3better and 4better
                if determine_ip_oop(threebetter_pos, fourbetter_pos)[0] == threebetter_pos:
                    # 3better is IP
                    fourbet_block = f"Cash, 100bb, 8-max, 3bet, IP, 4bet"
                else:
                    # 3better is OOP
                    fourbet_block = f"Cash, 100bb, 8-max, 3bet, OOP, 4bet"
                
                if fourbet_block in chart:
                    # Try exact position first, then position group
                    if fourbetter_pos in chart[fourbet_block]:
                        results[fourbetter_pos] = (fourbet_block, chart[fourbet_block][fourbetter_pos])
                    else:
                        fourbetter_group = POSITION_GROUPS.get(fourbetter_pos, fourbetter_pos)
                        if fourbetter_group in chart[fourbet_block]:
                            results[fourbetter_pos] = (fourbet_block, chart[fourbet_block][fourbetter_group])
        
        elif previous_action == "raise":
            # This is the original raiser - they get RFI ranges only if this is the first raise
            if len([a for a in actions if a[1] == "raise"]) == 1:
                # Only one raise in the sequence, so this is RFI
                rfi_block = "Cash, 100bb, 8-max, RFI"
                if rfi_block in chart and previous_pos in chart[rfi_block]:
                    results[previous_pos] = (rfi_block, chart[rfi_block][previous_pos])
    
    return results


def print_section(title: str) -> None:
    """Print a section title with formatting."""
    red = '\033[91m'
    reset = '\033[0m'
    print(f"\n{red}{'=' * 60}{reset}")
    print(f"{red}{'=' * 5}{reset} {title} {red}{'=' * 5}{reset}")
    print(f"{red}{'=' * 60}{reset}")


def main():
    """Run the preflop scenario generator."""
    print_section("Preflop Scenario Generator with Range Extraction")
    
    # Generate one random scenario and analyze it immediately
    scenario = generate_preflop_scenario()
    print(f"Generated scenario: {scenario}")
    
    analyze_scenario_with_chart(scenario)
    
    print_section("Interactive Mode")
    
    # Let user analyze additional scenarios
    while True:
        choice = input("\nGenerate new scenario? (y/n/q to quit) [y]: ").lower()
        if choice in ['q', 'quit']:
            break
        elif choice in ['n', 'no']:
            # Let user input their own scenario
            custom_scenario = input("Enter custom scenario: ").strip()
            if custom_scenario:
                analyze_scenario_with_chart(custom_scenario)
        else:
            # Generate new scenario
            scenario = generate_preflop_scenario()
            print(f"Generated: {scenario}")
            analyze_scenario_with_chart(scenario)


def analyze_scenario(scenario: str) -> None:
    """Analyze a preflop scenario and show ranges from the original lookup.
    
    Parameters
    ----------
    scenario : str
        The preflop action string to analyze
    """
    try:
        lookup = PreflopLookup()
        
        # Try to get ranges for the scenario
        ranges = lookup.get_ranges(scenario)
        
        print(f"\nScenario: {scenario}")
        print(f"Hero range: {ranges.get('hero', 'Not found')}")
        print(f"Villain range: {ranges.get('villain', 'Not found')}")
        
    except Exception as e:
        print(f"Error analyzing scenario '{scenario}': {e}")
        print("This scenario might not be supported by the preflop lookup system.")


def get_ranges_with_frequencies(scenario: str) -> Dict[str, Any]:
    """Get ranges with frequencies for overlapping hands.
    
    Parameters
    ----------
    scenario : str
        The preflop scenario string
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with position ranges and frequency information
    """
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        return {}
    
    chart = load_preflop_chart()
    if not chart:
        return {}
    
    results = {}
    
    # Get basic ranges first
    basic_ranges = get_ranges_from_chart(scenario)
    results['basic_ranges'] = basic_ranges
    
    # Handle frequency calculations based on scenario type
    if len(actions) == 2:
        # Single raised pot
        last_action = actions[-1]
        acting_pos, acting_action = last_action
        
        if acting_action == "call":
            results.update(get_single_raised_pot_frequencies(actions, chart))
    
    else:
        # Multi-action pot (3bet, 4bet, etc.)
        results.update(get_multi_action_frequencies(actions, chart))
    
    return results


def get_single_raised_pot_frequencies(actions: List[Tuple[str, str]], chart: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Get frequency data for single raised pots."""
    results = {}
    
    raiser_pos, raiser_action = actions[0]
    caller_pos, caller_action = actions[1]
    
    # Get raiser's RFI range
    rfi_block = "Cash, 100bb, 8-max, RFI"
    rfi_range_str = ""
    if rfi_block in chart and raiser_pos in chart[rfi_block]:
        rfi_range_str = chart[rfi_block][raiser_pos]
    
    # Store raiser's range information
    results['prev_call_range'] = {
        'block': rfi_block,
        'range': rfi_range_str
    }
    results['prev_acting_position'] = raiser_pos
    results['prev_primary_action'] = 'RFI'
    
    # Get both calling and 3betting ranges for the caller
    raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
    
    # Get calling range
    call_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, call"
    call_range_str = ""
    if call_block in chart:
        if caller_pos in chart[call_block]:
            call_range_str = chart[call_block][caller_pos]
        else:
            caller_group = POSITION_GROUPS.get(caller_pos, caller_pos)
            if caller_group in chart[call_block]:
                call_range_str = chart[call_block][caller_group]
    
    # Get 3betting range
    threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
    threbet_range_str = ""
    if threbet_block in chart:
        if caller_pos in chart[threbet_block]:
            threbet_range_str = chart[threbet_block][caller_pos]
        else:
            caller_group = POSITION_GROUPS.get(caller_pos, caller_pos)
            if caller_group in chart[threbet_block]:
                threbet_range_str = chart[threbet_block][caller_group]
    
    # Store both ranges for caller
    results['call_range'] = {
        'block': call_block,
        'range': call_range_str
    }
    results['threbet_range'] = {
        'block': threbet_block,
        'range': threbet_range_str
    }
    results['primary_action'] = 'call'
    
    # Calculate frequencies for caller
    if call_range_str and threbet_range_str:
        try:
            call_hands = set(expand_range(call_range_str))
            threbet_hands = set(expand_range(threbet_range_str))
            
            # Create frequency dictionary - iterate over the primary action range
            frequencies = {}
            
            # Determine which range to iterate over based on primary action
            if results['primary_action'] == "call":
                primary_hands = call_hands
                secondary_hands = threbet_hands
            else:
                # For 3bet, 4bet, allin - iterate over the secondary action range
                primary_hands = threbet_hands
                secondary_hands = call_hands
            
            for hand in sorted(primary_hands):
                if hand in secondary_hands:
                    # Overlapping hand - split 50/50
                    frequencies[hand] = 0.5
                else:
                    # Only in primary action range - full frequency
                    frequencies[hand] = 1.0
            
            results['frequencies'] = frequencies
            results['acting_position'] = caller_pos
            
        except Exception as e:
            print(f"Error calculating frequencies: {e}")
    
    return results


def get_multi_action_frequencies(actions: List[Tuple[str, str]], chart: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Get frequency data for multi-action pots (3bet, 4bet, etc.)."""
    results = {}
    
    last_action = actions[-1]
    second_last_action = actions[-2]
    
    acting_pos, acting_action = last_action
    previous_pos, previous_action = second_last_action
    
    # Determine IP/OOP
    ip_pos, oop_pos = determine_ip_oop(acting_pos, previous_pos)
    
    # Handle the last acting player (caller)
    if acting_action == "call":
        if previous_action == "3bet":
            # Calling a 3bet - look for call and 4bet ranges
            position_type = "IP" if previous_pos == ip_pos else "OOP"
            
            call_block = f"Cash, 100bb, 8-max, 3bet, {position_type}, call"
            fourbet_block = f"Cash, 100bb, 8-max, 3bet, {position_type}, 4bet"
            
            results.update(get_frequencies_for_two_actions(
                acting_pos, call_block, fourbet_block, chart, "call"
            ))
            
        elif previous_action == "4bet":
            # Calling a 4bet - look for call and allin ranges
            position_type = "IP" if previous_pos == ip_pos else "OOP"
            
            call_block = f"Cash, 100bb, 8-max, 4bet, {position_type}, call"
            allin_block = f"Cash, 100bb, 8-max, 4bet, {position_type}, allin"
            
            results.update(get_frequencies_for_two_actions(
                acting_pos, call_block, allin_block, chart, "call"
            ))
    
    # Handle the previous acting player (3better/4better)
    if previous_action == "3bet":
        # Find the original raiser to determine correct block
        original_raiser = None
        for pos, action in actions[:-1]:
            if action == "raise":
                original_raiser = pos
                break
        
        if original_raiser:
            raiser_group = POSITION_GROUPS.get(original_raiser, original_raiser)
            
            call_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, call"
            threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
            
            prev_results = get_frequencies_for_two_actions(
                previous_pos, call_block, threbet_block, chart, "3bet"
            )
            
            # Add prefix to avoid conflicts
            for key, value in prev_results.items():
                if key.startswith(('call_range', 'threbet_range', 'frequencies')):
                    results[f"prev_{key}"] = value
                else:
                    results[f"prev_{key}"] = value
    
    elif previous_action == "4bet":
        # Find the 3better to determine correct block
        threebetter_pos = None
        for i in range(len(actions) - 2, -1, -1):
            if actions[i][1] == "3bet":
                threebetter_pos = actions[i][0]
                break
        
        if threebetter_pos:
            position_type = "IP" if threebetter_pos == ip_pos else "OOP"
            
            call_block = f"Cash, 100bb, 8-max, 3bet, {position_type}, call"
            fourbet_block = f"Cash, 100bb, 8-max, 3bet, {position_type}, 4bet"
            
            prev_results = get_frequencies_for_two_actions(
                previous_pos, call_block, fourbet_block, chart, "4bet"
            )
            
            # Add prefix to avoid conflicts
            for key, value in prev_results.items():
                if key.startswith(('call_range', 'threbet_range', 'frequencies')):
                    results[f"prev_{key}"] = value
                else:
                    results[f"prev_{key}"] = value
    
    return results


def get_frequencies_for_two_actions(position: str, action1_block: str, action2_block: str, 
                                  chart: Dict[str, Dict[str, str]], primary_action: str) -> Dict[str, Any]:
    """Get frequency data for a position with two possible actions."""
    results = {}
    
    # Get range for first action
    action1_range_str = ""
    if action1_block in chart:
        if position in chart[action1_block]:
            action1_range_str = chart[action1_block][position]
        elif "ALL" in chart[action1_block]:
            # Universal range (used for 4bet, allin blocks)
            action1_range_str = chart[action1_block]["ALL"]
        else:
            pos_group = POSITION_GROUPS.get(position, position)
            if pos_group in chart[action1_block]:
                action1_range_str = chart[action1_block][pos_group]
            else:
                # Fallback: try to find the closest position
                available_positions = list(chart[action1_block].keys())
                if available_positions:
                    # Use the most aggressive position available as fallback
                    fallback_pos = available_positions[-1] if available_positions else None
                    if fallback_pos:
                        action1_range_str = chart[action1_block][fallback_pos]
                        print(f"Warning: Using {fallback_pos} ranges as fallback for {position} in {action1_block}")
    
    # Get range for second action
    action2_range_str = ""
    if action2_block in chart:
        if position in chart[action2_block]:
            action2_range_str = chart[action2_block][position]
        elif "ALL" in chart[action2_block]:
            # Universal range (used for 4bet, allin blocks)
            action2_range_str = chart[action2_block]["ALL"]
        else:
            pos_group = POSITION_GROUPS.get(position, position)
            if pos_group in chart[action2_block]:
                action2_range_str = chart[action2_block][pos_group]
            else:
                # Fallback: try to find the closest position
                available_positions = list(chart[action2_block].keys())
                if available_positions:
                    # Use the most aggressive position available as fallback
                    fallback_pos = available_positions[-1] if available_positions else None
                    if fallback_pos:
                        action2_range_str = chart[action2_block][fallback_pos]
                        print(f"Warning: Using {fallback_pos} ranges as fallback for {position} in {action2_block}")
    
    # Store both ranges with proper naming
    results['call_range'] = {'block': action1_block, 'range': action1_range_str}
    
    # Determine second action type from block name
    if "4bet" in action2_block:
        results['fourbet_range'] = {'block': action2_block, 'range': action2_range_str}
        results['primary_action'] = primary_action
        results['secondary_action'] = "4bet"
    elif "allin" in action2_block:
        results['allin_range'] = {'block': action2_block, 'range': action2_range_str}
        results['primary_action'] = primary_action
        results['secondary_action'] = "allin"
    elif "3bet" in action2_block:
        results['threbet_range'] = {'block': action2_block, 'range': action2_range_str}
        results['primary_action'] = primary_action
        results['secondary_action'] = "3bet"
    else:
        # Fallback
        results['alt_range'] = {'block': action2_block, 'range': action2_range_str}
        results['primary_action'] = primary_action
        results['secondary_action'] = "unknown"
    
    # Calculate frequencies if both ranges exist
    if action1_range_str and action2_range_str:
        try:
            action1_hands = set(expand_range(action1_range_str))
            action2_hands = set(expand_range(action2_range_str))
            
            # Create frequency dictionary - iterate over the primary action range
            frequencies = {}
            
            # Determine which range to iterate over based on primary action
            if primary_action == "call":
                primary_hands = action1_hands
                secondary_hands = action2_hands
            else:
                # For 3bet, 4bet, allin - iterate over the secondary action range
                primary_hands = action2_hands
                secondary_hands = action1_hands
            
            for hand in sorted(primary_hands):
                if hand in secondary_hands:
                    # Overlapping hand - split 50/50
                    frequencies[hand] = 0.5
                else:
                    # Only in primary action range - full frequency
                    frequencies[hand] = 1.0
            
            results['frequencies'] = frequencies
            results['acting_position'] = position
            
        except Exception as e:
            print(f"Error calculating frequencies: {e}")
    
    return results


def analyze_scenario_with_chart(scenario: str) -> None:
    """Analyze a preflop scenario using the preflop chart.
    
    Parameters
    ----------
    scenario : str
        The preflop action string to analyze
    """
    print(f"\nScenario: {scenario}")
    
    # Parse the scenario to understand positions
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        print("Invalid scenario - need at least 2 actions")
        return
    
    # Determine IP/OOP
    pos1, pos2 = actions[-1][0], actions[-2][0]
    ip_pos, oop_pos = determine_ip_oop(pos1, pos2)
    print(f"IP (In Position): {ip_pos}")
    print(f"OOP (Out of Position): {oop_pos}")
    
    # Get ranges with frequency analysis
    range_data = get_ranges_with_frequencies(scenario)
    
    print_section("Range Analysis from Preflop Chart")
    
    # Display basic ranges first
    basic_ranges = range_data.get('basic_ranges', {})
    
    # Display ranges for the last acting player first
    if 'call_range' in range_data:
        acting_pos = range_data.get('acting_position', '')
        
        print(f"\n{acting_pos} ranges:")
        
        # Show the call range first
        if 'call_range' in range_data and range_data['call_range']['range']:
            call_info = range_data['call_range']
            print(f"Block: {call_info['block']}")
            print(f"{acting_pos}: {call_info['range']}")
        
        # Show additional range based on scenario type
        if 'threbet_range' in range_data and range_data['threbet_range']['range']:
            threbet_info = range_data['threbet_range']
            print(f"Block: {threbet_info['block']}")
            print(f"{acting_pos}: {threbet_info['range']}")
        
        elif 'fourbet_range' in range_data and range_data['fourbet_range']['range']:
            fourbet_info = range_data['fourbet_range']
            print(f"Block: {fourbet_info['block']}")
            print(f"{acting_pos}: {fourbet_info['range']}")
        
        elif 'allin_range' in range_data and range_data['allin_range']['range']:
            allin_info = range_data['allin_range']
            print(f"Block: {allin_info['block']}")
            print(f"{acting_pos}: {allin_info['range']}")
        
        # Show frequency array
        frequencies = range_data.get('frequencies', {})
        if frequencies:
            primary_action = range_data.get('primary_action', 'call')
            print(f"\n{acting_pos} frequency array ({primary_action} frequencies):")
            display_frequencies(frequencies)
    
    # Display ranges for the previous acting player
    if 'prev_call_range' in range_data:
        prev_acting_pos = range_data.get('prev_acting_position', '')
        
        print(f"\n{prev_acting_pos} ranges:")
        
        # Show the call range first
        if 'prev_call_range' in range_data and range_data['prev_call_range']['range']:
            call_info = range_data['prev_call_range']
            print(f"Block: {call_info['block']}")
            print(f"{prev_acting_pos}: {call_info['range']}")
        
        # Show additional range based on scenario type
        if 'prev_threbet_range' in range_data and range_data['prev_threbet_range']['range']:
            threbet_info = range_data['prev_threbet_range']
            print(f"Block: {threbet_info['block']}")
            print(f"{prev_acting_pos}: {threbet_info['range']}")
        
        elif 'prev_fourbet_range' in range_data and range_data['prev_fourbet_range']['range']:
            fourbet_info = range_data['prev_fourbet_range']
            print(f"Block: {fourbet_info['block']}")
            print(f"{prev_acting_pos}: {fourbet_info['range']}")
        
        # Show frequency array
        prev_frequencies = range_data.get('prev_frequencies', {})
        if prev_frequencies:
            primary_action = range_data.get('prev_primary_action', 'call')
            print(f"\n{prev_acting_pos} frequency array ({primary_action} frequencies):")
            display_frequencies(prev_frequencies)
        elif range_data.get('prev_primary_action') == 'RFI':
            # For RFI, no frequency array needed - they just raise with their full range
            print(f"\n{prev_acting_pos} raises with their full RFI range")
    
    # If no frequency data, show basic ranges
    if not range_data.get('call_range') and not range_data.get('prev_call_range'):
        for position, (block_header, range_str) in basic_ranges.items():
            print(f"\n{position} ranges:")
            print(f"Block: {block_header}")
            print(f"{position}: {range_str}")
    
    if not basic_ranges:
        print("No ranges found in preflop chart for this scenario")


def display_frequencies(frequencies: Dict[str, float]) -> None:
    """Display frequency array in a formatted way."""
    freq_strings = []
    for hand in sorted(frequencies.keys()):
        if frequencies[hand] == 1.0:
            freq_strings.append(hand)
        else:
            freq_strings.append(f"{hand}:{frequencies[hand]}")
    
    # Print in lines of reasonable length
    line = ""
    for freq_str in freq_strings:
        if len(line + freq_str + ", ") > 80:
            print(line.rstrip(", "))
            line = freq_str + ", "
        else:
            line += freq_str + ", "
    if line:
        print(line.rstrip(", "))


if __name__ == "__main__":
    main() 