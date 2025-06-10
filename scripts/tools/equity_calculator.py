import eval7
import random
import os
import sys
import re
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Add solver directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver'))

# Import preflop scenario functions
from preflop_scenario_generator import (
    get_ranges_with_frequencies,
    parse_scenario,
    determine_ip_oop,
    POSITION_GROUPS
)

# Ranges
UTG_RANGE = "66+, A3s+, K9s+, QTs+, JTs, T9s, 65s, AJo+, KQo"
BB_RANGE = "22-JJ, AQs, AJs, K4s+, Q9s+, J8s+, T7s+, 97s+, 86s+, 75s+, 64s+, 53s+, 43s, AJo, AQo, KJo+, JTo"

def fix_range_notation(range_str: str) -> str:
    """Fix range notation to be compatible with eval7.
    
    Converts notations like:
    - "JJ-" to "22-JJ"
    - "AJo-AQo" to "AJo,AQo"
    - "AQs-" to "A2s-AQs"
    - "A3s-" to "A2s-A3s"
    - "KTs-" to "K2s-KTs"
    - "AJo-" to "A2o-AJo"
    """
    if not range_str:
        return range_str
    
    # Split by comma and process each part
    parts = []
    for part in range_str.split(','):
        part = part.strip()
        
        # Handle "XX-" notation (like "JJ-")
        if re.match(r'^[AKQJT2-9]{2}-$', part):
            rank = part[0]
            # Map rank to all pairs below
            rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            idx = rank_order.index(rank)
            pairs = [f"{r}{r}" for r in rank_order[:idx+1]]
            parts.append("22-" + part[:-1])  # Convert "JJ-" to "22-JJ"
            
        # Handle "AXo-AYo" notation (like "AJo-AQo")
        elif re.match(r'^[AKQJT2-9]{2}[os]-[AKQJT2-9]{2}[os]$', part):
            # Split into individual hands
            start, end = part.split('-')
            rank1_start, rank2_start, suit_start = start[0], start[1], start[2]
            rank1_end, rank2_end, suit_end = end[0], end[1], end[2]
            
            if rank1_start == rank1_end and suit_start == suit_end:
                # Same first rank and suit type, enumerate second ranks
                rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
                start_idx = rank_order.index(rank2_start)
                end_idx = rank_order.index(rank2_end)
                
                if start_idx <= end_idx:
                    hands = [f"{rank1_start}{r}{suit_start}" for r in rank_order[start_idx:end_idx+1]]
                else:
                    hands = [f"{rank1_start}{r}{suit_start}" for r in rank_order[end_idx:start_idx+1]]
                parts.extend(hands)
            else:
                # Can't handle this format, keep as is
                parts.append(part)
                
        # Handle "XYs-" notation (like "AQs-", "KJs-", "A3s-")
        elif re.match(r'^[AKQJT2-9]{2}s-$', part):
            # Extract the two ranks
            high_rank = part[0]
            low_rank = part[1]
            
            # For ace-X suited, convert to A2s-AXs
            if high_rank == 'A':
                parts.append(f"A2s-A{low_rank}s")
            # For king-X suited, convert to K2s-KXs
            elif high_rank == 'K':
                parts.append(f"K2s-K{low_rank}s")
            # For queen-X suited, convert to Q2s-QXs
            elif high_rank == 'Q':
                parts.append(f"Q2s-Q{low_rank}s")
            # For jack-X suited, convert to J2s-JXs
            elif high_rank == 'J':
                parts.append(f"J2s-J{low_rank}s")
            # For ten-X suited, convert to T2s-TXs
            elif high_rank == 'T':
                parts.append(f"T2s-T{low_rank}s")
            else:
                # For other high cards, just keep as is for now
                parts.append(part)
        
        # Handle "XYo-" notation (like "AJo-", "KTo-")
        elif re.match(r'^[AKQJT2-9]{2}o-$', part):
            # Extract the two ranks
            high_rank = part[0]
            low_rank = part[1]
            
            # For ace-X offsuit, convert to A2o-AXo
            if high_rank == 'A':
                parts.append(f"A2o-A{low_rank}o")
            # For king-X offsuit, convert to K2o-KXo
            elif high_rank == 'K':
                parts.append(f"K2o-K{low_rank}o")
            # For queen-X offsuit, convert to Q2o-QXo
            elif high_rank == 'Q':
                parts.append(f"Q2o-Q{low_rank}o")
            # For jack-X offsuit, convert to J2o-JXo
            elif high_rank == 'J':
                parts.append(f"J2o-J{low_rank}o")
            # For ten-X offsuit, convert to T2o-TXo
            elif high_rank == 'T':
                parts.append(f"T2o-T{low_rank}o")
            else:
                # For other high cards, just keep as is for now
                parts.append(part)
                
        else:
            # Keep as is
            parts.append(part)
    
    return ', '.join(parts)

def parse_flop_input(flop_input: str) -> List[str]:
    """Parse flop input, handling both space-separated and continuous formats.
    
    Parameters
    ----------
    flop_input : str
        Either "As 8c 4d" or "As8c4d" format
        
    Returns
    -------
    List[str]
        List of card strings
    """
    flop_input = flop_input.strip()
    
    # If contains spaces, split by space
    if ' ' in flop_input:
        return flop_input.split()
    
    # Otherwise, split every 2 characters
    cards = []
    i = 0
    while i < len(flop_input):
        if i + 1 < len(flop_input):
            cards.append(flop_input[i:i+2])
            i += 2
        else:
            # Incomplete card
            raise ValueError(f"Invalid flop format: {flop_input}")
    
    # Validate we have exactly 3 cards
    if len(cards) != 3:
        raise ValueError(f"Flop must contain exactly 3 cards, got {len(cards)}")
    
    # Validate card format
    valid_ranks = set('23456789TJQKA')
    valid_suits = set('hdcs')
    
    for card in cards:
        if len(card) != 2 or card[0] not in valid_ranks or card[1] not in valid_suits:
            raise ValueError(f"Invalid card: {card}")
    
    return cards

def parse(rng):
    """Parse a range string into list of hand combinations."""
    # Strip brackets if present
    if rng.startswith('[') and rng.endswith(']'):
        rng = rng[1:-1]
    
    # Fix range notation before parsing
    fixed_range = fix_range_notation(rng)
    try:
        return [cards for cards, weight in eval7.HandRange(fixed_range)]
    except Exception as e:
        print(f"Error parsing range: {rng}")
        print(f"Fixed range: {fixed_range}")
        raise

def hand_to_string(hand, show_suits=True):
    """Convert a hand tuple to readable string like 'AhKh' or 'AKs' or '4h4s'"""
    c1, c2 = hand
    
    # Map rank integers to characters
    rank_map = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 
                7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    
    # Map suit integers to characters
    suit_map = {0: 'h', 1: 'd', 2: 'c', 3: 's'}
    
    r1 = rank_map[c1.rank]
    r2 = rank_map[c2.rank]
    s1 = suit_map[c1.suit]
    s2 = suit_map[c2.suit]
    
    if show_suits:
        # Always show full card notation when show_suits is True
        # Order by rank first, then by suit for consistent display
        if c1.rank > c2.rank or (c1.rank == c2.rank and c1.suit > c2.suit):
            return f"{r1}{s1}{r2}{s2}"
        else:
            return f"{r2}{s2}{r1}{s1}"
    else:
        # Show generic notation (AKs/AKo/QQ)
        if r1 == r2:
            return f"{r1}{r2}"
        else:
            suited = "s" if c1.suit == c2.suit else "o"
            if c1.rank > c2.rank:
                return f"{r1}{r2}{suited}"
            else:
                return f"{r2}{r1}{suited}"

def get_ranges_from_scenario(scenario: str) -> Dict[str, str]:
    """Extract ranges from a preflop scenario.
    
    Parameters
    ----------
    scenario : str
        The preflop action string (e.g., "UTG raise, BB call")
        
    Returns
    -------
    Dict[str, str]
        Dictionary with two entries - one for each player's range
    """
    # Get range data from the preflop scenario
    range_data = get_ranges_with_frequencies(scenario)
    
    # Parse the scenario to get positions
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        raise ValueError("Scenario must have at least 2 actions")
    
    # Extract the two main players
    acting_pos = range_data.get('acting_position', '')
    prev_acting_pos = range_data.get('prev_acting_position', '')
    
    # If positions not found in range_data, get from actions
    if not acting_pos or not prev_acting_pos:
        if len(actions) >= 2:
            acting_pos = actions[-1][0]  # Last action position
            prev_acting_pos = actions[-2][0]  # Second to last action position
    
    if not acting_pos or not prev_acting_pos:
        raise ValueError("Could not determine player positions")
    
    # Get ranges from basic_ranges if available
    basic_ranges = range_data.get('basic_ranges', {})
    
    # Initialize ranges
    ranges = {}
    
    # For acting position (last player to act)
    if acting_pos in basic_ranges:
        _, range_str = basic_ranges[acting_pos]
        ranges[acting_pos] = range_str if range_str else f"ERROR: Empty range for {acting_pos}"
    else:
        # Try to get from frequencies if available
        frequencies = range_data.get('frequencies', {})
        if frequencies:
            ranges[acting_pos] = format_range_from_frequencies(frequencies)
        else:
            # Fallback to reasonable defaults based on position and action
            ranges[acting_pos] = get_default_range(acting_pos, scenario, "call")
    
    # For previous acting position
    if prev_acting_pos in basic_ranges:
        _, range_str = basic_ranges[prev_acting_pos]
        ranges[prev_acting_pos] = range_str if range_str else f"ERROR: Empty range for {prev_acting_pos}"
    else:
        # Try to get from prev_frequencies if available
        prev_frequencies = range_data.get('prev_frequencies', {})
        if prev_frequencies:
            ranges[prev_acting_pos] = format_range_from_frequencies(prev_frequencies)
        else:
            # Fallback to reasonable defaults based on position and action
            ranges[prev_acting_pos] = get_default_range(prev_acting_pos, scenario, "raise")
    
    return ranges

def get_default_range(position: str, scenario: str, action_type: str) -> str:
    """Get a reasonable default range based on position and action type."""
    
    # Define reasonable default ranges for different positions and actions
    opening_ranges = {
        'UTG': '77+, ATs+, KQs, AJo+, KQo',
        'UTG+1': '66+, A9s+, KQs, QJs, AJo+, KQo',
        'LJ': '55+, A8s+, KJs+, QJs+, J9s+, T9s, ATo+, KJo+',
        'HJ': '44+, A7s+, KTs+, QTs+, J9s+, T9s, 98s, ATo+, KJo+, QJo',
        'CO': '22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 87s, 76s, A8o+, K9o+, Q9o+, J9o+',
        'BTN': '22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 86s+, 75s+, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o',
        'SB': '22+, A2s+, K2s+, Q2s+, J5s+, T6s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K5o+, Q7o+, J7o+, T7o+, 97o+',
        'BB': '77+, A9s+, KQs, QJs, AJo+, KQo'  # Calling vs. single raise
    }
    
    calling_ranges = {
        'UTG': 'JJ+, AKs, AKo',
        'UTG+1': 'TT+, AQs+, AKo',
        'LJ': '99+, AJs+, KQs, AQo+',
        'HJ': '88+, ATs+, KQs, AJo+',
        'CO': '77+, A9s+, KJs+, QJs, AJo+, KQo',
        'BTN': '55+, A7s+, KTs+, QTs+, J9s+, T9s, ATo+, KJo+, QJo',
        'SB': '66+, A8s+, KQs, QJs, AJo+, KQo',
        'BB': '22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, 65s, A5o+, K9o+, Q9o+, J9o+, T9o'
    }
    
    # Determine which range to use
    if action_type == "call":
        return calling_ranges.get(position, opening_ranges.get(position, '77+, ATs+, KQs, AJo+, KQo'))
    else:  # opening/raising
        return opening_ranges.get(position, '77+, ATs+, KQs, AJo+, KQo')

def format_range_from_frequencies(frequencies: Dict[str, float]) -> str:
    """Convert frequency dictionary to range string."""
    # Filter hands with frequency > 0
    hands = [hand for hand, freq in frequencies.items() if freq > 0]
    
    if not hands:
        return "ERROR: No hands found with frequency > 0"
    
    # Join hands with comma
    return ", ".join(sorted(hands))

def calculate_range_equity(hero_range_str: str, villain_range_str: str, board_cards: List[eval7.Card], samples: int = 10000) -> float:
    """Calculate equity for range vs range.
    
    Parameters
    ----------
    hero_range_str : str
        Hero's range string
    villain_range_str : str
        Villain's range string
    board_cards : List[eval7.Card]
        Board cards (3 for flop, 4 for turn, 5 for river)
    samples : int
        Number of samples for Monte Carlo simulation
        
    Returns
    -------
    float
        Hero's equity as a fraction (0-1)
    """
    # Parse ranges
    hero_combos = parse(hero_range_str)
    villain_combos = parse(villain_range_str)
    
    if not hero_combos or not villain_combos:
        raise ValueError("Invalid ranges provided")
    
    deck = eval7.Deck()
    wins = ties = 0
    
    for _ in range(samples):
        # Sample hero hand
        while True:
            h = random.choice(hero_combos)
            if not any(c in board_cards for c in h):
                break

        # Sample villain hand
        attempts = 0
        while attempts < 100:
            v = random.choice(villain_combos)
            if not (set(v) & (set(h) | set(board_cards))):
                break
            attempts += 1
        else:
            continue  # Skip if can't find valid villain hand
        
        # Get remaining cards
        dead = set(h) | set(v) | set(board_cards)
        remaining = [c for c in deck.cards if c not in dead]
        
        # Sample turn/river if needed
        cards_needed = 5 - len(board_cards)
        if cards_needed > 0:
            if len(remaining) < cards_needed:
                continue
            additional = random.sample(remaining, cards_needed)
            full_board = list(board_cards) + additional
        else:
            full_board = list(board_cards)
        
        # Evaluate
        h_score = eval7.evaluate(list(h) + full_board)
        v_score = eval7.evaluate(list(v) + full_board)
        
        if h_score > v_score:  # Higher is better
            wins += 1
        elif h_score == v_score:
            ties += 1
    
    return (wins + 0.5 * ties) / samples

def calculate_hand_equity(hero_hand, villain_combos, board_cards, samples=1000):
    """Calculate equity for a specific hero hand against villain range"""
    dead = set(hero_hand + tuple(board_cards))
    deck = eval7.Deck()
    
    wins = ties = 0
    valid_samples = 0
    
    for _ in range(samples):
        # Try to find valid villain hand
        attempts = 0
        while attempts < 100:
            v = random.choice(villain_combos)
            if not (set(v) & dead):
                break
            attempts += 1
        else:
            continue
            
        valid_samples += 1
        
        # Get remaining cards for turn/river
        remaining = [c for c in deck.cards if c not in list(hero_hand) + list(v) + board_cards]
        if len(remaining) < 2:
            continue
            
        turn, river = random.sample(remaining, 2)

        # Evaluate
        h_score = eval7.evaluate(list(hero_hand) + board_cards + [turn, river])
        v_score = eval7.evaluate(list(v) + board_cards + [turn, river])

        if h_score > v_score:  # Higher is better
            wins += 1
        elif h_score == v_score:
            ties += 1

    if valid_samples == 0:
        return 0.0
    
    return (wins + 0.5 * ties) / valid_samples

def analyze_range_equity_streamlined(pos1: str, range1: str, pos2: str, range2: str, board_str: str, samples_per_hand: int = 300):
    """Streamlined analyze equity with minimal output and optional detailed breakdown"""
    print(f"\n{'='*70}")
    print(f"Board: {board_str}")
    print(f"{pos1}: {range1[:60]}..." if len(range1) > 60 else f"{pos1}: {range1}")
    print(f"{pos2}: {range2[:60]}..." if len(range2) > 60 else f"{pos2}: {range2}")
    print(f"{'='*70}")
    
    # Parse board - handle both formats
    board_cards = []
    if ' ' in board_str:
        board_cards = [eval7.Card(c) for c in board_str.split()]
    else:
        # Parse continuous format
        parsed_cards = parse_flop_input(board_str)
        board_cards = [eval7.Card(c) for c in parsed_cards]
    
    # Get ranges
    pos1_combos = parse(range1)
    pos2_combos = parse(range2)
    
    # Filter out hands that conflict with the board
    valid_pos1_combos = [hand for hand in pos1_combos if not any(c in board_cards for c in hand)]
    valid_pos2_combos = [hand for hand in pos2_combos if not any(c in board_cards for c in hand)]
    
    # Calculate overall equity first
    print("\nCalculating overall equity...", end="", flush=True)
    
    # Create range strings from valid combos
    valid_pos1_hands = [hand_to_string(hand, show_suits=False) for hand in valid_pos1_combos]
    valid_pos2_hands = [hand_to_string(hand, show_suits=False) for hand in valid_pos2_combos]
    
    # Remove duplicates and join
    valid_pos1_range = ', '.join(sorted(set(valid_pos1_hands)))
    valid_pos2_range = ', '.join(sorted(set(valid_pos2_hands)))
    
    overall_equity = calculate_range_equity(valid_pos1_range, valid_pos2_range, board_cards, samples=5000)
    print(" Done!")
    
    # Calculate individual hand equities
    print(f"Calculating {pos1} individual hand equities...", end="", flush=True)
    pos1_hand_equities = {}
    
    for i, hero_hand in enumerate(valid_pos1_combos):
        equity = calculate_hand_equity(hero_hand, valid_pos2_combos, board_cards, samples_per_hand)
        hand_str = hand_to_string(hero_hand, show_suits=True)
        pos1_hand_equities[hand_str] = equity
        
        if (i + 1) % 30 == 0:
            print(".", end="", flush=True)
    
    print(" Done!")
    
    print(f"Calculating {pos2} individual hand equities...", end="", flush=True)
    pos2_hand_equities = {}
    
    for i, hero_hand in enumerate(valid_pos2_combos):
        equity = calculate_hand_equity(hero_hand, valid_pos1_combos, board_cards, samples_per_hand)
        hand_str = hand_to_string(hero_hand, show_suits=True)
        pos2_hand_equities[hand_str] = equity
        
        if (i + 1) % 30 == 0:
            print(".", end="", flush=True)
    
    print(" Done!")
    
    # Display streamlined summary
    display_streamlined_summary(pos1, pos1_hand_equities, pos2, pos2_hand_equities, overall_equity)
    
    # Ask if user wants detailed breakdown
    show_details = input("\nShow detailed hand breakdown? (y/n) [n]: ").strip().lower()
    
    if show_details == 'y':
        # Display results for both positions
        display_hand_breakdown(pos1, pos1_hand_equities)
        display_hand_breakdown(pos2, pos2_hand_equities)

def display_streamlined_summary(pos1: str, pos1_equities: Dict[str, float], 
                               pos2: str, pos2_equities: Dict[str, float], overall_equity: float):
    """Display streamlined summary with key statistics"""
    
    # Calculate summary statistics
    pos1_avg = sum(pos1_equities.values()) / len(pos1_equities) if pos1_equities else 0
    pos2_avg = sum(pos2_equities.values()) / len(pos2_equities) if pos2_equities else 0
    
    # Count hands with >=82.5% equity (nut hands)
    pos1_nuts = sum(1 for eq in pos1_equities.values() if eq >= 0.825)
    pos2_nuts = sum(1 for eq in pos2_equities.values() if eq >= 0.825)
    
    # Display position summaries
    print(f"\n{'POSITION SUMMARIES':^70}")
    print("-" * 70)
    print(f"{pos1}: Average equity: {pos1_avg:.1%}, Hands ≥82.5%: {pos1_nuts}")
    print(f"{pos2}: Average equity: {pos2_avg:.1%}, Hands ≥82.5%: {pos2_nuts}")
    
    # Calculate and display advantages
    print(f"\n{'ADVANTAGES':^70}")
    print("-" * 70)
    
    # Range advantage
    range_margin = pos1_avg - pos2_avg
    range_margin_pct = abs(range_margin) * 100  # Convert to percentage points
    
    # Classify range advantage
    if range_margin_pct < 1.0:  # Less than 1% difference
        range_class = ""
        print(f"Range advantage: Neither player has significant advantage ({range_margin:+.1%})")
    else:
        if range_margin_pct < 10.0:
            range_class = " (small)"
        elif range_margin_pct < 20.0:
            range_class = " (medium)"
        else:
            range_class = " (big)"
            
        if range_margin > 0:
            print(f"Range advantage: {pos1} (+{range_margin:.1%}){range_class}")
        else:
            print(f"Range advantage: {pos2} (+{abs(range_margin):.1%}){range_class}")
    
    # Nut advantage
    pos1_nuts_pct = pos1_nuts / len(pos1_equities) * 100 if pos1_equities else 0
    pos2_nuts_pct = pos2_nuts / len(pos2_equities) * 100 if pos2_equities else 0
    nut_margin = pos1_nuts_pct - pos2_nuts_pct
    nut_margin_abs = abs(nut_margin)
    
    # Classify nut advantage
    if nut_margin_abs < 1.0:  # Less than 1% difference
        nut_class = ""
        print(f"Nut advantage: Neither player has significant advantage ({nut_margin:+.1f}%)")
    else:
        if nut_margin_abs < 7.0:
            nut_class = " (small)"
        elif nut_margin_abs < 14.0:
            nut_class = " (medium)"
        else:
            nut_class = " (big)"
            
        if nut_margin > 0:
            print(f"Nut advantage: {pos1} (+{nut_margin:.1f}% more nut hands){nut_class}")
        else:
            print(f"Nut advantage: {pos2} (+{nut_margin_abs:.1f}% more nut hands){nut_class}")
    
    print("-" * 70)

def analyze_range_equity(pos1: str, range1: str, pos2: str, range2: str, board_str: str, samples_per_hand: int = 300):
    """Analyze equity for each hand in one range against another range"""
    print(f"\n{'='*70}")
    print(f"Board: {board_str}")
    print(f"{pos1}: {range1[:60]}..." if len(range1) > 60 else f"{pos1}: {range1}")
    print(f"{pos2}: {range2[:60]}..." if len(range2) > 60 else f"{pos2}: {range2}")
    print(f"{'='*70}")
    
    # Parse board - handle both formats
    board_cards = []
    if ' ' in board_str:
        board_cards = [eval7.Card(c) for c in board_str.split()]
    else:
        # Parse continuous format
        parsed_cards = parse_flop_input(board_str)
        board_cards = [eval7.Card(c) for c in parsed_cards]
    
    # Get ranges
    pos1_combos = parse(range1)
    pos2_combos = parse(range2)
    
    # Filter out hands that conflict with the board BEFORE calculating overall equity
    valid_pos1_combos = [hand for hand in pos1_combos if not any(c in board_cards for c in hand)]
    valid_pos2_combos = [hand for hand in pos2_combos if not any(c in board_cards for c in hand)]
    
    # Calculate overall equity first using only valid combos
    print("\nCalculating overall equity...", end="", flush=True)
    
    # Create range strings from valid combos
    valid_pos1_hands = [hand_to_string(hand, show_suits=False) for hand in valid_pos1_combos]
    valid_pos2_hands = [hand_to_string(hand, show_suits=False) for hand in valid_pos2_combos]
    
    # Remove duplicates and join
    valid_pos1_range = ', '.join(sorted(set(valid_pos1_hands)))
    valid_pos2_range = ', '.join(sorted(set(valid_pos2_hands)))
    
    overall_equity = calculate_range_equity(valid_pos1_range, valid_pos2_range, board_cards, samples=5000)
    print(" Done!")
    
    # Display overall equity for both players
    print(f"\n{'OVERALL EQUITY':^70}")
    print("-" * 70)
    print(f"{pos1}: {overall_equity:.1%}")
    print(f"{pos2}: {(1 - overall_equity):.1%}")
    print("-" * 70)
    
    # Ask if user wants detailed breakdown
    show_details = input("\nShow detailed hand breakdown? (y/n) [n]: ").strip().lower()
    
    if show_details == 'y':
        # Calculate equities for pos1 hands (using already filtered valid combos)
        print(f"\nCalculating {pos1} individual hand equities...", end="", flush=True)
        pos1_hand_equities = {}
        
        for i, hero_hand in enumerate(valid_pos1_combos):
            equity = calculate_hand_equity(hero_hand, valid_pos2_combos, board_cards, samples_per_hand)
            # Use specific combo as key (showing suits)
            hand_str = hand_to_string(hero_hand, show_suits=True)
            pos1_hand_equities[hand_str] = equity
            
            if (i + 1) % 30 == 0:
                print(".", end="", flush=True)
        
        print(" Done!")
        
        # Calculate equities for pos2 hands (using already filtered valid combos)
        print(f"\nCalculating {pos2} individual hand equities...", end="", flush=True)
        pos2_hand_equities = {}
        
        for i, hero_hand in enumerate(valid_pos2_combos):
            equity = calculate_hand_equity(hero_hand, valid_pos1_combos, board_cards, samples_per_hand)
            # Use specific combo as key (showing suits)
            hand_str = hand_to_string(hero_hand, show_suits=True)
            pos2_hand_equities[hand_str] = equity
            
            if (i + 1) % 30 == 0:
                print(".", end="", flush=True)
        
        print(" Done!")
        
        # Display results for pos1
        display_hand_breakdown(pos1, pos1_hand_equities)
        
        # Display results for pos2
        display_hand_breakdown(pos2, pos2_hand_equities)
        
        # Calculate range advantage and nut advantage
        advantages = calculate_advantages(pos1, pos1_hand_equities, pos2, pos2_hand_equities)
        display_advantages(advantages, pos1, pos2)

def display_hand_breakdown(position: str, hand_equities: Dict[str, float]):
    """Display hand breakdown for a position"""
    # Sort hands by equity
    sorted_hands = sorted(hand_equities.items(), key=lambda x: x[1], reverse=True)
    
    # Display results in a nice table format
    print(f"\n{position} HAND BREAKDOWN:")
    print("{:<10} {:<8} {:<50}".format("Hand", "Equity", "Visual"))
    print("-" * 70)
    
    for hand, equity in sorted_hands:
        # Create visual bar
        bar_length = int(equity * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        # Color coding
        if equity >= 0.70:
            status = "★★★"  # Strong
        elif equity >= 0.55:
            status = "★★"   # Good
        elif equity >= 0.45:
            status = "★"    # Marginal
        else:
            status = ""     # Weak
        
        print(f"{hand:<10} {equity:>6.1%}  {bar} {status}")
    
    # Summary statistics
    all_equities = list(hand_equities.values())
    if all_equities:
        avg_total = sum(all_equities) / len(all_equities)
        
        print(f"\n{position} Summary:")
        print("-" * 40)
        print(f"Average equity: {avg_total:.1%}")
        print(f"Best hand: {sorted_hands[0][0]} ({sorted_hands[0][1]:.1%})")
        print(f"Worst hand: {sorted_hands[-1][0]} ({sorted_hands[-1][1]:.1%})")
        print(f"Hands above 60%: {sum(1 for _, eq in sorted_hands if eq >= 0.60)}")
        print(f"Hands below 40%: {sum(1 for _, eq in sorted_hands if eq < 0.40)}")

def calculate_advantages(pos1: str, pos1_equities: Dict[str, float], 
                        pos2: str, pos2_equities: Dict[str, float]) -> Dict[str, Any]:
    """Calculate range advantage and nut advantage.
    
    Parameters
    ----------
    pos1, pos2 : str
        Position names
    pos1_equities, pos2_equities : Dict[str, float]
        Hand-to-equity mappings for each player
        
    Returns
    -------
    Dict[str, Any]
        Contains range_advantage and nut_advantage data
    """
    # Calculate average equities (range advantage)
    pos1_avg = sum(pos1_equities.values()) / len(pos1_equities) if pos1_equities else 0
    pos2_avg = sum(pos2_equities.values()) / len(pos2_equities) if pos2_equities else 0
    
    # Define nut thresholds
    strong_threshold = 0.70  # 70%+ is considered strong
    nut_threshold = 0.825    # 82.5%+ is considered nuts
    
    # Count hands in each category for pos1
    pos1_strong = sum(1 for eq in pos1_equities.values() if eq >= strong_threshold)
    pos1_nuts = sum(1 for eq in pos1_equities.values() if eq >= nut_threshold)
    pos1_strong_pct = pos1_strong / len(pos1_equities) * 100 if pos1_equities else 0
    pos1_nuts_pct = pos1_nuts / len(pos1_equities) * 100 if pos1_equities else 0
    
    # Count hands in each category for pos2
    pos2_strong = sum(1 for eq in pos2_equities.values() if eq >= strong_threshold)
    pos2_nuts = sum(1 for eq in pos2_equities.values() if eq >= nut_threshold)
    pos2_strong_pct = pos2_strong / len(pos2_equities) * 100 if pos2_equities else 0
    pos2_nuts_pct = pos2_nuts / len(pos2_equities) * 100 if pos2_equities else 0
    
    # Calculate equity of top portions of range
    pos1_sorted = sorted(pos1_equities.values(), reverse=True)
    pos2_sorted = sorted(pos2_equities.values(), reverse=True)
    
    # Top 10% of range
    pos1_top10_count = max(1, len(pos1_sorted) // 10)
    pos2_top10_count = max(1, len(pos2_sorted) // 10)
    pos1_top10_avg = sum(pos1_sorted[:pos1_top10_count]) / pos1_top10_count if pos1_sorted else 0
    pos2_top10_avg = sum(pos2_sorted[:pos2_top10_count]) / pos2_top10_count if pos2_sorted else 0
    
    return {
        'range_advantage': {
            'averages': {pos1: pos1_avg, pos2: pos2_avg},
            'leader': pos1 if pos1_avg > pos2_avg else pos2,
            'margin': abs(pos1_avg - pos2_avg)
        },
        'nut_advantage': {
            'strong_hands': {
                pos1: {'count': pos1_strong, 'percentage': pos1_strong_pct},
                pos2: {'count': pos2_strong, 'percentage': pos2_strong_pct}
            },
            'nut_hands': {
                pos1: {'count': pos1_nuts, 'percentage': pos1_nuts_pct},
                pos2: {'count': pos2_nuts, 'percentage': pos2_nuts_pct}
            },
            'top_10_percent': {
                pos1: pos1_top10_avg,
                pos2: pos2_top10_avg
            },
            'leader': pos1 if pos1_nuts_pct > pos2_nuts_pct else pos2
        }
    }

def display_advantages(advantages: Dict[str, Any], pos1: str, pos2: str):
    """Display range and nut advantages in a clear format."""
    print(f"\n{'='*70}")
    print(f"{'RANGE & NUT ADVANTAGE ANALYSIS':^70}")
    print(f"{'='*70}")
    
    # Range Advantage
    range_adv = advantages['range_advantage']
    print(f"\n{'RANGE ADVANTAGE':^70}")
    print("-" * 70)
    print(f"{pos1} average equity: {range_adv['averages'][pos1]:.1%}")
    print(f"{pos2} average equity: {range_adv['averages'][pos2]:.1%}")
    print(f"\n→ {range_adv['leader']} has range advantage by {range_adv['margin']:.1%}")
    
    # Nut Advantage
    nut_adv = advantages['nut_advantage']
    print(f"\n{'NUT ADVANTAGE':^70}")
    print("-" * 70)
    
    # Strong hands (70%+)
    print(f"\nStrong hands (≥70% equity):")
    pos1_strong = nut_adv['strong_hands'][pos1]
    pos2_strong = nut_adv['strong_hands'][pos2]
    print(f"{pos1}: {pos1_strong['count']} hands ({pos1_strong['percentage']:.1f}% of range)")
    print(f"{pos2}: {pos2_strong['count']} hands ({pos2_strong['percentage']:.1f}% of range)")
    
    # Nut hands (82.5%+)
    print(f"\nNut hands (≥82.5% equity):")
    pos1_nuts = nut_adv['nut_hands'][pos1]
    pos2_nuts = nut_adv['nut_hands'][pos2]
    print(f"{pos1}: {pos1_nuts['count']} hands ({pos1_nuts['percentage']:.1f}% of range)")
    print(f"{pos2}: {pos2_nuts['count']} hands ({pos2_nuts['percentage']:.1f}% of range)")
    
    # Top 10% of range
    print(f"\nTop 10% of range average equity:")
    print(f"{pos1}: {nut_adv['top_10_percent'][pos1]:.1%}")
    print(f"{pos2}: {nut_adv['top_10_percent'][pos2]:.1%}")
    
    # Overall nut advantage
    if pos1_nuts['percentage'] > pos2_nuts['percentage']:
        margin = pos1_nuts['percentage'] - pos2_nuts['percentage']
        print(f"\n→ {pos1} has nut advantage ({margin:.1f}% more nut hands)")
    elif pos2_nuts['percentage'] > pos1_nuts['percentage']:
        margin = pos2_nuts['percentage'] - pos1_nuts['percentage']
        print(f"\n→ {pos2} has nut advantage ({margin:.1f}% more nut hands)")
    else:
        print(f"\n→ Neither player has clear nut advantage")
    
    print("=" * 70)

def main():
    """Main function for streamlined equity calculation."""
    print("\nPOKER EQUITY CALCULATOR")
    print("="*70)
    print("Enter preflop action (e.g., 'UTG raise, BB call') or 'q' to quit")
    print("="*70)
    
    while True:
        print("\nExample scenarios:")
        print("  - UTG raise, BB call")
        print("  - CO raise, BTN 3bet, CO call")
        print("  - BTN raise, SB 3bet, BTN 4bet, SB call")
        
        scenario = input("\nEnter preflop action (or 'q' to quit): ").strip()
        
        if scenario.lower() == 'q':
            print("\nGoodbye!")
            break
            
        try:
            # Get ranges from scenario
            ranges = get_ranges_from_scenario(scenario)
            positions = list(ranges.keys())
            
            if len(positions) != 2:
                print("Error: Could not extract exactly 2 player ranges")
                continue
            
            print(f"\nScenario: {scenario}")
            print(f"Ranges: {positions[0]} vs {positions[1]}")
            print("="*50)
            
            # Inner loop for multiple flops with same preflop scenario
            while True:
                print(f"\nEnter flop for '{scenario}' (e.g., 'As8c4d' or 'As 8c 4d')")
                print("Type 'back' to change preflop scenario, or 'q' to quit completely:")
                flop_input = input().strip()
                
                if flop_input.lower() == 'q':
                    print("\nGoodbye!")
                    return
                
                if flop_input.lower() == 'back':
                    print("\nReturning to preflop scenario selection...")
                    break
                
                # Parse flop
                try:
                    flop_cards = parse_flop_input(flop_input)
                    flop_str = ' '.join(flop_cards)
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
                
                                # Analyze equity with streamlined output
                try:
                    analyze_range_equity_streamlined(
                        positions[0], ranges[positions[0]], 
                        positions[1], ranges[positions[1]], 
                        flop_str
                    )
                except Exception as e:
                    print(f"Error analyzing equity: {e}")
            
        except Exception as e:
            print(f"Error: {e}")

# Example usage when imported
def calculate_equity(scenario: str, flop: str) -> Dict[str, Any]:
    """
    Calculate equity for a given preflop scenario and flop.
    
    Parameters
    ----------
    scenario : str
        Preflop action (e.g., "UTG raise, BB call")
    flop : str
        Flop cards (e.g., "As8c4d" or "As 8c 4d")
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'positions': List of position names
        - 'ranges': Dict of position to range string
        - 'flop': Flop string
        - 'equity': Dict of position to equity percentage
    """
    ranges = get_ranges_from_scenario(scenario)
    positions = list(ranges.keys())
    
    if len(positions) != 2:
        raise ValueError("Could not extract exactly 2 player ranges")
    
    # Parse flop
    flop_cards = parse_flop_input(flop)
    board_cards = [eval7.Card(c) for c in flop_cards]
    
    # Calculate overall equity
    equity = calculate_range_equity(
        ranges[positions[0]], 
        ranges[positions[1]], 
        board_cards
    )
    
    return {
        'positions': positions,
        'ranges': ranges,
        'flop': ' '.join(flop_cards),
        'equity': {
            positions[0]: round(equity * 100, 1),
            positions[1]: round((1 - equity) * 100, 1)
        }
    }

if __name__ == "__main__":
    main() 