#!/usr/bin/env python3
"""
Generate poker training data for System B with TOOL_TAGS.

This script creates training examples that include tool information such as
board texture, ranges, equity advantages, and hand categorization.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Import solver utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'solver'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from preflop_scenario_generator import (
    POSITION_GROUPS,
    determine_ip_oop,
    parse_scenario,
    get_ranges_with_frequencies
)
from solver_inputer import (
    RANK_VALUES,
    analyze_suits,
    analyze_pairing,
    analyze_hirank,
    analyze_connectivity
)

# Import equity calculator
from equity_calculator import (
    parse_flop_input,
    get_ranges_from_scenario,
    calculate_range_equity,
    calculate_hand_equity,
    parse as parse_range,
    hand_to_string
)

import eval7

# Complete list of positions including BB
PREFLOP_POSITIONS = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

# Preflop bet sizes
INITIAL_RAISE_SIZE = 2.5
IP_3BET_SIZE = 7.5  # 3x the raise
OOP_3BET_SIZE = 11
IP_4BET_SIZE = 25
OOP_4BET_SIZE = 22

# Position mapping for generalized positions
POSITION_MAPPING = {
    'EP': ['UTG', 'UTG+1'],
    'MP': ['LJ', 'HJ'],
    'IP': ['CO', 'BTN'],  # For 3bet/4bet scenarios
    'OOP': ['SB', 'BB']   # For 3bet/4bet scenarios
}

# Helper function to count examples in jsonl file
def count_jsonl_examples(filename: str) -> int:
    """Count the number of lines (examples) in a JSONL file."""
    if not os.path.exists(filename):
        return 0
    
    count = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Only count non-empty lines
                    count += 1
    except Exception as e:
        print(f"Warning: Could not count examples in {filename}: {e}")
        return 0
    
    return count

def get_existing_custom_ids(filename: str) -> set:
    """Extract all existing custom_ids from a JSONL file."""
    custom_ids = set()
    if not os.path.exists(filename):
        return custom_ids
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'custom_id' in data:
                            custom_ids.add(data['custom_id'])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not read custom_ids from {filename}: {e}")
    
    return custom_ids

def generate_unique_custom_id(solver_file: str, existing_ids: set, tag: str = '', session_ids: set = None) -> str:
    """Generate a unique custom_id based on solver filename, handling duplicates."""
    if session_ids is None:
        session_ids = set()
    
    # Extract base name from solver file (remove .json extension)
    base_name = solver_file.replace('.json', '')
    
    # Apply tag if provided
    if tag:
        base_id = f"{tag}_{base_name}"
    else:
        base_id = base_name
    
    # Combine existing and session IDs
    all_existing_ids = existing_ids | session_ids
    
    # Check if this ID already exists
    if base_id not in all_existing_ids:
        return base_id
    
    # Find next available number
    counter = 1
    while True:
        numbered_id = f"{base_id} ({counter})"
        if numbered_id not in all_existing_ids:
            return numbered_id
        counter += 1

def get_solver_files() -> List[str]:
    """Get all available solver output files."""
    solver_outputs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'solver', 'solver_outputs')
    return [f for f in os.listdir(solver_outputs_dir) if f.endswith('.json')]

def parse_filename(filename: str) -> Dict[str, str]:
    """Parse solver filename to extract scenario information.
    
    Returns dict with: flop, pos1, pos2, pot_type, suits, pairing, hirank, connectivity
    """
    base_name = filename.replace('.json', '')
    parts = base_name.split('_')
    
    if len(parts) >= 8:
        # New format with flop cards
        return {
            'flop': parts[0],
            'pos1': parts[1],  # Aggressor
            'pos2': parts[2],  # Defender
            'pot_type': parts[3],
            'suits': parts[4],
            'pairing': parts[5],
            'hirank': parts[6],
            'connectivity': parts[7]
        }
    else:
        # Old format without flop cards
        return {
            'flop': '',
            'pos1': parts[0] if len(parts) > 0 else '',
            'pos2': parts[1] if len(parts) > 1 else '',
            'pot_type': parts[2] if len(parts) > 2 else 'SRP',
            'suits': parts[3] if len(parts) > 3 else '',
            'pairing': parts[4] if len(parts) > 4 else '',
            'hirank': parts[5] if len(parts) > 5 else '',
            'connectivity': parts[6] if len(parts) > 6 else ''
        }

def expand_position(pos: str, scenario_context: Dict[str, str] = None) -> str:
    """Expand generalized position to specific position.
    
    For IP/OOP in 3bet/4bet pots, we need context to determine valid positions.
    """
    if pos in PREFLOP_POSITIONS:
        return pos
    
    if pos == 'EP':
        return random.choice(['UTG', 'UTG+1'])
    elif pos == 'MP':
        return random.choice(['LJ', 'HJ'])
    elif pos == 'IP':
        # For 3bet/4bet pots, IP is the player in position
        # This is typically CO or BTN
        return random.choice(['CO', 'BTN'])
    elif pos == 'OOP':
        # For 3bet/4bet pots, OOP could be:
        # - Blinds if they 3bet
        # - Original raiser if they're OOP vs 3better
        # We'll default to blinds
        return random.choice(['SB', 'BB'])
    
    return pos  # Fallback

def construct_preflop_history(pos1: str, pos2: str, pot_type: str) -> Tuple[str, str, str]:
    """Construct the full preflop action sequence.
    
    Returns: (preflop_history, hero_pos, villain_pos)
    """
    # Randomly choose who is hero
    if random.random() < 0.5:
        hero_pos = pos1
        villain_pos = pos2
    else:
        hero_pos = pos2
        villain_pos = pos1
    
    actions = []
    
    if pot_type == 'SRP':
        # Single raised pot: pos1 raises, pos2 calls
        raiser = pos1
        caller = pos2
        
        raiser_idx = PREFLOP_POSITIONS.index(raiser)
        caller_idx = PREFLOP_POSITIONS.index(caller)
        
        # Track who has acted
        acted = set()
        
        # Build complete action sequence
        for i, position in enumerate(PREFLOP_POSITIONS):
            if position == raiser:
                actions.append(f"{position} raises {INITIAL_RAISE_SIZE}bb")
                acted.add(position)
            elif position == caller:
                # If we haven't passed the raiser yet, skip (will handle later)
                if i < raiser_idx:
                    continue
                else:
                    actions.append(f"{position} calls {INITIAL_RAISE_SIZE}bb")
                    acted.add(position)
            else:
                # Check if this position should have acted
                if caller_idx < raiser_idx:  # Caller is in blinds
                    # This position folds if:
                    # 1. It comes before the raiser, OR
                    # 2. It comes after the raiser but before we wrap to blinds
                    if i < raiser_idx or i > caller_idx:
                        actions.append(f"{position} folds")
                        acted.add(position)
                else:  # Normal flow
                    # This position folds if it's between start and caller
                    if i < caller_idx:
                        actions.append(f"{position} folds")
                        acted.add(position)
        
        # Add remaining players who haven't acted yet (like blinds)
        for position in PREFLOP_POSITIONS:
            if position not in acted:
                actions.append(f"{position} folds")
    
    elif pot_type == '3bet':
        # 3bet pot: pos2 raises, pos1 3bets, pos2 calls
        initial_raiser = pos2
        threebetter = pos1
        
        raiser_idx = PREFLOP_POSITIONS.index(initial_raiser)
        threebetter_idx = PREFLOP_POSITIONS.index(threebetter)
        
        # Determine 3bet size based on position
        if threebetter in ['SB', 'BB']:
            bet_size = OOP_3BET_SIZE
        else:
            bet_size = IP_3BET_SIZE
        
        # Track who has acted
        acted = set()
        
        # Initial folds up to raiser
        for i in range(raiser_idx):
            actions.append(f"{PREFLOP_POSITIONS[i]} folds")
            acted.add(PREFLOP_POSITIONS[i])
        
        # Initial raise
        actions.append(f"{initial_raiser} raises {INITIAL_RAISE_SIZE}bb")
        acted.add(initial_raiser)
        
        # Folds between raiser and 3better
        if threebetter_idx > raiser_idx:
            # 3better comes after raiser in normal order
            for i in range(raiser_idx + 1, threebetter_idx):
                actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                acted.add(PREFLOP_POSITIONS[i])
        else:
            # 3better is in blinds - fold everyone after raiser first
            for i in range(raiser_idx + 1, len(PREFLOP_POSITIONS)):
                if PREFLOP_POSITIONS[i] != threebetter and PREFLOP_POSITIONS[i] not in acted:
                    actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                    acted.add(PREFLOP_POSITIONS[i])
        
        # 3bet
        actions.append(f"{threebetter} 3bets to {bet_size}bb")
        acted.add(threebetter)
        
        # Remaining folds (everyone who hasn't acted yet except the initial raiser)
        for position in PREFLOP_POSITIONS:
            if position not in acted and position != initial_raiser:
                actions.append(f"{position} folds")
        
        # Initial raiser calls
        actions.append(f"{initial_raiser} calls {bet_size}bb")
    
    elif pot_type == '4bet':
        # 4bet pot: pos1 raises, pos2 3bets, pos1 4bets, pos2 calls
        initial_raiser = pos1
        threebetter = pos2
        
        raiser_idx = PREFLOP_POSITIONS.index(initial_raiser)
        threebetter_idx = PREFLOP_POSITIONS.index(threebetter)
        
        # Determine sizes based on who is IP/OOP
        if initial_raiser in ['SB', 'BB']:
            # Raiser is OOP, threebetter is IP
            threebet_size = IP_3BET_SIZE    # 7.5bb
            fourbet_size = OOP_4BET_SIZE    # 22bb
        else:
            # Raiser is IP, threebetter is OOP
            threebet_size = OOP_3BET_SIZE   # 11bb
            fourbet_size = IP_4BET_SIZE     # 25bb
        
        # Build complete action sequence
        acted = set()
        
        # Special handling based on raiser position
        if initial_raiser in ['SB', 'BB']:
            # Blind raise scenario - action wraps around
            if initial_raiser == 'SB':
                # SB raises, BB acts next
                # Only fold positions before SB that are NOT the 3better
                for i in range(raiser_idx):
                    if PREFLOP_POSITIONS[i] != threebetter:
                        actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                        acted.add(PREFLOP_POSITIONS[i])
                
                actions.append(f"{initial_raiser} raises {INITIAL_RAISE_SIZE}bb")
                acted.add(initial_raiser)
                
                # BB acts next - if BB is 3better, 3bet; otherwise fold
                if threebetter == 'BB':
                    actions.append(f"BB 3bets to {threebet_size}bb")
                    acted.add("BB")
                else:
                    actions.append("BB folds")
                    acted.add("BB")
                    # Action goes back to threebetter
                    actions.append(f"{threebetter} 3bets to {threebet_size}bb")
                    acted.add(threebetter)
            else:
                # BB raise scenario  
                # Only fold positions before BB that are NOT the 3better
                for i in range(raiser_idx):
                    if PREFLOP_POSITIONS[i] != threebetter:
                        actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                        acted.add(PREFLOP_POSITIONS[i])
                
                actions.append(f"{initial_raiser} raises {INITIAL_RAISE_SIZE}bb")
                acted.add(initial_raiser)
                
                # Action goes to threebetter
                actions.append(f"{threebetter} 3bets to {threebet_size}bb")
                acted.add(threebetter)
        else:
            # Normal position raise
            for i in range(raiser_idx):
                actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                acted.add(PREFLOP_POSITIONS[i])
            
            actions.append(f"{initial_raiser} raises {INITIAL_RAISE_SIZE}bb")
            acted.add(initial_raiser)
            
            # Folds between raiser and 3better
            if threebetter_idx > raiser_idx:
                for i in range(raiser_idx + 1, threebetter_idx):
                    actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                    acted.add(PREFLOP_POSITIONS[i])
            else:
                # 3better is in blinds - fold everyone after raiser first
                for i in range(raiser_idx + 1, len(PREFLOP_POSITIONS)):
                    if PREFLOP_POSITIONS[i] != threebetter:
                        actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                        acted.add(PREFLOP_POSITIONS[i])
            
            # 3bet
            actions.append(f"{threebetter} 3bets to {threebet_size}bb")
            acted.add(threebetter)
        
        # Remaining folds (everyone who hasn't acted yet except the two main players)
        for position in PREFLOP_POSITIONS:
            if position not in acted and position != initial_raiser and position != threebetter:
                actions.append(f"{position} folds")
        
        # 4bet and call
        actions.append(f"{initial_raiser} 4bets to {fourbet_size}bb")
        actions.append(f"{threebetter} calls {fourbet_size}bb")
    
    preflop_history = ", ".join(actions)
    return preflop_history, hero_pos, villain_pos 

def determine_who_acts_first_postflop(pos1: str, pos2: str) -> Tuple[str, str]:
    """Determine who acts first postflop between two positions.
    
    Returns: (first_to_act, second_to_act)
    """
    postflop_order = ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"]
    
    pos1_idx = postflop_order.index(pos1) if pos1 in postflop_order else 999
    pos2_idx = postflop_order.index(pos2) if pos2 in postflop_order else 999
    
    if pos1_idx < pos2_idx:
        return pos1, pos2
    else:
        return pos2, pos1

def load_solver_data(filename: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Load solver JSON and corresponding input file.
    
    Returns: (json_data, input_data)
    """
    # Load JSON
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'solver', 'solver_outputs', filename)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load corresponding input file
    txt_filename = filename.replace('.json', '.txt')
    txt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'solver', 'solver_inputs', txt_filename)
    
    input_data = {}
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('set_pot'):
                    input_data['pot'] = float(line.split()[1])
                elif line.startswith('set_effective_stack'):
                    input_data['effective_stack'] = float(line.split()[1])
                elif line.startswith('set_board'):
                    input_data['board'] = line.split()[1]
                elif line.startswith('set_range_ip'):
                    input_data['range_ip'] = line.split()[1]
                elif line.startswith('set_range_oop'):
                    input_data['range_oop'] = line.split()[1]
    
    return json_data, input_data

def get_ranges_from_preflop_chart(scenario: str) -> Dict[str, str]:
    """Extract ranges from preflop chart for the given scenario."""
    # Load preflop chart
    chart_path = os.path.join(os.path.dirname(__file__), '..', '..', 'solver', 'preflop_chart.txt')
    if not os.path.exists(chart_path):
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
                chart[current_block]["ALL"] = line.strip()
    
    # Use the existing function to get ranges
    range_data = get_ranges_with_frequencies(scenario)
    basic_ranges = range_data.get('basic_ranges', {})
    
    result = {}
    for pos, (block_header, range_str) in basic_ranges.items():
        if range_str:
            result[pos] = range_str
    
    return result

def convert_board_texture(suits: str, pairing: str, hirank: str, connectivity: str) -> str:
    """Convert board texture abbreviations to full descriptions."""
    # Convert suits
    if suits == 'mono':
        suits_full = 'monotone'
    elif suits == 'tt':
        suits_full = 'two-tone'
    elif suits == 'rb':
        suits_full = 'rainbow'
    else:
        suits_full = suits
    
    # Convert pairing
    if pairing == 'np':
        pairing_full = 'unpaired'
    else:
        pairing_full = pairing  # Already descriptive (e.g., 'lowpair', 'trips')
    
    # Convert high rank
    if hirank == 'ah':
        hirank_full = 'ace-high'
    elif hirank == 'bh':
        hirank_full = 'broadway-high'
    elif hirank == 'mh':
        hirank_full = 'mid-high'
    elif hirank == 'low':
        hirank_full = 'low'
    else:
        hirank_full = hirank
    
    # Convert connectivity
    if connectivity == 'high':
        connectivity_full = 'connected'
    elif connectivity == 'semi':
        connectivity_full = 'semi-connected'
    elif connectivity == 'dry':
        connectivity_full = 'dry'
    else:
        connectivity_full = connectivity
    
    # Combine in order: tone, pairing, high rank, connectivity
    return f"{suits_full}, {pairing_full}, {hirank_full}, {connectivity_full}"

def analyze_hand_category(hero_cards: List[str], board_cards: List[str]) -> str:
    """Analyze hero's hand category based on board texture."""
    # Convert to eval7 cards for evaluation
    hero_eval = [eval7.Card(c) for c in hero_cards]
    board_eval = [eval7.Card(c) for c in board_cards]
    
    # Get hand strength
    hand_strength = eval7.evaluate(hero_eval + board_eval)
    hand_type = eval7.handtype(hand_strength)
    
    # Extract ranks and suits
    hero_ranks = [c[0] for c in hero_cards]
    hero_suits = [c[1] for c in hero_cards]
    board_ranks = [c[0] for c in board_cards]
    board_suits = [c[1] for c in board_cards]
    all_ranks = hero_ranks + board_ranks
    all_suits = hero_suits + board_suits
    
    # Count occurrences
    rank_counts = {}
    for rank in all_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    suit_counts = {}
    for suit in all_suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    # Check for specific hand types
    categories = []
    
    # 1. Quads
    if hand_type == 'Four of a Kind':
        return "quads"
    
    # 2. Full house
    if hand_type == 'Full House':
        return "full house"
    
    # 3. Flush/flush draws
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    if flush_suit:
        return "flush"
    
    # Check for flush draws
    hero_flush_draw = False
    backdoor_flush_draw = False
    for suit in ['h', 'd', 'c', 's']:
        hero_suit_count = sum(1 for s in hero_suits if s == suit)
        board_suit_count = sum(1 for s in board_suits if s == suit)
        total_suit = hero_suit_count + board_suit_count
        
        if total_suit == 4:
            hero_flush_draw = True
        elif total_suit == 3 and hero_suit_count >= 1:
            backdoor_flush_draw = True
    
    # 4. Straight/straight draws
    if hand_type == 'Straight':
        if backdoor_flush_draw:
            return "straight with backdoor flush draw"
        else:
            return "straight"
    
    # Check for straight draws
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    # Get rank values for hero cards and board
    hero_rank_vals = [rank_values[r] for r in hero_ranks]
    board_rank_vals = [rank_values[r] for r in board_ranks]
    all_rank_values = sorted(hero_rank_vals + board_rank_vals)
    unique_ranks = sorted(list(set(all_rank_values)))
    
    # Check for OESD, gutshot, and backdoor straight draws
    oesd = False
    gutshot = False
    backdoor_straight = False
    
    # Check for all possible 5-card straights
    for start in range(1, 11):  # Check all possible straights (A-5 through T-A)
        straight_vals = list(range(start, start + 5))
        # Handle A-5 straight (wheel)
        if start == 1:
            straight_vals = [14, 2, 3, 4, 5]
        
        # Count how many cards we have for this straight
        have = sum(1 for val in straight_vals if val in all_rank_values)
        # Count how many cards we have that include at least one hero card
        hero_contributes = any(val in hero_rank_vals for val in straight_vals if val in all_rank_values)
        
        # Check if this straight involves an Ace
        has_ace_in_straight = 14 in straight_vals and 14 in all_rank_values
        
        if have == 4 and hero_contributes:
            # Need 1 card for straight - check if it's OESD or gutshot
            missing = [val for val in straight_vals if val not in all_rank_values][0]
            
            # Check if we have 4 consecutive cards (OESD)
            temp_unique = sorted(list(set([v for v in all_rank_values if v in straight_vals])))
            is_connected = len(temp_unique) == 4 and temp_unique[-1] - temp_unique[0] == 3
            
            if is_connected and not has_ace_in_straight:
                # True OESD - 4 connected cards without Ace
                oesd = True
            else:
                # Either not connected or involves Ace - it's a gutshot
                gutshot = True
                
        elif have == 3 and hero_contributes:
            # Need 2 cards - check if it's backdoor
            # For backdoor straight, we need exactly 3 consecutive cards
            cards_in_straight = sorted([v for v in all_rank_values if v in straight_vals])
            if len(cards_in_straight) == 3:
                # Check if these 3 cards are consecutive
                if cards_in_straight[-1] - cards_in_straight[0] == 2 and not has_ace_in_straight:
                    # True backdoor straight - 3 consecutive cards without Ace
                    backdoor_straight = True
                # If has Ace, no backdoor straight draw (downgrades to nothing)
    
    # If we have straight draws, continue checking other categories
    if hero_flush_draw:
        categories.append("flush draw")
    elif backdoor_flush_draw:
        categories.append("backdoor flush draw")
    
    if oesd:
        categories.append("open ended straight draw")
    elif gutshot:
        categories.append("gutshot")
    elif backdoor_straight:
        categories.append("backdoor straight draw")
    
    # 5. Set/trips
    if hand_type == 'Three of a Kind':
        # Check if it's a set (pocket pair) or trips
        if hero_ranks[0] == hero_ranks[1]:
            # Pocket pair - it's a set
            if categories:
                return "set with " + " with ".join(categories)
            else:
                return "set"
        else:
            # It's trips - need to determine kicker
            kicker = None
            for rank in hero_ranks:
                if rank_counts[rank] == 1:
                    kicker = rank
                    break
            
            if kicker:
                kicker_val = rank_values[kicker]
                if kicker == 'A':
                    kicker_cat = "ace kicker"
                elif kicker_val >= 10:  # T, J, Q, K
                    kicker_cat = "broadway kicker"
                else:
                    kicker_cat = "low kicker"
                
                if categories:
                    return f"trips with {kicker_cat} with " + " with ".join(categories)
                else:
                    return f"trips with {kicker_cat}"
            else:
                if categories:
                    return "trips with " + " with ".join(categories)
                else:
                    return "trips"
    
    # 6. Two pair
    if hand_type == 'Two Pair':
        board_paired = len([r for r in board_ranks if board_ranks.count(r) >= 2]) > 0
        
        if board_paired and hero_ranks[0] == hero_ranks[1]:
            # This is actually just one pair (pocket pair on paired board)
            # Treat it as one pair based on pocket pair's rank
            # Get distinct board ranks for comparison
            distinct_board_ranks = []
            seen = set()
            for r in board_ranks:
                if r not in seen:
                    distinct_board_ranks.append(r)
                    seen.add(r)
            
            board_vals = sorted([rank_values[r] for r in distinct_board_ranks], reverse=True)
            pair_val = rank_values[hero_ranks[0]]
            
            # Determine pair position
            if pair_val > board_vals[0]:
                pair_pos = "overpair"
            elif pair_val == board_vals[0]:
                pair_pos = "top pair"
            elif len(board_vals) > 1 and pair_val == board_vals[1]:
                pair_pos = "2nd pair"
            else:
                pair_pos = "bottom pair"
            
            if categories:
                return f"{pair_pos} on a paired board with " + " with ".join(categories)
            else:
                return f"{pair_pos} on a paired board"
        elif board_paired:
            # One card makes two pair - has kicker
            kicker = None
            for rank in hero_ranks:
                if rank_counts[rank] == 1:
                    kicker = rank
                    break
            
            if kicker:
                kicker_val = rank_values[kicker]
                if kicker == 'A':
                    kicker_cat = "ace kicker"
                elif kicker_val >= 10:
                    kicker_cat = "broadway kicker"
                else:
                    kicker_cat = "low kicker"
                
                # Determine which pair we made
                board_vals = sorted([rank_values[r] for r in set(board_ranks)], reverse=True)
                our_pair_rank = None
                for rank in hero_ranks:
                    if rank in board_ranks:
                        our_pair_rank = rank
                        break
                
                if our_pair_rank:
                    our_pair_val = rank_values[our_pair_rank]
                    if our_pair_val == board_vals[0]:
                        pair_pos = "top pair"
                    elif len(board_vals) > 1 and our_pair_val == board_vals[1]:
                        pair_pos = "2nd pair"
                    else:
                        pair_pos = "bottom pair"
                else:
                    pair_pos = ""
                
                result = f"two pair with {kicker_cat} on a paired board"
                if pair_pos:
                    result += f" ({pair_pos})"
                
                if categories:
                    return result + " with " + " with ".join(categories)
                else:
                    return result
        else:
            # Unpaired board - both cards play, no kicker
            if categories:
                return "two pair with " + " with ".join(categories)
            else:
                return "two pair"
    
    # 7. One pair
    if hand_type == 'Pair':
        board_paired = len([r for r in board_ranks if board_ranks.count(r) >= 2]) > 0
        
        # Check if we have a pocket pair
        is_pocket_pair = (hero_ranks[0] == hero_ranks[1])
        
        # Find which card makes the pair
        pair_rank = None
        kicker_rank = None
        
        for rank in hero_ranks:
            if rank_counts[rank] >= 2:
                pair_rank = rank
            elif not is_pocket_pair:
                kicker_rank = rank
        
        # Determine pair position
        board_vals = sorted([rank_values[r] for r in set(board_ranks)], reverse=True)
        
        if pair_rank:
            pair_val = rank_values[pair_rank]
            
            # Check if it's overpair (pocket pair higher than board)
            if is_pocket_pair and pair_val > board_vals[0]:
                pair_pos = "overpair"
            elif pair_val == board_vals[0]:
                pair_pos = "top pair"
            elif len(board_vals) > 1 and pair_val == board_vals[1]:
                pair_pos = "2nd pair"  
            elif len(board_vals) > 2 and pair_val == board_vals[2]:
                pair_pos = "bottom pair"
            else:
                # For pocket pairs below board or between board cards
                if is_pocket_pair:
                    # Determine position based on where the pocket pair would rank
                    if len(board_vals) == 1:
                        # Only one board rank, pocket pair is below it
                        pair_pos = "2nd pair"
                    elif len(board_vals) == 2:
                        # Two board ranks
                        if pair_val > board_vals[1]:
                            pair_pos = "2nd pair"  # Between highest and second
                        else:
                            pair_pos = "bottom pair"  # Below both
                    else:
                        # Three board ranks  
                        if pair_val > board_vals[1]:
                            pair_pos = "2nd pair"  # Between highest and second
                        elif pair_val > board_vals[2]:
                            pair_pos = "bottom pair"  # Between second and third
                        else:
                            pair_pos = "bottom pair"  # Below all
                else:
                    pair_pos = "bottom pair"  # Default for made pairs
            
            # Only add kicker if not a pocket pair
            if kicker_rank and not is_pocket_pair:
                kicker_val = rank_values[kicker_rank]
                if kicker_rank == 'A':
                    kicker_cat = "ace kicker"
                elif kicker_val >= 10:
                    kicker_cat = "broadway kicker"
                else:
                    kicker_cat = "low kicker"
                
                result = f"{pair_pos} with {kicker_cat}"
            else:
                result = pair_pos
            
            if board_paired:
                result += " on a paired board"
            
            if categories:
                return result + " with " + " with ".join(categories)
            else:
                return result
    
    # 8. No pair (and draws without made hands)
    if hand_type == 'High Card':
        # Count overcards
        board_vals = sorted([rank_values[r] for r in board_ranks], reverse=True)
        highest_board = board_vals[0]
        
        overcards = 0
        has_ace = False
        for rank in hero_ranks:
            rank_val = rank_values[rank]
            if rank_val > highest_board:
                overcards += 1
            if rank == 'A':
                has_ace = True
        
        if overcards == 2:
            result = "two overcards"
        elif overcards == 1:
            result = "one overcard"
        else:
            result = "no overcards"
        
        if has_ace:
            result += " with ace-high"
        
        if categories:
            return result + " with " + " with ".join(categories)
        else:
            return result

    # Final fallback - should not reach here but handle gracefully  
    return "high card"

def calculate_equity_and_advantages(hero_pos: str, villain_pos: str, hero_range: str, 
                                   villain_range: str, board_str: str, hero_cards: List[str]) -> Dict[str, Any]:
    """Calculate equity advantages and hero's hand ranking."""
    # Parse board
    board_cards = [eval7.Card(c) for c in board_str.split()]
    
    # Calculate overall equity
    overall_equity = calculate_range_equity(hero_range, villain_range, board_cards, samples=5000)
    
    # Parse ranges for individual hand analysis
    hero_combos = parse_range(hero_range)
    villain_combos = parse_range(villain_range)
    
    # Filter out hands that conflict with board
    valid_hero_combos = [hand for hand in hero_combos if not any(c in board_cards for c in hand)]
    valid_villain_combos = [hand for hand in villain_combos if not any(c in board_cards for c in hand)]
    
    # Calculate hero hand equities
    hero_hand_equities = {}
    for hero_hand in valid_hero_combos:
        equity = calculate_hand_equity(hero_hand, valid_villain_combos, board_cards, samples=300)
        hand_str = hand_to_string(hero_hand, show_suits=True)
        hero_hand_equities[hand_str] = equity
    
    # Calculate villain hand equities  
    villain_hand_equities = {}
    for villain_hand in valid_villain_combos:
        equity = calculate_hand_equity(villain_hand, valid_hero_combos, board_cards, samples=300)
        hand_str = hand_to_string(villain_hand, show_suits=True)
        villain_hand_equities[hand_str] = equity
    
    # Calculate averages
    hero_avg = sum(hero_hand_equities.values()) / len(hero_hand_equities) if hero_hand_equities else 0
    villain_avg = sum(villain_hand_equities.values()) / len(villain_hand_equities) if villain_hand_equities else 0
    
    # Range advantage
    range_margin = hero_avg - villain_avg
    range_margin_pct = abs(range_margin) * 100
    
    if range_margin_pct < 1.0:
        range_adv_cat = "neutral"
        range_adv_desc = "neutral"
    else:
        if range_margin_pct < 10.0:
            size = "small"
        elif range_margin_pct < 20.0:
            size = "medium"
        else:
            size = "large"
        
        if range_margin > 0:
            range_adv_cat = f"{size} edge for hero"
            range_adv_desc = f"hero +{range_margin:.2f}"
        else:
            range_adv_cat = f"{size} edge for villain"
            range_adv_desc = f"villain +{abs(range_margin):.2f}"
    
    # Nut advantage
    hero_nuts = sum(1 for eq in hero_hand_equities.values() if eq >= 0.825)
    villain_nuts = sum(1 for eq in villain_hand_equities.values() if eq >= 0.825)
    
    hero_nuts_pct = hero_nuts / len(hero_hand_equities) * 100 if hero_hand_equities else 0
    villain_nuts_pct = villain_nuts / len(villain_hand_equities) * 100 if villain_hand_equities else 0
    nut_margin = hero_nuts_pct - villain_nuts_pct
    nut_margin_abs = abs(nut_margin)
    
    if nut_margin_abs < 1.0:
        nut_adv_cat = "neutral"
        nut_adv_desc = "neutral"
    else:
        if nut_margin_abs < 7.0:
            size = "small"
        elif nut_margin_abs < 14.0:
            size = "medium"
        else:
            size = "large"
        
        if nut_margin > 0:
            nut_adv_cat = f"{size} edge for hero"
            nut_adv_desc = f"hero +{nut_margin:.1f}%"
        else:
            nut_adv_cat = f"{size} edge for villain"
            nut_adv_desc = f"villain +{nut_margin_abs:.1f}%"
    
    # Calculate hero's specific hand equity and ranking
    hero_eval_cards = [eval7.Card(c) for c in hero_cards]
    hero_specific_equity = calculate_hand_equity(tuple(hero_eval_cards), valid_villain_combos, board_cards, samples=500)
    
    # Find hero's percentile
    sorted_equities = sorted(hero_hand_equities.values(), reverse=True)
    better_hands = sum(1 for eq in sorted_equities if eq > hero_specific_equity)
    percentile = (1 - better_hands / len(sorted_equities)) * 100 if sorted_equities else 50
    
    return {
        'range_adv_category': range_adv_cat,
        'range_adv_value': range_margin,
        'range_adv_desc': range_adv_desc,
        'nut_adv_category': nut_adv_cat,
        'nut_adv_value': nut_margin,
        'nut_adv_desc': nut_adv_desc,
        'hero_equity': hero_specific_equity,
        'hero_percentile': percentile
    }

def get_villain_action(json_data: Dict[str, Any], villain_hand: str, villain_pos: str, 
                       hero_pos: str) -> Tuple[str, float]:
    """Get villain's action for a specific hand from solver data - always choose highest frequency.
    
    Returns: (action, bet_size_in_bb)
    """
    # Determine who acts first postflop
    first_to_act, second_to_act = determine_who_acts_first_postflop(villain_pos, hero_pos)
    
    # If villain acts first, look at root level strategy
    if villain_pos == first_to_act:
        if 'actions' in json_data and 'strategy' in json_data and 'strategy' in json_data['strategy']:
            hand_strategies = json_data['strategy']['strategy']
            if villain_hand in hand_strategies:
                actions = json_data['actions']
                probs = hand_strategies[villain_hand]
                
                # Find the highest frequency action
                max_prob_idx = probs.index(max(probs))
                sampled_action = actions[max_prob_idx]
                
                # Parse action to get bet size
                if sampled_action == "CHECK":
                    return "CHECK", 0.0
                elif sampled_action.startswith("BET"):
                    # Extract bet size
                    bet_size = float(sampled_action.split()[1])
                    return "BET", bet_size
                else:
                    return sampled_action, 0.0
    
    # If villain is IP or no action found, return check
    return "CHECK", 0.0

def format_legal_actions(hero_pos: str, street: str, facing_action: str, pot_type: str = 'SRP', pot_size: float = 6.0) -> List[str]:
    """Format legal actions based on game state."""
    actions = []
    
    # Define bet formatting helper
    def format_bet(percentage: int) -> str:
        bet_bb = pot_size * percentage / 100
        return f"bet {percentage}% ({bet_bb:.1f}bb)"
    
    # Check/Call actions
    if facing_action in ["CHECK", "check", "first_to_act"]:
        actions.append("check")
        # Can bet
        actions.extend([
            format_bet(33),
            format_bet(50),
            format_bet(100)
        ])
    elif facing_action in ["BET", "bet"]:
        actions.append("fold")
        actions.append("call")
        # Can raise (simplified to one size for now)
        actions.append("raise 75%")
    else:
        # Default case
        actions.append("check")
        actions.extend([
            format_bet(33),
            format_bet(50),
            format_bet(100)
        ])
    
    # All-in is always an option
    actions.append("allin")
    
    return actions

def get_hero_strategy(json_data: Dict[str, Any], hero_hand: str, hero_pos: str, 
                      villain_pos: str, villain_action: str) -> Dict[str, float]:
    """Get hero's strategy for a specific hand given villain's action.
    
    Returns: Dict mapping actions to probabilities
    """
    # Determine who acts first postflop
    first_to_act, second_to_act = determine_who_acts_first_postflop(hero_pos, villain_pos)
    
    # If hero acts first and villain hasn't acted yet
    if hero_pos == first_to_act and villain_action == "first_to_act":
        # Hero's strategy is at the root level
        if 'strategy' in json_data and 'strategy' in json_data['strategy']:
            hand_strategies = json_data['strategy']['strategy']
            if hero_hand in hand_strategies:
                actions = json_data['actions']
                probs = hand_strategies[hero_hand]
                return dict(zip(actions, probs))
        return {}
    
    # If villain acted first, we need to navigate to their action node to get hero's response
    if villain_pos == first_to_act:
        current_node = json_data
        
        # Navigate to villain's action
        if villain_action == "CHECK":
            action_key = "CHECK"
        elif villain_action.startswith("BET"):
            # Find the bet size in villain's available actions
            if 'actions' in current_node:
                for action in current_node['actions']:
                    if action.startswith('BET'):
                        action_key = action
                        break
            else:
                return {}
        else:
            return {}
        
        # Navigate to the action node
        if 'childrens' in current_node and action_key in current_node['childrens']:
            current_node = current_node['childrens'][action_key]
            
            # After villain's action, it's hero's turn
            if 'strategy' in current_node and 'strategy' in current_node['strategy']:
                hand_strategies = current_node['strategy']['strategy']
                if hero_hand in hand_strategies:
                    actions = current_node['actions']
                    probs = hand_strategies[hero_hand]
                    return dict(zip(actions, probs))
    
    return {}

def convert_solver_action_to_token(solver_action: str, pot_size: float = 6.0, 
                                   effective_stack: float = 97.5, current_bet: float = 0.0) -> str:
    """Convert solver action like 'BET 2.000000' to readable format with percentages."""
    if solver_action == "CHECK":
        return "Check"
    elif solver_action == "CALL":
        return "Call"
    elif solver_action == "FOLD":
        return "Fold"
    elif solver_action.startswith("BET"):
        # Extract bet amount
        bet_amount = float(solver_action.split()[1])
        
        # Calculate percentage of pot
        if pot_size > 0:
            percentage = (bet_amount / pot_size) * 100
            
            # Detect standard bet sizes (with tolerance for rounding)
            if 30 <= percentage <= 40:
                return f"Bet 33% ({bet_amount:.1f}bb)"
            elif 45 <= percentage <= 55:
                return f"Bet 50% ({bet_amount:.1f}bb)"
            elif 95 <= percentage <= 115:
                return f"Bet 100% ({bet_amount:.1f}bb)"
            else:
                # Non-standard size
                return f"Bet {percentage:.0f}% ({bet_amount:.1f}bb)"
        else:
            return f"Bet {bet_amount:.1f}bb"
    
    elif solver_action.startswith("RAISE"):
        # Extract raise amount
        raise_amount = float(solver_action.split()[1])
        
        # For raises, calculate the raise size relative to the previous bet
        if current_bet > 0:
            raise_over_bet = raise_amount - current_bet
            percentage = (raise_over_bet / pot_size) * 100
            
            # Detect standard raise sizes
            if 30 <= percentage <= 40:
                return f"Raise to {raise_amount:.1f}bb (33% pot)"
            elif 45 <= percentage <= 55:
                return f"Raise to {raise_amount:.1f}bb (50% pot)"
            elif 95 <= percentage <= 115:
                return f"Raise to {raise_amount:.1f}bb (100% pot)"
            else:
                return f"Raise to {raise_amount:.1f}bb ({percentage:.0f}% pot)"
        else:
            return f"Raise to {raise_amount:.1f}bb"
    
    elif solver_action == "ALLIN":
        if effective_stack > 0:
            percentage = (effective_stack / pot_size) * 100
            return f"All-in {effective_stack:.1f}bb ({percentage:.0f}% pot)"
        else:
            return "All-in"
    else:
        return solver_action

def calculate_preflop_investment(preflop_history: str, position: str) -> float:
    """Calculate how much a position invested preflop based on the action."""
    # Parse the preflop history to find the position's final action
    actions = preflop_history.split(", ")
    
    investment = 0.0
    
    for action in actions:
        if position in action:
            if "raises" in action or "raise to" in action:
                # Extract amount: "UTG raises 2.5bb" or "UTG raise to 2.5bb"
                try:
                    amount_str = action.split()[-1].replace("bb", "")
                    investment = float(amount_str)
                except:
                    investment = 2.5  # Default raise size
            elif "calls" in action:
                # Need to find what they're calling
                # Look for the most recent bet/raise amount before this
                for prev_action in reversed(actions[:actions.index(action)]):
                    if "raises" in prev_action or "bets to" in prev_action or "3bets to" in prev_action or "4bets to" in prev_action:
                        try:
                            amount_str = prev_action.split()[-1].replace("bb", "")
                            investment = float(amount_str)
                            break
                        except:
                            pass
            elif "3bets to" in action:
                try:
                    amount_str = action.split()[-1].replace("bb", "")
                    investment = float(amount_str)
                except:
                    investment = 11.0  # Default 3bet size
            elif "4bets to" in action:
                try:
                    amount_str = action.split()[-1].replace("bb", "")
                    investment = float(amount_str)
                except:
                    investment = 25.0  # Default 4bet size
    
    # Special case for BB who posts 1bb
    if position == "BB" and investment == 0:
        investment = 1.0
    # Special case for SB who posts 0.5bb
    elif position == "SB" and investment == 0:
        investment = 0.5
    
    return investment

def generate_training_example(solver_file: str, tag: str = '', 
                            output_file: str = '', session_ids: set = None) -> Optional[Dict[str, str]]:
    """Generate a training example for System B with TOOL_TAGS."""
    # Parse filename
    file_info = parse_filename(solver_file)
    if not file_info['pos1'] or not file_info['pos2']:
        print(f"Skipping {solver_file}: Invalid filename format")
        return None
    
    # Expand positions if needed
    pos1 = expand_position(file_info['pos1'])
    pos2 = expand_position(file_info['pos2'])
    
    # Construct preflop history
    preflop_history, hero_pos, villain_pos = construct_preflop_history(pos1, pos2, file_info['pot_type'])
    
    # Load solver data
    json_data, input_data = load_solver_data(solver_file)
    
    # Get board and ranges
    board = input_data.get('board', '')
    if file_info['flop'] and len(file_info['flop']) == 6:
        # Parse flop from filename
        board_cards = [file_info['flop'][i:i+2] for i in range(0, 6, 2)]
        board = ' '.join(board_cards)
    
    # Determine IP/OOP
    ip_pos, oop_pos = determine_ip_oop(hero_pos, villain_pos)
    
    # Get ranges from solver input
    if hero_pos == ip_pos:
        hero_range = input_data.get('range_ip', '')
        villain_range = input_data.get('range_oop', '')
    else:
        hero_range = input_data.get('range_oop', '')
        villain_range = input_data.get('range_ip', '')
    
    # Parse ranges to get valid hands
    board_cards_list = board.split()
    
    # Parse weighted ranges from solver
    def parse_weighted_range(range_str):
        """Parse range string that may contain weights like 'AA:0.5'"""
        hands = []
        parts = range_str.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                # Weighted hand
                hand, weight = part.split(':')
                hands.append(hand.strip())
            else:
                # Full weight hand
                hands.append(part)
        # Join back without weights for eval7
        return ','.join(hands)
    
    hero_range_clean = parse_weighted_range(hero_range)
    villain_range_clean = parse_weighted_range(villain_range)
    
    hero_combos = parse_range(hero_range_clean)
    villain_combos = parse_range(villain_range_clean)
    
    # Filter out blocked hands
    valid_hero_combos = []
    for hand in hero_combos:
        cards = [hand_to_string(hand, show_suits=True)[i:i+2] for i in range(0, 4, 2)]
        if not any(c in board_cards_list for c in cards):
            valid_hero_combos.append((hand_to_string(hand, show_suits=False), cards))
    
    valid_villain_combos = []
    for hand in villain_combos:
        cards = [hand_to_string(hand, show_suits=True)[i:i+2] for i in range(0, 4, 2)]
        if not any(c in board_cards_list for c in cards):
            valid_villain_combos.append((hand_to_string(hand, show_suits=False), cards))
    
    if not valid_hero_combos or not valid_villain_combos:
        print(f"Skipping {solver_file}: No valid hands available")
        return None
    
    # Select random hands
    hero_hand_generic, hero_cards = random.choice(valid_hero_combos)
    villain_hand_generic, villain_cards = random.choice(valid_villain_combos)
    
    # Format hands for solver (remove spaces)
    hero_hand_solver = ''.join(hero_cards).replace(' ', '')
    villain_hand_solver = ''.join(villain_cards).replace(' ', '')
    
    # Get actual scenario for preflop ranges
    scenario = ""
    if file_info['pot_type'] == 'SRP':
        scenario = f"{pos1} raise, {pos2} call"
    elif file_info['pot_type'] == '3bet':
        scenario = f"{pos2} raise, {pos1} 3bet, {pos2} call"
    elif file_info['pot_type'] == '4bet':
        scenario = f"{pos1} raise, {pos2} 3bet, {pos1} 4bet, {pos2} call"
    
    # Get preflop ranges from chart
    preflop_ranges = get_ranges_from_preflop_chart(scenario)
    
    # Convert board texture
    board_texture = convert_board_texture(
        file_info.get('suits', ''),
        file_info.get('pairing', ''),
        file_info.get('hirank', ''),
        file_info.get('connectivity', '')
    )
    
    # Analyze hand category
    hand_category = analyze_hand_category(hero_cards, board_cards_list)
    
    # Calculate equity and advantages
    equity_data = calculate_equity_and_advantages(
        hero_pos, villain_pos, hero_range_clean, villain_range_clean, board, hero_cards
    )
    
    # Calculate pot and stack info first
    pot_size = input_data.get('pot', 6.0)
    
    # Calculate effective stack based on preflop action
    # Parse the preflop history to determine actual investments
    hero_investment = calculate_preflop_investment(preflop_history, hero_pos)
    villain_investment = calculate_preflop_investment(preflop_history, villain_pos)
    
    # The effective stack is starting stack minus the larger investment
    max_investment = max(hero_investment, villain_investment)
    effective_stack = 100.0 - max_investment
    
    # Override with input data if available (for special cases)
    if 'effective_stack' in input_data:
        effective_stack = input_data['effective_stack']
    
    # Determine who acts first postflop
    first_to_act, second_to_act = determine_who_acts_first_postflop(hero_pos, villain_pos)
    hero_is_oop = (hero_pos == oop_pos)
    
    # Initialize variables for potential extended scenario
    extended_scenario = False
    villain_bet_action = ""
    villain_bet_size = 0.0
    hero_facing_action = "first_to_act"
    previous_action_str = ""
    
    # Check if hero acts first
    if hero_pos == first_to_act:
        # Hero acts first - get hero's initial strategy
        hero_strategy = get_hero_strategy(json_data, hero_hand_solver, hero_pos, villain_pos, "first_to_act")
        
        if hero_strategy:
            best_action = max(hero_strategy.items(), key=lambda x: x[1])
            solver_action, frequency = best_action
            
            # Check for extended scenario: OOP hero checks + random chance for extended scenario
            if hero_is_oop and solver_action == "CHECK" and random.random() < 0.3:
                # Navigate to CHECK node to see villain's response
                if 'childrens' in json_data and 'CHECK' in json_data['childrens']:
                    check_node = json_data['childrens']['CHECK']
                    
                    # Get villain's strategy after hero checks
                    if 'strategy' in check_node and 'strategy' in check_node['strategy']:
                        villain_strat = check_node['strategy']['strategy']
                        if villain_hand_solver in villain_strat:
                            villain_actions = check_node['actions']
                            villain_probs = villain_strat[villain_hand_solver]
                            
                            # Get villain's highest probability action
                            max_prob_idx = villain_probs.index(max(villain_probs))
                            villain_solver_action = villain_actions[max_prob_idx]
                            
                            # Check if villain wants to bet
                            if villain_solver_action.startswith("BET"):
                                extended_scenario = True
                                
                                # Extract bet size and format it
                                bet_size_bb = float(villain_solver_action.split()[1])
                                bet_percentage = (bet_size_bb / pot_size) * 100
                                
                                # Format to standard sizes and calculate exact amount
                                if 30 <= bet_percentage <= 40:
                                    display_pct = 33
                                    actual_bet_amount = round(pot_size * 0.33, 1)
                                elif 45 <= bet_percentage <= 55:
                                    display_pct = 50
                                    actual_bet_amount = round(pot_size * 0.50, 1)
                                elif 95 <= bet_percentage <= 115:
                                    display_pct = 100
                                    actual_bet_amount = round(pot_size * 1.00, 1)
                                else:
                                    display_pct = int(bet_percentage)
                                    actual_bet_amount = bet_size_bb
                                
                                villain_bet_action = f"{villain_pos} bets {actual_bet_amount:.1f}bb ({display_pct}% pot)"
                                villain_bet_size = actual_bet_amount
                                
                                # Get hero's response strategy from the BET node
                                bet_node_key = None
                                if 'childrens' in check_node:
                                    for action_key in check_node['childrens']:
                                        if action_key.startswith('BET'):
                                            bet_node_key = action_key
                                            break
                                
                                if bet_node_key and bet_node_key in check_node['childrens']:
                                    bet_response_node = check_node['childrens'][bet_node_key]
                                    if 'strategy' in bet_response_node and 'strategy' in bet_response_node['strategy']:
                                        hand_strategies = bet_response_node['strategy']['strategy']
                                        if hero_hand_solver in hand_strategies:
                                            actions = bet_response_node['actions']
                                            probs = hand_strategies[hero_hand_solver]
                                            hero_strategy = dict(zip(actions, probs))
                                            hero_facing_action = "bet"
                                            previous_action_str = f"{hero_pos} checks, {villain_bet_action}"
                                
        # Set up scenario variables
        if not extended_scenario:
            # Normal scenario: Hero acts first
            hero_facing_action = "first_to_act"
            previous_action_str = ""
    else:
        # Villain acts first - get their action and have hero respond
        villain_action, villain_bet_size = get_villain_action(json_data, villain_hand_solver, villain_pos, hero_pos)
        
        # Format villain's action
        if villain_action == "CHECK":
            previous_action_str = f"{villain_pos} checks"
            hero_facing_action = "check"
        elif villain_action.startswith("BET"):
            # Calculate percentage correctly
            bet_percentage = (villain_bet_size / pot_size) * 100
            
            # Detect standard bet sizes and calculate exact amount
            if 30 <= bet_percentage <= 40:
                display_pct = 33
                actual_bet_amount = round(pot_size * 0.33, 1)
            elif 45 <= bet_percentage <= 55:
                display_pct = 50
                actual_bet_amount = round(pot_size * 0.50, 1)
            elif 95 <= bet_percentage <= 115:
                display_pct = 100
                actual_bet_amount = round(pot_size * 1.00, 1)
            else:
                display_pct = int(bet_percentage)
                actual_bet_amount = villain_bet_size
                
            previous_action_str = f"{villain_pos} bets {actual_bet_amount:.1f}bb ({display_pct}% pot)"
            hero_facing_action = "bet"
        else:
            previous_action_str = f"{villain_pos} {villain_action}"
            hero_facing_action = "check"
        
        # Get hero's strategy facing villain's action
        hero_strategy = get_hero_strategy(json_data, hero_hand_solver, hero_pos, villain_pos, villain_action)
    
    # If we don't have a strategy yet, get it now
    if not hero_strategy:
        hero_strategy = {"CHECK": 1.0}
    
    # Find highest frequency action for hero
    if hero_strategy:
        best_action = max(hero_strategy.items(), key=lambda x: x[1])
        solver_action = best_action[0]
        solver_freq = best_action[1]
    else:
        solver_action = "CHECK"
        solver_freq = 1.0
    
    # Format the prompt
    prompt_parts = []
    
    # HAND_META
    prompt_parts.append("### HAND_META")
    prompt_parts.append("game: cash")
    prompt_parts.append("seats: 8-max")
    prompt_parts.append("stacks: 100bb")
    prompt_parts.append(f"hero_pos: {hero_pos}")
    prompt_parts.append(f"hero_hand: {hero_cards[0][0]}{hero_cards[0][1].lower()} {hero_cards[1][0]}{hero_cards[1][1].lower()}")
    prompt_parts.append("villain_profile: balanced")
    
    # HISTORY_PREFLOP
    prompt_parts.append("\n### HISTORY_PREFLOP")
    prompt_parts.append(f"preflop: {preflop_history}")
    
    # HISTORY 1
    prompt_parts.append("\n### HISTORY 1")
    prompt_parts.append(f"flop: ({board})    pot: {pot_size:.1f}bb")
    prompt_parts.append(f"stacks: {effective_stack:.1f}bb")
    
    # Add action history
    if previous_action_str:
        prompt_parts.append(f"actions: {previous_action_str}")
    else:
        prompt_parts.append("actions:")
    
    # Get legal actions based on what hero is facing
    legal_actions = format_legal_actions(hero_pos, 'flop', hero_facing_action, file_info['pot_type'], pot_size)
    
    # DECISION 1
    prompt_parts.append("\n### DECISION 1")
    prompt_parts.append("street: flop")
    prompt_parts.append(f"pot: {pot_size:.1f}bb")
    prompt_parts.append("to_act: HERO")
    prompt_parts.append(f"legal: [{','.join(legal_actions)}]")
    
    # TOOL_TAGS
    prompt_parts.append("\n### TOOL_TAGS")
    prompt_parts.append(f"board_texture: {board_texture}")
    
    # Add ranges
    if hero_pos in preflop_ranges:
        prompt_parts.append(f"hero_range: [{preflop_ranges[hero_pos]}]")
    else:
        prompt_parts.append(f"hero_range: [{hero_range}]")
    
    if villain_pos in preflop_ranges:
        prompt_parts.append(f"villain_range: [{preflop_ranges[villain_pos]}]")
    else:
        prompt_parts.append(f"villain_range: [{villain_range}]")
    
    # Add advantages
    prompt_parts.append(f"range_adv: {equity_data['range_adv_category']}    # eq_gap = {equity_data['range_adv_value']*100:+.1f}%")
    prompt_parts.append(f"nut_adv: {equity_data['nut_adv_category']}    # nut_gap = {equity_data['nut_adv_value']:+.1f}%")
    
    # Add hand category and ranking
    prompt_parts.append(f"hero_hand_category: {hand_category}")
    prompt_parts.append(f"hero_hand_ranking: top {100-equity_data['hero_percentile']:.0f}%")
    
    prompt = "\n".join(prompt_parts)
    
    # Format completion
    completion = f"**{convert_solver_action_to_token(solver_action, pot_size, effective_stack)}**"
    
    # Generate unique custom_id
    if session_ids is None:
        session_ids = set()
    
    existing_ids = get_existing_custom_ids(output_file) if output_file else set()
    custom_id = generate_unique_custom_id(solver_file, existing_ids, tag, session_ids)
    session_ids.add(custom_id)
    
    return {
        "custom_id": custom_id,
        "prompt": prompt,
        "completion": completion
    }

def main():
    """Main function to generate training data."""
    parser = argparse.ArgumentParser(description='Generate System B poker training data with TOOL_TAGS')
    parser.add_argument('-n', '--num-examples', type=int, default=10,
                        help='Number of examples to generate')
    parser.add_argument('-o', '--output', type=str, default='systemB_training_data.jsonl',
                        help='Output JSONL file')
    parser.add_argument('-t', '--tag', type=str, default='',
                        help='Tag to prepend to custom_ids')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing file instead of appending (default: append)')
    
    args = parser.parse_args()
    
    # Get solver files
    solver_files = get_solver_files()
    if not solver_files:
        print("No solver output files found!")
        return
    
    print(f"Found {len(solver_files)} solver output files")
    
    # Track session IDs to avoid duplicates within this run
    session_ids = set()
    
    # Open file in append or write mode
    mode = 'w' if args.overwrite else 'a'
    
    # Generate and write examples one at a time
    examples_generated = 0
    attempts = 0
    max_attempts = args.num_examples * 3  # Allow for some failures
    
    with open(args.output, mode) as f:
        while examples_generated < args.num_examples and attempts < max_attempts:
            attempts += 1
            
            # Pick random solver file
            solver_file = random.choice(solver_files)
            
            try:
                example = generate_training_example(solver_file, args.tag, args.output, session_ids)
                if example:
                    # Write immediately and flush to ensure it's written to disk
                    f.write(json.dumps(example) + '\n')
                    f.flush()  # Force write to disk
                    examples_generated += 1
                    print(f"Generated example {examples_generated}/{args.num_examples} from {solver_file}")
            except Exception as e:
                print(f"Error generating example from {solver_file}: {e}")
                import traceback
                traceback.print_exc()
    
    total_examples = count_jsonl_examples(args.output)
    print(f"\nSuccessfully generated {examples_generated} examples")
    print(f"Total examples in {args.output}: {total_examples}")

if __name__ == "__main__":
    main() 