#!/usr/bin/env python3
"""
Generate poker training data in JSONL format from solver outputs.

This script creates training examples for a poker-reasoning language model
by parsing solver outputs and generating properly formatted prompts and completions.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Try to import OpenAI - it's optional
try:
    import openai
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()
    # Configure OpenAI
    openai.api_key = os.getenv('OPENAI_API_KEY')
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Suppressed warning for evaluation scripts - we don't need OpenAI for local evaluation

# Import solver utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'solver'))
from preflop_scenario_generator import (
    POSITION_GROUPS,
    determine_ip_oop,
    parse_scenario
)
from solver_inputer import RANK_VALUES

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
        
        # Determine bet sizes based on positions
        if threebetter in ['SB', 'BB']:
            threebet_size = OOP_3BET_SIZE  # 11bb
            fourbet_size = IP_4BET_SIZE    # 25bb
        else:
            threebet_size = IP_3BET_SIZE   # 7.5bb  
            fourbet_size = OOP_4BET_SIZE   # 22bb
        
        # Build complete action sequence
        acted = set()
        
        # Everyone folds up to the initial raiser
        for i in range(raiser_idx):
            actions.append(f"{PREFLOP_POSITIONS[i]} folds")
            acted.add(PREFLOP_POSITIONS[i])
        
        # Initial raise
        actions.append(f"{initial_raiser} raises {INITIAL_RAISE_SIZE}bb")
        acted.add(initial_raiser)
        
        # Handle folds between raiser and 3better
        if threebetter_idx > raiser_idx:
            # Normal flow: 3better comes after raiser
            for i in range(raiser_idx + 1, threebetter_idx):
                actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                acted.add(PREFLOP_POSITIONS[i])
        else:
            # 3better is in earlier position (blinds) - fold everyone after raiser
            for i in range(raiser_idx + 1, len(PREFLOP_POSITIONS)):
                if PREFLOP_POSITIONS[i] not in [initial_raiser, threebetter]:
                    actions.append(f"{PREFLOP_POSITIONS[i]} folds")
                    acted.add(PREFLOP_POSITIONS[i])
        
        # 3bet
        actions.append(f"{threebetter} 3bets to {threebet_size}bb")
        acted.add(threebetter)
        
        # Everyone else folds (except the two active players)
        for position in PREFLOP_POSITIONS:
            if position not in acted and position not in [initial_raiser, threebetter]:
                actions.append(f"{position} folds")
        
        # 4bet and call
        actions.append(f"{initial_raiser} 4bets to {fourbet_size}bb")
        actions.append(f"{threebetter} calls {fourbet_size}bb")
    
    preflop_history = ", ".join(actions)
    return preflop_history, hero_pos, villain_pos


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


def parse_range(range_str: str) -> List[Tuple[str, float]]:
    """Parse a range string into list of (hand, weight) tuples."""
    hands = []
    
    if not range_str:
        return hands
    
    # Split by comma
    parts = range_str.split(',')
    
    for part in parts:
        if ':' in part:
            # Weighted hand (e.g., "AA:0.5")
            hand, weight = part.split(':')
            hands.append((hand.strip(), float(weight)))
        else:
            # Full weight hand
            hands.append((part.strip(), 1.0))
    
    return hands


def get_valid_hand(range_hands: List[Tuple[str, float]], 
                   board_cards: List[str], 
                   blocked_cards: List[str] = None) -> Optional[Tuple[str, List[str]]]:
    """Get a valid hand from range that doesn't conflict with board or blocked cards."""
    if blocked_cards is None:
        blocked_cards = []
    
    # Convert to eval7 style cards
    def hand_to_cards(hand_str: str) -> List[str]:
        """Convert hand notation to cards."""
        # Handle different formats
        if len(hand_str) == 4:  # e.g., "AsKh"
            return [hand_str[0:2], hand_str[2:4]]
        elif len(hand_str) == 3:  # e.g., "AKs"
            rank1, rank2, suit = hand_str[0], hand_str[1], hand_str[2]
            if suit == 's':
                # Suited - pick random suits
                suits = ['h', 'd', 'c', 's']
                random.shuffle(suits)
                return [rank1 + suits[0], rank2 + suits[0]]
            else:
                # Offsuit - pick different suits
                suits = ['h', 'd', 'c', 's']
                random.shuffle(suits)
                return [rank1 + suits[0], rank2 + suits[1]]
        elif len(hand_str) == 2:  # e.g., "AA"
            rank = hand_str[0]
            suits = ['h', 'd', 'c', 's']
            random.shuffle(suits)
            return [rank + suits[0], rank + suits[1]]
        else:
            return []
    
    # Filter valid hands
    valid_hands = []
    for hand_str, weight in range_hands:
        if weight > 0:
            cards = hand_to_cards(hand_str)
            if cards and not any(c in board_cards + blocked_cards for c in cards):
                valid_hands.append((hand_str, weight, cards))
    
    if not valid_hands:
        return None
    
    # Weighted random selection
    total_weight = sum(weight for _, weight, _ in valid_hands)
    r = random.uniform(0, total_weight)
    
    cumsum = 0
    for hand_str, weight, cards in valid_hands:
        cumsum += weight
        if r <= cumsum:
            return hand_str, cards
    
    return valid_hands[-1][0], valid_hands[-1][2]  # Fallback


def get_postflop_action_order(active_positions: List[str]) -> List[str]:
    """Get the correct postflop action order for active positions.
    
    Postflop order: SB, BB, UTG, UTG+1, LJ, HJ, CO, BTN
    """
    postflop_order = ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"]
    return [pos for pos in postflop_order if pos in active_positions]


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


def get_villain_action(json_data: Dict[str, Any], villain_hand: str, villain_pos: str, 
                       hero_pos: str) -> Tuple[str, float]:
    """Get villain's action for a specific hand from solver data - always choose highest frequency.
    
    Returns: (action, size_in_bb)
    """
    # Determine who acts first postflop
    first_to_act, second_to_act = determine_who_acts_first_postflop(hero_pos, villain_pos)
    
    # If hero acts first, villain acts second (so villain responds to hero's action)
    # If villain acts first, we get villain's action from the root
    if villain_pos == first_to_act:
        # Villain acts first - look at root level strategy
        if 'strategy' in json_data and 'strategy' in json_data['strategy']:
            hand_strategies = json_data['strategy']['strategy']
            if villain_hand in hand_strategies:
                actions = json_data['strategy']['actions']
                probs = hand_strategies[villain_hand]
                
                # Find the highest frequency action
                max_prob_idx = probs.index(max(probs))
                sampled_action = actions[max_prob_idx]
                
                # Parse action to get bet size
                if sampled_action == "CHECK":
                    return "check", 0.0
                elif sampled_action.startswith("BET"):
                    # Extract size from "BET X.XX"
                    size = float(sampled_action.split()[1])
                    return f"bet", size
                else:
                    return sampled_action.lower(), 0.0
    else:
        # Hero acts first, villain acts second - for simplicity, have them check
        # In a more complete implementation, we'd navigate the tree based on hero's action
        return "check", 0.0
    
    # Default to check if strategy not found
    return "check", 0.0


def format_legal_actions(hero_pos: str, street: str, facing_action: str, pot_type: str = 'SRP', pot_size: float = 6.0) -> List[str]:
    """Determine legal actions based on game context.
    
    Args:
        hero_pos: Hero's position
        street: Current street
        facing_action: What action hero is facing
        pot_type: Type of pot (SRP, 3bet, 4bet) to determine effective stack
        pot_size: Current pot size in bb for calculating bet amounts
    """
    if street == "preflop":
        # This shouldn't happen in our current implementation
        return ["fold", "call", "raise_2.5"]
    
    # Helper function to format bet amounts
    def format_bet(percentage: int) -> str:
        bet_amount = round((percentage / 100.0) * pot_size, 1)
        return f"bet {percentage}% ({bet_amount}bb)"
    
    # Postflop cases
    if facing_action == "first_to_act":
        return ["check", format_bet(33), format_bet(50), format_bet(100), "allin"]
    elif facing_action == "check":
        return ["check", format_bet(33), format_bet(50), format_bet(100), "allin"]
    elif facing_action in ["bet", "raise"]:
        # When facing a bet, typical actions are fold, call, or raise
        # Raise sizes depend on the pot type and stack depth
        if pot_type == '4bet':
            # In 4bet pots with shallow stacks, often just fold/call/allin
            return ["fold", "call", "allin"]
        else:
            return ["fold", "call", "raise 100%", "allin"]
    else:
        return ["check", format_bet(33), format_bet(50), format_bet(100)]


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
                actions = json_data['strategy']['actions']
                probs = hand_strategies[hero_hand]
                return dict(zip(actions, probs))
        return {}
    
    # If villain acted first, we need to navigate to their action node to get hero's response
    if villain_pos == first_to_act:
        current_node = json_data
        
        # Navigate to villain's action
        if villain_action == "check":
            action_key = "CHECK"
        elif villain_action.startswith("bet"):
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
                    actions = current_node['strategy']['actions']
                    probs = hand_strategies[hero_hand]
                    return dict(zip(actions, probs))
    
    return {}


def convert_solver_action_to_token(solver_action: str, pot_size: float = 6.0, 
                                   effective_stack: float = 97.5, current_bet: float = 0.0) -> str:
    """Convert solver action format to training token format.
    
    Args:
        solver_action: Action string from solver (e.g., "BET 2.0", "RAISE 10.0")
        pot_size: Current pot size in bb
        effective_stack: Effective stack size in bb
        current_bet: Current bet to call (for raises)
    """
    if solver_action == "CHECK":
        return "check"
    elif solver_action == "FOLD":
        return "fold"
    elif solver_action == "CALL":
        return "call"
    elif solver_action == "ALLIN":
        return "allin"
    elif solver_action.startswith("BET"):
        # Extract size in bb
        size_bb = float(solver_action.split()[1])
        
        # Check if it's all-in
        if size_bb >= effective_stack - 0.5:
            return "allin"
        
        # Calculate percentage of pot
        percentage = (size_bb / pot_size) * 100
        
        # Detect standard bet sizes (accounting for solver rounding)
        if 30 <= percentage <= 40:
            return "bet_33"
        elif 45 <= percentage <= 55:
            return "bet_50"
        elif 95 <= percentage <= 115:
            return "bet_100"
        else:
            # For non-standard sizes, map to closest standard size
            if percentage <= 40:
                return "bet_33"
            elif percentage <= 75:
                return "bet_50"
            else:
                return "bet_100"
    elif solver_action.startswith("RAISE"):
        # Extract total raise size in bb
        total_size_bb = float(solver_action.split()[1])
        
        # Check if it's all-in
        if total_size_bb >= effective_stack - 0.5:
            return "allin"
        
        # Calculate raise amount over the bet
        pot_after_call = pot_size + current_bet
        raise_amount = total_size_bb - current_bet
        percentage = (raise_amount / pot_after_call) * 100
        
        # Detect standard raise sizes
        if 70 <= percentage <= 80:
            return "raise_75"
        elif 95 <= percentage <= 105:
            return "raise_100"
        else:
            # For non-standard sizes, map to closest
            if percentage <= 85:
                return "raise_75"
            else:
                return "raise_100"
    else:
        return "check"  # Default


def calculate_preflop_investment(preflop_history: str, position: str) -> float:
    """Calculate how much a position invested during preflop action.
    
    Args:
        preflop_history: The preflop action string
        position: The position to calculate investment for
    
    Returns:
        Amount invested in bb
    """
    investment = 0.0
    
    # Parse the preflop history to find this position's actions
    actions = preflop_history.split(', ')
    
    for action in actions:
        if action.startswith(position + ' '):
            action_part = action[len(position) + 1:]  # Remove "POSITION "
            
            if 'raises' in action_part:
                # Extract amount from "raises X.Xbb"
                amount_str = action_part.split('raises ')[1].replace('bb', '')
                investment = float(amount_str)
            elif 'calls' in action_part:
                # Extract amount from "calls X.Xbb" 
                amount_str = action_part.split('calls ')[1].replace('bb', '')
                investment = float(amount_str)
            elif '3bets to' in action_part:
                # Extract amount from "3bets to X.Xbb"
                amount_str = action_part.split('3bets to ')[1].replace('bb', '')
                investment = float(amount_str)
            elif '4bets to' in action_part:
                # Extract amount from "4bets to X.Xbb"
                amount_str = action_part.split('4bets to ')[1].replace('bb', '')
                investment = float(amount_str)
    
    return investment


def generate_training_example(solver_file: str, mode: str = 'no-tools', tag: str = '', 
                               max_tokens: int = 100, temperature: float = 0.7,
                               output_file: str = '', session_ids: set = None) -> Optional[Dict[str, str]]:
    """Generate a single training example from a solver file.
    
    Returns dict with 'prompt' and 'completion' keys.
    """
    # Parse filename
    file_info = parse_filename(solver_file)
    
    # Expand generalized positions
    pos1 = expand_position(file_info['pos1'], file_info)
    pos2 = expand_position(file_info['pos2'], file_info)
    
    # Construct preflop history
    preflop_history, hero_pos, villain_pos = construct_preflop_history(
        pos1, pos2, file_info['pot_type']
    )
    
    # Choose villain profile with specified probabilities
    profiles = [
        ("balanced", 0.70),
        ("tag (tight-aggressive)", 0.05),
        ("lag (loose-aggressive)", 0.05),
        ("nit (super-tight)", 0.05),
        ("station (calling-station)", 0.05),
        ("maniac (loose, hyper-aggressive)", 0.05),
        ("whale (loose-passive)", 0.05)
    ]
    
    # Weighted random selection
    r = random.random()
    cumsum = 0
    villain_profile = "balanced"  # default
    for profile, weight in profiles:
        cumsum += weight
        if r <= cumsum:
            villain_profile = profile
            break
    
    # Load solver data
    json_data, input_data = load_solver_data(solver_file)
    
    # Get pot size and effective stack
    pot_size = input_data.get('pot', 6.0)  # Default to 6.0 for SRP
    
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
    
    # Parse board
    board_cards = []
    if 'board' in input_data:
        board_str = input_data['board']
        board_cards = [card.strip() for card in board_str.split(',')]
    elif file_info['flop']:
        # Parse from filename
        flop = file_info['flop']
        board_cards = [flop[i:i+2] for i in range(0, len(flop), 2)]
    
    # Determine IP/OOP
    ip_pos, oop_pos = determine_ip_oop(hero_pos, villain_pos)
    hero_is_oop = (hero_pos == oop_pos)
    
    # Get ranges
    if hero_pos == ip_pos:
        hero_range_str = input_data.get('range_ip', '')
        villain_range_str = input_data.get('range_oop', '')
    else:
        hero_range_str = input_data.get('range_oop', '')
        villain_range_str = input_data.get('range_ip', '')
    
    # Parse ranges
    hero_range = parse_range(hero_range_str)
    villain_range = parse_range(villain_range_str)
    
    # Select valid hands
    hero_hand_result = get_valid_hand(hero_range, board_cards)
    if not hero_hand_result:
        return None
    hero_hand_str, hero_cards = hero_hand_result
    
    villain_hand_result = get_valid_hand(villain_range, board_cards, hero_cards)
    if not villain_hand_result:
        return None
    villain_hand_str, villain_cards = villain_hand_result
    
    # Determine who acts first postflop
    first_to_act, second_to_act = determine_who_acts_first_postflop(hero_pos, villain_pos)
    
    # Randomly decide whether hero acts first or responds to villain
    # This creates more balanced training data
    hero_acts_first = (hero_pos == first_to_act)
    
    # Initialize variables for potential extended scenario
    extended_scenario = False
    villain_bet_action = ""
    villain_bet_size = 0.0
    
    # Initialize variables
    best_action_display = ""
    hero_strategy = {}
    
    if hero_acts_first:
        # Hero acts first - get hero's best action
        hero_hand_solver = ''.join(hero_cards)
        hero_strategy = get_hero_strategy(json_data, hero_hand_solver, hero_pos, villain_pos, "first_to_act")
        
        if hero_strategy:
            best_action = max(hero_strategy.items(), key=lambda x: x[1])
            solver_action, frequency = best_action
            action_token = convert_solver_action_to_token(solver_action, pot_size, effective_stack, 0.0)
            
            # Check for extended scenario: OOP hero checks + 30% chance
            if hero_is_oop and action_token == "check" and random.random() < 0.3:
                # Navigate to CHECK node to see villain's response
                if 'childrens' in json_data and 'CHECK' in json_data['childrens']:
                    check_node = json_data['childrens']['CHECK']
                    
                    # Get villain's strategy after hero checks
                    villain_hand_solver = ''.join(villain_cards)
                    if 'strategy' in check_node and 'strategy' in check_node['strategy']:
                        villain_strat = check_node['strategy']['strategy']
                        if villain_hand_solver in villain_strat:
                            villain_actions = check_node['strategy']['actions']
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
                                
                                # Format to standard sizes
                                if 30 <= bet_percentage <= 40:
                                    display_pct = 33
                                elif 45 <= bet_percentage <= 55:
                                    display_pct = 50
                                elif 95 <= bet_percentage <= 115:
                                    display_pct = 100
                                else:
                                    display_pct = int(bet_percentage)
                                
                                villain_bet_action = f"{villain_pos} bets {bet_size_bb:.1f}bb ({display_pct}% pot)"
                                villain_bet_size = bet_size_bb
        
        # Set up the scenario
        if extended_scenario:
            # Extended scenario: Hero checked, villain bet, now hero responds
            previous_action_str = f"{hero_pos} checks. {villain_bet_action}"
            facing_action = "bet"
            current_bet = villain_bet_size
            hero_facing_action = "bet"
            
            # Get hero's response strategy from the BET node
            bet_node_key = None
            check_node = json_data['childrens']['CHECK']
            if 'childrens' in check_node:
                for action_key in check_node['childrens']:
                    if action_key.startswith('BET'):
                        bet_node_key = action_key
                        break
            
            if bet_node_key and bet_node_key in check_node['childrens']:
                bet_response_node = check_node['childrens'][bet_node_key]
                hero_strategy = {}
                if 'strategy' in bet_response_node and 'strategy' in bet_response_node['strategy']:
                    hand_strategies = bet_response_node['strategy']['strategy']
                    if hero_hand_solver in hand_strategies:
                        actions = bet_response_node['strategy']['actions']
                        probs = hand_strategies[hero_hand_solver]
                        hero_strategy = dict(zip(actions, probs))
        else:
            # Normal scenario: Hero acts first
            previous_action_str = ""
            facing_action = "first_to_act"
            current_bet = 0.0
            hero_facing_action = "first_to_act"
    else:
        # Villain acts first - get their action and have hero respond
        villain_hand_solver = ''.join(villain_cards)
        villain_action, villain_bet_size = get_villain_action(json_data, villain_hand_solver, villain_pos, hero_pos)
        
        # Format villain's action
        if villain_action == "check":
            previous_action_str = f"{villain_pos} checks."
            facing_action = "check"
            current_bet = 0.0
        elif villain_action.startswith("bet"):
            # Calculate percentage correctly
            bet_percentage = (villain_bet_size / pot_size) * 100
            
            # Detect standard bet sizes
            if 30 <= bet_percentage <= 40:
                display_pct = 33
            elif 45 <= bet_percentage <= 55:
                display_pct = 50
            elif 95 <= bet_percentage <= 115:
                display_pct = 100
            else:
                display_pct = int(bet_percentage)
                
            previous_action_str = f"{villain_pos} bets {villain_bet_size:.1f}bb ({display_pct}% pot)"
            facing_action = "bet"
            current_bet = villain_bet_size
        else:
            previous_action_str = f"{villain_pos} {villain_action}"
            facing_action = "check"
            current_bet = 0.0
        
        hero_facing_action = villain_action
    
    # Determine legal actions for hero
    if extended_scenario:
        # Hero is facing a bet after checking - get legal actions from the bet response node
        legal_actions = []
        check_node = json_data['childrens']['CHECK']
        bet_node_key = None
        if 'childrens' in check_node:
            for action_key in check_node['childrens']:
                if action_key.startswith('BET'):
                    bet_node_key = action_key
                    break
        
        if bet_node_key and bet_node_key in check_node['childrens']:
            bet_response_node = check_node['childrens'][bet_node_key]
            if 'actions' in bet_response_node:
                for action in bet_response_node['actions']:
                    if action == "FOLD":
                        legal_actions.append("fold")
                    elif action == "CALL":
                        legal_actions.append("call")
                    elif action.startswith("RAISE"):
                        legal_actions.append("raise 75%")  # Standard raise sizing
                    elif action == "ALLIN":
                        legal_actions.append("allin")
        
        if not legal_actions:  # Fallback
            legal_actions = ["fold", "call", "raise 75%", "allin"]
    else:
        legal_actions = format_legal_actions(hero_pos, "flop", facing_action, file_info['pot_type'], pot_size)
    
    # Format hero hand
    hero_hand_formatted = ' '.join(hero_cards)
    
    # Format flop string
    flop_str = ' '.join(board_cards)
    
    # Get hero's strategy to determine the best action (only needed for tools modes)
    if mode != 'no-tools':
        if not extended_scenario:
            hero_hand_solver_format = ''.join(hero_cards)
            hero_strategy = get_hero_strategy(json_data, hero_hand_solver_format, hero_pos, villain_pos, hero_facing_action)
    
        # Get the best action and format it for display (only for tools modes)
        if hero_strategy:
            best_action = max(hero_strategy.items(), key=lambda x: x[1])
            solver_action, frequency = best_action
            action_token = convert_solver_action_to_token(solver_action, pot_size, effective_stack, current_bet)
            
            # Ensure the action is actually legal - if not, pick the first legal action
            legal_action_tokens = []
            for legal_action in legal_actions:
                if legal_action.startswith("bet "):
                    # Extract the token from "bet 33% (2.1bb)" format
                    if "33%" in legal_action:
                        legal_action_tokens.append("bet_33")
                    elif "50%" in legal_action:
                        legal_action_tokens.append("bet_50")
                    elif "100%" in legal_action:
                        legal_action_tokens.append("bet_100")
                else:
                    legal_action_tokens.append(legal_action)
            
            # If the solver's best action isn't legal, find the best legal alternative
            if action_token not in legal_action_tokens:
                # Try to find the best legal action from the strategy
                legal_solver_actions = []
                for solver_act, prob in hero_strategy.items():
                    token = convert_solver_action_to_token(solver_act, pot_size, effective_stack, current_bet)
                    if token in legal_action_tokens:
                        legal_solver_actions.append((solver_act, prob, token))
                
                if legal_solver_actions:
                    # Pick the highest probability legal action
                    best_legal = max(legal_solver_actions, key=lambda x: x[1])
                    solver_action, frequency, action_token = best_legal
                else:
                    # Fallback to first legal action
                    action_token = legal_action_tokens[0] if legal_action_tokens else "check"
            
            # Format the action for display in BEST_ACTION line
            if action_token.startswith("bet_"):
                # Extract percentage and format properly
                if action_token == "bet_33":
                    percentage = 33
                elif action_token == "bet_50":
                    percentage = 50
                elif action_token == "bet_100":
                    percentage = 100
                else:
                    percentage = 50  # Default
                
                bet_amount = round((percentage / 100.0) * pot_size, 1)
                best_action_display = f"bet {percentage}% ({bet_amount}bb)"
            else:
                best_action_display = action_token
        else:
            # No strategy available - pick first legal action (only for tools modes)
            first_legal = legal_actions[0] if legal_actions else "check"
            if first_legal.startswith("bet "):
                best_action_display = first_legal
            else:
                best_action_display = first_legal
    
    # Calculate remaining stacks after preflop investment - but use effective_stack as the base
    # since it already accounts for preflop investment
    remaining_stack = effective_stack
    
    # Build prompt based on mode
    if mode == 'no-tools':
        # No tools mode: Don't show solver's best action, ask model to determine it
        prompt_lines = [
            "### HAND_META",
            "game: cash",
            "seats: 8-max",
            "stacks: 100bb",
            f"hero_pos: {hero_pos}",
            f"hero_hand: {hero_hand_formatted}",
            f"villain_profile: {villain_profile}",
            "",
            "### HISTORY_PREFLOP",
            f"preflop: {preflop_history}",
            "",
            "### HISTORY 1",
            f"flop: ({flop_str})    pot: {pot_size:.1f}bb",
            f"stacks: {remaining_stack:.1f}bb",
        ]
        
        # Add action line
        if previous_action_str:
            prompt_lines.append(f"actions: {previous_action_str}")
        else:
            prompt_lines.append("actions:")
        
        prompt_lines.extend([
            "",
            "### DECISION 1",
            "street: flop",
            f"pot: {pot_size:.1f}bb",
            "to_act: HERO",
            f"legal: [{','.join(legal_actions)}]"
        ])
        
        prompt = '\n'.join(prompt_lines)
        
        # User message asks model to choose action and justify
        user_message = f"{prompt}\n\nChoose the best action from the legal options and provide a justification under 5 sentences. State the best action first. Consider position (IP/OOP), board texture, range advantage, nut advantage, and villain profile for possible exploits."
        
    else:
        # Tools modes: Show solver's best action (for tools-no-cot and tools-cot)
        prompt_lines = [
            "### HAND_META",
            "game: cash",
            "seats: 8-max",
            "stacks: 100bb",
            f"hero_pos: {hero_pos}",
            f"hero_hand: {hero_hand_formatted}",
            f"villain_profile: {villain_profile}",
            "",
            "### HISTORY_PREFLOP",
            f"preflop: {preflop_history}",
            "",
            "### HISTORY 1",
            f"flop: ({flop_str})    pot: {pot_size:.1f}bb",
            f"stacks: {remaining_stack:.1f}bb",
        ]
        
        # Add action line
        if previous_action_str:
            prompt_lines.append(f"actions: {previous_action_str}")
        else:
            prompt_lines.append("actions:")
        
        prompt_lines.extend([
            "",
            "### DECISION 1",
            "street: flop",
            f"pot: {pot_size:.1f}bb",
            "to_act: HERO",
            f"legal: [{','.join(legal_actions)}]",
            f"BEST_ACTION (solver): {best_action_display}"
        ])
        
        prompt = '\n'.join(prompt_lines)
        
        if mode == 'tools-no-cot':
            # Just return the action token
            user_message = f"{prompt}\n\nReturn only the action token from the legal options."
        else:  # tools-cot
            # Full justification with solver action provided
            user_message = f"{prompt}\n\nJustify the action '{best_action_display}' in at most 3 concise sentences. Consider position (IP/OOP), board texture, range advantage, nut advantage, and villain profile for possible exploits."
    
    # Create system message with general poker expert instructions
    system_message = "You are a world-class poker professional with deep understanding of game theory optimal (GTO) play, opponent profiling, and advanced poker strategy. You excel at providing concise, strategic justifications for poker decisions."
    
    # Print the prompt for visibility
    print("\n" + "="*80)
    print("GENERATED PROMPT:")
    print("="*80)
    print(user_message)
    print("="*80 + "\n")
    
    # Generate custom_id with optional tag
    existing_ids = get_existing_custom_ids(output_file) if output_file else set()
    custom_id = generate_unique_custom_id(solver_file, existing_ids, tag, session_ids)
    
    # Return in OpenAI Batch API format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "temperature": temperature,
            "max_tokens": max_tokens,  # Keep max_tokens for batch API compatibility
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ]
        }
    }


def main():
    """Main function to generate training data."""
    parser = argparse.ArgumentParser(description='Generate poker training data from solver outputs')
    parser.add_argument('-n', '--num-examples', type=int, default=10,
                       help='Number of training examples to generate')
    parser.add_argument('-o', '--output', type=str, default='poker_training_data.jsonl',
                       help='Output JSONL filename')
    parser.add_argument('--mode', type=str, choices=['no-tools', 'tools-no-cot', 'tools-cot'], 
                       default='no-tools',
                       help='Experimental mode: no-tools (model intuition only), tools-no-cot (solver action only), tools-cot (full system)')
    parser.add_argument('--tag', type=str, default='',
                       help='Optional tag to add to custom_id for organizing experiments')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens for completion')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for completion generation')
    parser.add_argument('--real-time', action='store_true',
                       help='Generate and process examples one by one with immediate API calls')
    
    args = parser.parse_args()
    
    # Get available solver files
    solver_files = get_solver_files()
    if not solver_files:
        print("No solver output files found!")
        return
    
    print(f"Found {len(solver_files)} solver output files")
    print(f"Generating {args.num_examples} training examples...")
    print(f"Mode: {args.mode}")
    if args.tag:
        print(f"Tag: {args.tag}")
    if args.real_time:
        print("Real-time processing: Generate → API Call → Store → Repeat")
    
    if args.real_time:
        # Real-time processing: generate and process one by one
        process_real_time(solver_files, args)
    else:
        # Batch generation: generate all examples first
        process_batch(solver_files, args)


def process_real_time(solver_files: List[str], args):
    """Process examples one by one with immediate API calls."""
    import openai
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai.api_key:
        print("❌ OPENAI_API_KEY not found in environment!")
        print("Real-time processing requires API access.")
        return
    
    examples_processed = 0
    attempts = 0
    max_attempts = args.num_examples * 10
    session_ids = set()  # Track custom_ids generated in this session
    
    # Count existing examples
    existing_count = count_jsonl_examples(args.output)
    print(f"📊 Existing examples in {args.output}: {existing_count}")
    
    # Open output file for appending
    with open(args.output, 'a') as f:
        while examples_processed < args.num_examples and attempts < max_attempts:
            attempts += 1
            
            # Pick random solver file
            solver_file = random.choice(solver_files)
            
            try:
                # Generate training example
                batch_request = generate_training_example(
                    solver_file, args.mode, args.tag, args.max_tokens, args.temperature, args.output, session_ids
                )
                
                if not batch_request:
                    continue
                
                print(f"🎯 Generated example {examples_processed + 1}/{args.num_examples} from {solver_file} (custom_id: {batch_request['custom_id']})")
                
                # Make API call
                print("🚀 Calling OpenAI API...")
                messages = batch_request['body']['messages']
                
                try:
                    # Create API call parameters
                    api_params = {
                        "model": batch_request['body']['model'],
                        "max_tokens": batch_request['body']['max_tokens'],
                        "temperature": batch_request['body']['temperature'],
                        "messages": messages
                    }
                    
                    response = openai.chat.completions.create(**api_params)
                    
                    # Extract response content
                    response_content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
                    
                    # Check for empty response
                    if not response_content:
                        print(f"⚠️  Warning: Received empty response from API (but {response.usage.completion_tokens if hasattr(response, 'usage') else 0} completion tokens)")
                        print("   This might indicate an API issue or content filtering")
                        continue
                    
                    # Create training data entry
                    training_entry = {
                        "custom_id": batch_request['custom_id'],
                        "request": {
                            "messages": messages,
                            "mode": args.mode,
                            "solver_file": solver_file
                        },
                        "response": {
                            "content": response_content,
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
                            }
                        }
                    }
                    
                    # Write to file immediately
                    json.dump(training_entry, f)
                    f.write('\n')
                    f.flush()  # Ensure it's written to disk
                    
                    # Add custom_id to session tracking
                    session_ids.add(batch_request['custom_id'])
                    
                    examples_processed += 1
                    total_count = existing_count + examples_processed
                    
                    print(f"✅ API response received and stored ({response.usage.total_tokens if hasattr(response, 'usage') else 0} tokens)")
                    print(f"📝 Response: {response_content}")
                    print(f"📊 Total examples in {args.output}: {total_count}")
                    
                except Exception as api_error:
                    print(f"❌ API Error: {api_error}")
                    print("⏭️  Skipping to next example...")
                    continue
                    
            except Exception as e:
                print(f"❌ Error processing {solver_file}: {e}")
                continue
    
    print(f"\n🎉 Completed! Processed {examples_processed} examples")
    print(f"📁 Output saved to: {args.output}")


def process_batch(solver_files: List[str], args):
    """Original batch processing: generate all examples first."""
    examples = []
    attempts = 0
    max_attempts = args.num_examples * 10  # Allow for some failures
    session_ids = set()  # Track custom_ids generated in this session
    
    # Count existing examples
    existing_count = count_jsonl_examples(args.output)
    print(f"📊 Existing examples in {args.output}: {existing_count}")
    
    while len(examples) < args.num_examples and attempts < max_attempts:
        attempts += 1
        
        # Pick random solver file
        solver_file = random.choice(solver_files)
        
        try:
            example = generate_training_example(solver_file, args.mode, args.tag, 
                                              args.max_tokens, args.temperature, args.output, session_ids)
            if example:
                examples.append(example)
                # Add custom_id to session tracking
                session_ids.add(example['custom_id'])
                print(f"Generated example {len(examples)}/{args.num_examples} from {solver_file} (custom_id: {example['custom_id']})")
        except Exception as e:
            print(f"Error processing {solver_file}: {e}")
            continue
    
    # Write to JSONL (append mode)
    with open(args.output, 'a') as f:
        for i, example in enumerate(examples):
            json.dump(example, f)
            f.write('\n')
            # Print count after each example
            total_count = existing_count + i + 1
            print(f"📊 Total examples in {args.output}: {total_count}")
    
    print(f"\nGenerated {len(examples)} training examples")
    print(f"Output saved to: {args.output}")
    print(f"📊 Final total examples in {args.output}: {existing_count + len(examples)}")


if __name__ == "__main__":
    main() 