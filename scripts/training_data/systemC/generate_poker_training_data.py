#!/usr/bin/env python3
"""
Generate poker training data for System C with TOOL_TAGS + GPT-4o completions.

This script combines System B's TOOL_TAGS prompt generation with System A's OpenAI API handling.
It creates training examples with detailed tool information and gets completions from GPT-4o.
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import importlib.util

# Try to import OpenAI - it's required for SystemC
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
    print("Error: openai package not found. SystemC requires OpenAI API access.")
    exit(1)

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

# Import all the SystemB functions we need
def import_systemb_functions():
    """Import SystemB functions avoiding naming conflicts"""
    try:
        # Get the path to systemB's generate_poker_training_data.py
        systemb_path = os.path.join(os.path.dirname(__file__), '..', 'systemB', 'generate_poker_training_data.py')
        systemb_path = os.path.abspath(systemb_path)
        
        # Import using importlib
        spec = importlib.util.spec_from_file_location("systemb_module", systemb_path)
        systemb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(systemb_module)
        
        return systemb_module
        
    except Exception as e:
        print(f"Error importing SystemB functions: {e}")
        return None

# Import SystemB functions
systemb = import_systemb_functions()

# Extract the functions we need
if systemb:
    construct_preflop_history = systemb.construct_preflop_history
    determine_who_acts_first_postflop = systemb.determine_who_acts_first_postflop
    load_solver_data = systemb.load_solver_data
    get_ranges_from_preflop_chart = systemb.get_ranges_from_preflop_chart
    convert_board_texture = systemb.convert_board_texture
    analyze_hand_category = systemb.analyze_hand_category
    calculate_equity_and_advantages = systemb.calculate_equity_and_advantages
    get_villain_action = systemb.get_villain_action
    format_legal_actions = systemb.format_legal_actions
    get_hero_strategy = systemb.get_hero_strategy
    convert_solver_action_to_token = systemb.convert_solver_action_to_token
    calculate_preflop_investment = systemb.calculate_preflop_investment
else:
    print("ERROR: Could not import SystemB functions!")
    exit(1)

def generate_systemc_prompt(solver_file: str, tag: str = '', session_ids: set = None) -> Optional[Dict[str, Any]]:
    """Generate a SystemC training prompt using SystemB's TOOL_TAGS method."""
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
    
    # Get legal actions based on what hero is facing
    legal_actions = format_legal_actions(hero_pos, 'flop', hero_facing_action, file_info['pot_type'], pot_size)
    
    # Format the prompt with TOOL_TAGS (SystemB style)
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
    
    # Add the user's specific instruction after TOOL_TAGS (with two newlines)
    instruction = "\n\nChoose the best action from the legal options and provide a justification under 5 sentences. State the best action first by starting with \"**Best Action: \". Examples are **Best Action: Call**, **Best Action: Bet 33% (2.1bb)**, etc. Consider position (IP/OOP), villain profile for possible exploits, and the given tools in tool_tags about hand ranges, board texture, range advantage, nut advantage."
    
    full_prompt = prompt + instruction
    
    # Generate unique custom_id
    if session_ids is None:
        session_ids = set()
    
    custom_id = generate_unique_custom_id(solver_file, set(), tag, session_ids)
    
    return {
        "custom_id": custom_id,
        "prompt": full_prompt,
        "solver_file": solver_file
    }

def make_openai_request(prompt: str, custom_id: str) -> Dict[str, Any]:
    """Make OpenAI API request (adapted from SystemA)."""
    system_message = "You are a world-class poker professional with deep understanding of game theory optimal (GTO) play, opponent profiling, and advanced poker strategy. You excel at providing concise, strategic justifications for poker decisions."
    
    try:
        # Create API call parameters
        api_params = {
            "model": "gpt-4o",
            "max_tokens": 150,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        }
        
        response = openai.chat.completions.create(**api_params)
        
        # Extract response content
        response_content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        # Check for empty response
        if not response_content:
            print(f"⚠️  Warning: Received empty response from API")
            return None
        
        return {
            "custom_id": custom_id,
            "request": {
                "messages": api_params["messages"],
                "mode": "systemC"
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
        
    except Exception as api_error:
        print(f"❌ API Error: {api_error}")
        return None

def main():
    """Main function to generate SystemC training data."""
    parser = argparse.ArgumentParser(description='Generate SystemC poker training data with TOOL_TAGS + GPT-4o')
    parser.add_argument('-n', '--num-examples', type=int, default=10,
                        help='Number of examples to generate')
    parser.add_argument('-o', '--output', type=str, default='systemC_training_data.jsonl',
                        help='Output JSONL file')
    parser.add_argument('-t', '--tag', type=str, default='',
                        help='Tag to prepend to custom_ids')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing file instead of appending (default: append)')
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not openai.api_key:
        print("❌ OPENAI_API_KEY not found in environment!")
        print("SystemC requires OpenAI API access.")
        return
    
    # Get solver files
    solver_files = get_solver_files()
    if not solver_files:
        print("No solver output files found!")
        return
    
    print(f"Found {len(solver_files)} solver output files")
    print(f"Generating {args.num_examples} SystemC training examples...")
    print("SystemC: SystemB TOOL_TAGS + SystemA OpenAI API + GPT-4o completions")
    
    # Track session IDs to avoid duplicates within this run
    session_ids = set()
    
    # Count existing examples
    existing_count = count_jsonl_examples(args.output)
    if not args.overwrite:
        print(f"📊 Existing examples in {args.output}: {existing_count}")
    
    # Open file in append or write mode
    mode = 'w' if args.overwrite else 'a'
    
    # Generate and process examples one at a time (SystemA real-time style)
    examples_processed = 0
    attempts = 0
    max_attempts = args.num_examples * 3  # Allow for some failures
    
    with open(args.output, mode) as f:
        while examples_processed < args.num_examples and attempts < max_attempts:
            attempts += 1
            
            # Pick random solver file
            solver_file = random.choice(solver_files)
            
            try:
                # Generate SystemC prompt (SystemB style with TOOL_TAGS)
                prompt_data = generate_systemc_prompt(solver_file, args.tag, session_ids)
                if not prompt_data:
                    continue
                
                print(f"🎯 Generated prompt {examples_processed + 1}/{args.num_examples} from {solver_file} (custom_id: {prompt_data['custom_id']})")
                
                # Make OpenAI API call (SystemA style)
                print("🚀 Calling GPT-4o API...")
                
                training_entry = make_openai_request(prompt_data['prompt'], prompt_data['custom_id'])
                if not training_entry:
                    print("⏭️  Skipping due to API error...")
                    continue
                
                # Write to file immediately and flush (SystemB style)
                f.write(json.dumps(training_entry) + '\n')
                f.flush()  # Force write to disk
                
                # Add custom_id to session tracking
                session_ids.add(prompt_data['custom_id'])
                
                examples_processed += 1
                total_count = existing_count + examples_processed if not args.overwrite else examples_processed
                
                print(f"✅ API response received and stored ({training_entry['response']['usage']['total_tokens']} tokens)")
                print(f"📝 Response: {training_entry['response']['content'][:100]}...")
                print(f"📊 Total examples in {args.output}: {total_count}")
                
            except Exception as e:
                print(f"❌ Error processing {solver_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    final_count = count_jsonl_examples(args.output)
    print(f"\n🎉 Completed! Generated {examples_processed} new examples")
    print(f"📁 Output saved to: {args.output}")
    print(f"📊 Final total examples in {args.output}: {final_count}")

if __name__ == "__main__":
    main() 