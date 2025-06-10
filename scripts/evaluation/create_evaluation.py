#!/usr/bin/env python3
"""
SystemA vs SystemB vs SystemC Evaluation Script

This script evaluates SystemA, SystemB, and SystemC models on poker decision-making tasks.
It uses solver test outputs to generate prompts and compare model responses.
"""

import os
import json
import random
import argparse
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_data', 'systemA'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'solver'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Try to import transformers and torch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/peft not available. Model inference will be disabled.")

# Import SystemC-specific functions - try to import SystemB directly
try:
    # First try to import SystemB functions directly
    import importlib.util
    systemb_path = os.path.join(os.path.dirname(__file__), '..', 'training_data', 'systemB', 'generate_poker_training_data.py')
    systemb_path = os.path.abspath(systemb_path)
    
    # Import using importlib to avoid conflicts
    spec = importlib.util.spec_from_file_location("systemb_eval_module", systemb_path)
    systemb_module = importlib.util.module_from_spec(spec)
    
    # Temporarily suppress OpenAI import warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        spec.loader.exec_module(systemb_module)
    
    # Extract needed functions from SystemB
    get_ranges_from_preflop_chart = systemb_module.get_ranges_from_preflop_chart
    convert_board_texture = systemb_module.convert_board_texture
    analyze_hand_category = systemb_module.analyze_hand_category
    calculate_equity_and_advantages = systemb_module.calculate_equity_and_advantages
    determine_ip_oop = systemb_module.determine_ip_oop
    parse_range = systemb_module.parse_range
    hand_to_string = systemb_module.hand_to_string
        
    SYSTEMC_FUNCTIONS_AVAILABLE = True
    print("✅ Successfully imported SystemB functions for TOOL_TAGS generation")
    
except Exception as e:
    print(f"Warning: Could not import SystemB functions: {e}")
    print("SystemB/SystemC evaluation will run without TOOL_TAGS")
    get_ranges_from_preflop_chart = None
    convert_board_texture = None
    analyze_hand_category = None
    calculate_equity_and_advantages = None
    determine_ip_oop = None
    parse_range = None
    hand_to_string = None
    SYSTEMC_FUNCTIONS_AVAILABLE = False


class PokerEvaluator:
    """Main evaluation class for SystemA vs SystemB vs SystemC comparison."""
    
    def __init__(self, system: str, base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """Initialize the evaluator.
        
        Args:
            system: Either 'A', 'B', or 'C' for which system to evaluate
            base_model: Base model name for inference
        """
        self.system = system
        self.base_model_name = base_model
        self.tokenizer = None
        self.model = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the appropriate model and LoRA adapter."""
        print(f"Loading SystemA base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        # Get absolute path for LoRA adapter
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
        lora_path = os.path.join(base_dir, "out", "stageB", f"system{self.system}_bf16_r256")
        print(f"Loading LoRA adapter: {lora_path}")
        
        try:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"✅ Successfully loaded System{self.system} model")
        except Exception as e:
            print(f"❌ Error loading LoRA adapter {lora_path}: {e}")
            print("Continuing with base model only...")
    
    def get_test_files(self) -> List[str]:
        """Get all available solver test output files."""
        test_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'solver', 'solver_test_outputs')
        return [f for f in os.listdir(test_outputs_dir) if f.endswith('.json')]
    
    def generate_prompt(self, solver_file: str, hero_hand: str) -> Tuple[str, Dict[str, Any]]:
        """Generate a prompt from solver data and hero hand.
        
        Args:
            solver_file: Name of solver JSON file
            hero_hand: Hero's hole cards (e.g., "AsKh")
        
        Returns:
            Tuple of (formatted_prompt, metadata)
        """
        # Import functions locally (OpenAI warnings can be ignored)
        from generate_poker_training_data import (
            parse_filename,
            expand_position, 
            construct_preflop_history,
            determine_who_acts_first_postflop,
            get_villain_action,
            get_hero_strategy,
            format_legal_actions,
            calculate_preflop_investment
        )
        
        # Parse filename
        file_info = parse_filename(solver_file)
        
        # Load solver data
        json_data, input_data = self.load_solver_data(solver_file)
        
        # Expand positions
        pos1 = expand_position(file_info['pos1'])
        pos2 = expand_position(file_info['pos2'])
        
        # Construct preflop history
        preflop_history, hero_pos, villain_pos = construct_preflop_history(
            pos1, pos2, file_info['pot_type']
        )
        
        # Get board cards from filename if not in input_data
        board_cards = input_data.get('board', '')
        if not board_cards:
            # Extract from filename (first part before first underscore)
            board_cards = file_info['flop']
        
        # Parse board cards - they may be comma-separated from solver input
        if board_cards:
            if ',' in board_cards:
                # Board is comma-separated like "As,Kc,7s"
                flop_cards = board_cards.split(',')
            elif len(board_cards) >= 6:  # Concatenated format like "AsKc7s"
                flop_cards = [board_cards[i:i+2] for i in range(0, 6, 2)]
            else:
                flop_cards = []
        else:
            flop_cards = []
        
        # Calculate pot size from preflop action if not provided
        pot_size = input_data.get('pot')
        if pot_size is None:
            # Parse preflop action to calculate pot
            # For 3bet pots: blinds (1.5) + 3bet amount * 2
            if file_info['pot_type'] == '3bet':
                pot_size = 1.5 + 7.5 * 2  # 16.5bb for standard 3bet
            elif file_info['pot_type'] == '4bet':
                pot_size = 1.5 + 22 * 2   # 45.5bb for standard 4bet  
            else:  # SRP
                pot_size = 1.5 + 2.5 * 2  # 6.5bb for standard SRP
        
        effective_stack = input_data.get('effective_stack', 100.0)  # Default to 100bb
        
        # If effective_stack comes from solver input file, it's already the remaining stack
        # If it's the default 100.0, then we need to calculate remaining stack
        if effective_stack == 100.0:
            # Calculate hero's preflop investment and subtract from starting stack
            hero_investment = calculate_preflop_investment(preflop_history, hero_pos)
            remaining_stack = effective_stack - hero_investment
        else:
            # effective_stack from solver input is already correct
            remaining_stack = effective_stack
        
        # Get villain's action (use highest frequency action)
        villain_hand = "AsKs"  # Placeholder - this will be determined by solver
        villain_action, villain_amount = get_villain_action(
            json_data, villain_hand, villain_pos, hero_pos
        )
        
        # Determine who acts first postflop
        first_to_act, second_to_act = determine_who_acts_first_postflop(hero_pos, villain_pos)
        
        # Build villain profile with same distribution as SystemA training data
        profiles = [
            ("balanced", 0.70),
            ("tag (tight-aggressive)", 0.05),
            ("lag (loose-aggressive)", 0.05),
            ("nit (super-tight)", 0.05),
            ("station (calling-station)", 0.05),
            ("maniac (loose, hyper-aggressive)", 0.05),
            ("whale (loose-passive)", 0.05)
        ]
        
        # Weighted random selection (same logic as SystemA training data)
        r = random.random()
        cumsum = 0
        villain_profile = "balanced"  # default
        for profile, weight in profiles:
            cumsum += weight
            if r <= cumsum:
                villain_profile = profile
                break
        
        # Get legal actions for hero
        facing_action = villain_action if hero_pos == second_to_act else ""
        legal_actions = format_legal_actions(hero_pos, "flop", facing_action, file_info['pot_type'], pot_size)
        
        # Format actions with positions (like SystemC does)
        if hero_pos == second_to_act and villain_action:
            if villain_action == "check":
                actions_str = f"{villain_pos} checks."
            elif villain_action.startswith("bet"):
                # Extract bet amount and format properly
                bet_percentage = (villain_amount / pot_size) * 100
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
                    actual_bet_amount = villain_amount
                
                actions_str = f"{villain_pos} bets {actual_bet_amount:.1f}bb ({display_pct}% pot)"
            else:
                actions_str = f"{villain_pos} {villain_action}"
        else:
            actions_str = ""
        
        # Format hero hand like training data (space between cards, lowercase suits)
        if len(hero_hand) == 4:  # Format like "AsKh" -> "As kh"
            card1 = hero_hand[:2]  # "As"
            card2 = hero_hand[2:]  # "Kh"
            # Keep rank uppercase, make suit lowercase
            hero_hand_formatted = f"{card1[0]}{card1[1].lower()} {card2[0]}{card2[1].lower()}"
        else:
            hero_hand_formatted = hero_hand  # Fallback
        
        # Build base prompt sections (common to both systems)
        prompt_parts = []
        prompt_parts.append("### HAND_META")
        prompt_parts.append("game: cash")
        prompt_parts.append("seats: 8-max")
        prompt_parts.append("stacks: 100bb")
        prompt_parts.append(f"hero_pos: {hero_pos}")
        prompt_parts.append(f"hero_hand: {hero_hand_formatted}")
        prompt_parts.append(f"villain_profile: {villain_profile}")
        
        prompt_parts.append("\n### HISTORY_PREFLOP")
        prompt_parts.append(f"preflop: {preflop_history}")
        
        prompt_parts.append("\n### HISTORY 1")
        prompt_parts.append(f"flop: ({' '.join(flop_cards)})    pot: {pot_size:.1f}bb")
        prompt_parts.append(f"stacks: {remaining_stack:.1f}bb")
        prompt_parts.append(f"actions: {actions_str}")
        
        prompt_parts.append("\n### DECISION 1")
        prompt_parts.append("street: flop")
        prompt_parts.append(f"pot: {pot_size:.1f}bb")
        prompt_parts.append("to_act: HERO")
        prompt_parts.append(f"legal: [{','.join(legal_actions)}]")
        
        # Add TOOL_TAGS for SystemB and SystemC
        if self.system in ['B', 'C'] and SYSTEMC_FUNCTIONS_AVAILABLE:
            try:
                # Determine IP/OOP
                ip_pos, oop_pos = determine_ip_oop(hero_pos, villain_pos)
                
                # Get ranges from solver input
                if hero_pos == ip_pos:
                    hero_range = input_data.get('range_ip', '')
                    villain_range = input_data.get('range_oop', '')
                else:
                    hero_range = input_data.get('range_oop', '')
                    villain_range = input_data.get('range_ip', '')
                

                
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
                
                # Format hero cards for analysis (eval7 expects lowercase suits)
                hero_cards = []
                for i in range(0, 4, 2):
                    card = hero_hand[i:i+2]
                    if len(card) == 2:
                        # Convert to lowercase suit for eval7
                        hero_cards.append(card[0] + card[1].lower())
                    else:
                        hero_cards.append(card)
                
                # Also ensure board cards have lowercase suits
                board_cards_list = []
                for card in flop_cards:
                    if len(card) == 2:
                        board_cards_list.append(card[0] + card[1].lower())
                    else:
                        board_cards_list.append(card)
                

                
                # Analyze hand category
                hand_category = analyze_hand_category(hero_cards, board_cards_list)
                
                # Parse ranges to get clean version without weights
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
                
                # Calculate equity and advantages
                board_str = ' '.join(board_cards_list)
                equity_data = calculate_equity_and_advantages(
                    hero_pos, villain_pos, hero_range_clean, villain_range_clean, board_str, hero_cards
                )
                
                # Add TOOL_TAGS section
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
                
            except Exception as e:
                print(f"Warning: Could not generate TOOL_TAGS: {e}")
                import traceback
                traceback.print_exc()
                # Continue without TOOL_TAGS if there's an error
        
        prompt = "\n".join(prompt_parts)
        
        # Add instruction based on system
        if self.system == 'C':
            instruction = "\n\nChoose the best action from the legal options and provide a justification under 5 sentences. State the best action first by starting with \"**Best Action: \". Examples are **Best Action: Call**, **Best Action: Bet 33% (2.1bb)**, etc. Consider position (IP/OOP), villain profile for possible exploits, and the given tools in tool_tags about hand ranges, board texture, range advantage, nut advantage."
        elif self.system == 'A':
            instruction = "\n\nChoose the best action from the legal options and provide a justification under 5 sentences. State the best action first. Consider position (IP/OOP), board texture, range advantage, nut advantage, and villain profile for possible exploits."
        else:  # SystemB - no instruction
            instruction = ""
        
        full_prompt = prompt + instruction

        # Metadata for later analysis
        metadata = {
            'solver_file': solver_file,
            'hero_pos': hero_pos,
            'villain_pos': villain_pos,
            'hero_hand': hero_hand,
            'pot_type': file_info['pot_type'],
            'board': ' '.join(flop_cards),
            'pot_size': pot_size,
            'effective_stack': remaining_stack
        }
        
        return full_prompt, metadata
    
    def load_solver_data(self, filename: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Load solver JSON and corresponding input file from test outputs.
        
        Args:
            filename: Solver file name
        
        Returns:
            Tuple of (json_data, input_data)
        """
        # Load JSON from test outputs
        json_path = os.path.join(os.path.dirname(__file__), '..', 'solver', 'solver_test_outputs', filename)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Load corresponding input file (if exists)
        txt_filename = filename.replace('.json', '.txt')
        txt_path = os.path.join(os.path.dirname(__file__), '..', 'solver', 'solver_test_inputs', txt_filename)
        
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
        else:
            # Default values if input file doesn't exist - leave pot empty to calculate from filename
            input_data = {
                'effective_stack': 100.0,
                # Board and pot will be extracted/calculated from filename
            }
        
        return json_data, input_data
    
    def get_solver_recommendation(self, solver_file: str, hero_hand: str, metadata: Dict[str, Any]) -> str:
        """Extract solver's recommendation for the hero hand.
        
        Args:
            solver_file: Name of solver JSON file  
            hero_hand: Hero's hole cards
            metadata: Prompt metadata
        
        Returns:
            Solver recommendation string (e.g., "fold:0.7,call:0.3")
        """
        try:
            from generate_poker_training_data import get_hero_strategy
            
            json_data, _ = self.load_solver_data(solver_file)
            
            # Determine who acts first to set the correct villain_action parameter
            from generate_poker_training_data import determine_who_acts_first_postflop, get_villain_action
            first_to_act, second_to_act = determine_who_acts_first_postflop(
                metadata['hero_pos'], metadata['villain_pos']
            )
            
            # Set villain_action based on who acts first
            if metadata['hero_pos'] == first_to_act:
                villain_action_param = "first_to_act"
            else:
                # If villain acts first, get their actual action from solver data
                villain_hand = "AsKs"  # Placeholder hand for villain action lookup
                villain_action, _ = get_villain_action(
                    json_data, villain_hand, metadata['villain_pos'], metadata['hero_pos']
                )
                villain_action_param = villain_action
            
            hero_strategy = get_hero_strategy(
                json_data, hero_hand, metadata['hero_pos'], metadata['villain_pos'], villain_action_param
            )
            
            # If no strategy found for specific hand, try to get a representative strategy
            if not hero_strategy and villain_action_param != "first_to_act":
                # Try to get strategy from any available hand as a fallback
                current_node = json_data
                if villain_action_param == "check" and 'childrens' in current_node and 'CHECK' in current_node['childrens']:
                    check_node = current_node['childrens']['CHECK']
                    if 'strategy' in check_node and 'strategy' in check_node['strategy']:
                        available_hands = list(check_node['strategy']['strategy'].keys())
                        if available_hands:
                            # Use the first available hand as a representative
                            representative_hand = available_hands[0]
                            actions = check_node['strategy']['actions']
                            probs = check_node['strategy']['strategy'][representative_hand]
                            hero_strategy = dict(zip(actions, probs))
            
            # Format as action:probability pairs
            recommendations = []
            for action, prob in hero_strategy.items():
                if prob > 0:
                    recommendations.append(f"{action}:{prob:.3f}")
            
            return ",".join(recommendations)
            
        except Exception as e:
            print(f"Warning: Could not extract solver recommendation: {e}")
            return "unknown"
    
    def extract_action(self, completion: str) -> str:
        """Extract the recommended action from model completion.
        
        Args:
            completion: Model's completion text
            
        Returns:
            Extracted action (e.g., "fold", "bet_50", etc.)
        """
        # Look for various action patterns in ** ** format
        patterns = [
            r'\*\*Best Action:\s*([^*\n]+)\*\*',
            r'\*\*Action:\s*([^*\n]+)\*\*',
            r'\*\*([^*\n]*(?:fold|call|bet|raise|check|all-in|allin)[^*\n]*)\*\*'  # Any action in ** **
        ]
        
        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if match:
                action = match.group(1).strip().lower()
                return action
        
        # Fallback: look for common action keywords at start of completion
        completion_lower = completion.lower()
        action_keywords = ['fold', 'call', 'bet', 'raise', 'check', 'all-in', 'allin']
        
        for keyword in action_keywords:
            if completion_lower.startswith(keyword):
                return keyword
        
        return "unknown"
    
    def generate_completion(self, prompt: str) -> str:
        """Generate completion from the model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model completion
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return f"[System{self.system} completion would go here - model not loaded]"
        
        # Format with chat template
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        completion = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return completion.strip()
    
    def run_evaluation(self, num_runs: int, output_file: str):
        """Run the evaluation process.
        
        Args:
            num_runs: Number of evaluation runs
            output_file: Output JSONL file path (will append if exists)
        """
        # Get available test files
        test_files = self.get_test_files()
        print(f"Found {len(test_files)} test files")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if file exists to determine append vs create
        if os.path.exists(output_file):
            print(f"📂 Output file exists, will append to: {output_file}")
        else:
            print(f"📝 Creating new output file: {output_file}")
        
        completed_runs = 0
        
        for i in range(num_runs):
            print(f"\n--- Run {i+1}/{num_runs} ---")
            
            # Random file and hero hand selection
            solver_file = random.choice(test_files)
            print(f"Selected solver file: {solver_file}")
            
            # Load solver data to get available hands
            json_data, _ = self.load_solver_data(solver_file)
            
            # Get available hands from solver data
            available_hands = []
            if 'strategy' in json_data and 'strategy' in json_data['strategy']:
                available_hands = list(json_data['strategy']['strategy'].keys())
            
            if available_hands:
                # Choose random hand from available hands in solver
                hero_hand = random.choice(available_hands)
                print(f"Hero hand: {hero_hand} (chosen from {len(available_hands)} available hands)")
            else:
                # Fallback to generating random hand
                ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
                suits = ['s', 'h', 'd', 'c']
                
                card1 = random.choice(ranks) + random.choice(suits)
                card2 = random.choice(ranks) + random.choice(suits)
                while card1 == card2:
                    card2 = random.choice(ranks) + random.choice(suits)
                
                hero_hand = card1 + card2
                print(f"Hero hand: {hero_hand} (fallback - no hands found in solver)")
            
            # Generate prompt
            prompt, metadata = self.generate_prompt(solver_file, hero_hand)
            
            # Get solver recommendation
            solver_rec = self.get_solver_recommendation(solver_file, hero_hand, metadata)
            
            # Generate completion for this system
            completion = self.generate_completion(prompt)
            recommended_action = self.extract_action(completion)
            
            print(f"System{self.system} completion: {completion[:100]}...")
            print(f"Extracted action: {recommended_action}")
            
            # Create entry
            entry = {
                'prompt': prompt,
                'solver_recommendation': solver_rec,
                'metadata': metadata,
                f"system{self.system}": {
                    'completion': completion,
                    'recommended_action': recommended_action
                }
            }
            
            completed_runs += 1
            
            # Append result immediately
            print(f"💾 Writing result {completed_runs} to file...")
            with open(output_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Evaluation complete! Completed {completed_runs} runs total.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SystemA vs SystemB vs SystemC Poker Evaluation')
    parser.add_argument('--system', choices=['A', 'B', 'C'], required=True,
                       help='Which system to evaluate (A, B, or C)')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of evaluation runs (default: 10)')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                       help='Output JSONL file (will append if exists, create if not)')
    
    args = parser.parse_args()
    
    print(f"🃏 SystemA vs SystemB vs SystemC Poker Evaluation")
    print(f"System: {args.system}")
    print(f"Runs: {args.runs}")
    print(f"Output: {args.output_file}")
    
    # Create evaluator and run
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    evaluator = PokerEvaluator(args.system, base_model)
    evaluator.run_evaluation(args.runs, args.output_file)


if __name__ == "__main__":
    main() 