#!/usr/bin/env python3
"""
Poker Evaluation Script using Generalized Sigmoid Scoring

Evaluates SystemA, SystemB, or SystemC against solver recommendations
using the generalized sigmoid function: g(x) = 1 / (1 + e^(-k(x-μ)))^β
"""

import json
import argparse
import re
import math
import os
import random
from typing import Dict, List, Tuple, Optional

def generalized_sigmoid_score(frequency: float, k: float = 6, u: float = 0.5) -> float:
    """
    Simplified sigmoid scoring function with normalization
    
    g(x) = 1 / (1 + e^(-k(x-u)))
    f(x) = (g(x) - g(0)) / (g(1) - g(0))  # Normalized to [0,1]
    
    Args:
        frequency: Action frequency from solver (0.0 to 1.0)
        k: Steepness parameter (default: 6)
        u: Midpoint parameter (default: 0.5) 
    
    Returns:
        Normalized score between 0 and 1
    """
    def g(x):
        return 1 / (1 + math.exp(-k * (x - u)))
    
    # Calculate g(x), g(0), g(1)
    g_x = g(frequency)
    g_0 = g(0.0)
    g_1 = g(1.0)
    
    # Normalize to ensure f(0)=0 and f(1)=1
    normalized_score = (g_x - g_0) / (g_1 - g_0)
    return max(0.0, min(1.0, normalized_score))  # Clamp to [0,1]

def parse_solver_recommendation(solver_rec: str) -> Dict[str, float]:
    """
    Parse solver recommendation string into action->frequency mapping
    
    Format: "CHECK:0.000,BET 2.000000:0.380,BET 3.000000:0.578,BET 7.000000:0.042"
    
    Returns:
        Dict mapping action to frequency, e.g., {"check": 0.0, "bet_2.0": 0.38, ...}
    """
    actions = {}
    
    try:
        for action_freq in solver_rec.split(','):
            if ':' not in action_freq:
                continue
                
            action_part, freq_part = action_freq.split(':', 1)
            action_part = action_part.strip()
            frequency = float(freq_part.strip())
            
            # Normalize action names
            if action_part.upper() == "CHECK":
                actions["check"] = frequency
            elif action_part.upper() == "FOLD":
                actions["fold"] = frequency
            elif action_part.upper() == "CALL":
                actions["call"] = frequency
            elif action_part.upper().startswith("BET"):
                # Extract bet size from "BET 2.000000" -> "bet_2.0"
                bet_match = re.search(r'BET\s+([\d.]+)', action_part.upper())
                if bet_match:
                    bet_size = float(bet_match.group(1))
                    actions[f"bet_{bet_size:.1f}"] = frequency
            elif action_part.upper().startswith("RAISE"):
                # Handle raise actions similarly
                raise_match = re.search(r'RAISE\s+([\d.]+)', action_part.upper())
                if raise_match:
                    raise_size = float(raise_match.group(1))
                    actions[f"raise_{raise_size:.1f}"] = frequency
                    
    except (ValueError, IndexError) as e:
        print(f"Warning: Failed to parse solver recommendation: {solver_rec[:50]}... Error: {e}")
    
    return actions

def parse_llm_action(recommended_action: str, pot_size: float = 0.0) -> Tuple[str, Optional[float]]:
    """
    Parse LLM recommended action and extract action type and bet size
    
    Examples:
        "check" -> ("check", None)
        "bet 33% (2.1bb)" -> ("bet", 2.1)
        "bet 33%" -> ("bet", 0.33 * pot_size)  # Calculate from pot
        "raise 100%" -> ("raise", 1.0 * pot_size)  # Calculate from pot
        "fold" -> ("fold", None)
        "call" -> ("call", None)
    
    Returns:
        Tuple of (action_type, bet_size)
    """
    if not recommended_action:
        return ("unknown", None)
    
    action = recommended_action.strip().lower()
    
    # Check for simple actions first
    if "check" in action:
        return ("check", None)
    elif "fold" in action:
        return ("fold", None)
    elif "call" in action:
        return ("call", None)
    
    # Check for bet/raise with size - extract the bb value
    # Pattern to match: "bet 33% (2.1bb)" or "bet 2.1bb"
    bet_match = re.search(r'bet.*?(?:\((\d+(?:\.\d+)?)bb\)|(\d+(?:\.\d+)?)bb)', action)
    if bet_match:
        # Try to extract bet size in bb - prioritize parentheses format
        if bet_match.group(1):  # Format like "bet 33% (2.1bb)"
            bet_size = float(bet_match.group(1))
        elif bet_match.group(2):  # Format like "bet 2.1bb"
            bet_size = float(bet_match.group(2))
        else:
            return ("unknown", None)
        return ("bet", bet_size)
    
    # If no bb found, try to find percentage (like "bet 33%")
    bet_percent_match = re.search(r'bet.*?(\d+(?:\.\d+)?)%', action)
    if bet_percent_match:
        percentage = float(bet_percent_match.group(1)) / 100.0
        bet_size = percentage * pot_size if pot_size > 0 else 0.0
        return ("bet", bet_size)
    
    # If still no match, try any number
    bet_match_simple = re.search(r'bet.*?(\d+(?:\.\d+)?)', action)
    if bet_match_simple:
        number = float(bet_match_simple.group(1))
        return ("bet", number)
    
    # Similar pattern for raise
    raise_match = re.search(r'raise.*?(?:\((\d+(?:\.\d+)?)bb\)|(\d+(?:\.\d+)?)bb)', action)
    if raise_match:
        if raise_match.group(1):
            raise_size = float(raise_match.group(1))
        elif raise_match.group(2):
            raise_size = float(raise_match.group(2))
        else:
            return ("unknown", None)
        return ("raise", raise_size)
    
    # Check for raise percentage
    raise_percent_match = re.search(r'raise.*?(\d+(?:\.\d+)?)%', action)
    if raise_percent_match:
        percentage = float(raise_percent_match.group(1)) / 100.0
        raise_size = percentage * pot_size if pot_size > 0 else 0.0
        return ("raise", raise_size)
    
    # If no bb found for raise, try simple number extraction
    raise_match_simple = re.search(r'raise.*?(\d+(?:\.\d+)?)', action)
    if raise_match_simple:
        number = float(raise_match_simple.group(1))
        return ("raise", number)
    
    # If we can't parse it, return the original action
    return (action.split()[0] if action.split() else "unknown", None)

def find_closest_solver_action(llm_action_type: str, llm_bet_size: Optional[float], 
                              solver_actions: Dict[str, float]) -> Tuple[str, float]:
    """
    Find the closest matching solver action for the LLM's chosen action
    
    Note: Check action is completely separate from bet actions (not bet 0)
    
    Returns:
        Tuple of (closest_action, frequency)
    """
    # Direct matches for simple actions (check, fold, call)
    if llm_action_type in ["check", "fold", "call"]:
        if llm_action_type in solver_actions:
            return (llm_action_type, solver_actions[llm_action_type])
        else:
            return ("no_match", 0.0)
    
    # For bet/raise actions, find closest bet size using absolute difference
    if llm_action_type in ["bet", "raise"] and llm_bet_size is not None:
        action_prefix = f"{llm_action_type}_"
        
        # Find all solver actions with this prefix
        matching_actions = {k: v for k, v in solver_actions.items() if k.startswith(action_prefix)}
        
        if not matching_actions:
            return ("no_match", 0.0)
                
        # Find closest bet size using absolute difference
        best_action = None
        best_diff = float('inf')
        
        for action, freq in matching_actions.items():
            try:
                # Extract bet size from "bet_2.0" or "raise_3.5"
                solver_bet_size = float(action.split('_')[1])
                diff = abs(solver_bet_size - llm_bet_size)
                
                if diff < best_diff:
                    best_diff = diff
                    best_action = action
                    
            except (ValueError, IndexError):
                continue
        
        if best_action:
            return (best_action, solver_actions[best_action])
                
    # If no match found, return 0 frequency
    return ("no_match", 0.0)

def get_max_frequency_action(solver_actions: Dict[str, float]) -> float:
    """Get the frequency of the action with highest frequency"""
    if not solver_actions:
        return 0.0
    return max(solver_actions.values())

def get_random_action_score(solver_actions: Dict[str, float]) -> float:
    """Get score for a randomly selected action from solver recommendations"""
    if not solver_actions:
        return 0.0
    
    # Randomly pick one of the available actions
    random_action = random.choice(list(solver_actions.keys()))
    random_freq = solver_actions[random_action]
    
    return generalized_sigmoid_score(random_freq)

def evaluate_system(system: str) -> Dict[str, float]:
    """
    Evaluate a specific system (A, B, or C)
        
        Returns:
        Dictionary with evaluation metrics
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
            
    # Construct file path relative to script location
    results_file = os.path.join(script_dir, "results", f"system{system}_evaluation_results.jsonl")
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        return {}
    
    print(f"Evaluating System {system}...")
    print(f"Reading from: {results_file}")
    
    total_score = 0.0
    max_possible_score = 0.0
    random_baseline_score = 0.0
    valid_entries = 0
    
    # Set random seed for reproducible random baseline
    random.seed(42)
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract required fields
                    solver_rec = entry.get('solver_recommendation', '')
                    
                    # Get LLM action from the correct system field
                    system_data = entry.get(f'system{system}', {})
                    llm_action = system_data.get('recommended_action', '')
                    
                    if not llm_action:
                        print(f"Warning: Missing recommended_action at line {line_num}")
                        continue
                    
                    # Parse solver recommendations
                    solver_actions = parse_solver_recommendation(solver_rec)
                    if not solver_actions:
                        print(f"Warning: Could not parse solver recommendation at line {line_num}")
                        continue
                    
                    # Parse LLM action
                    pot_size = entry.get('metadata', {}).get('pot_size', 0.0)
                    llm_action_type, llm_bet_size = parse_llm_action(llm_action, pot_size)
        
                    # Find closest solver action
                    closest_action, llm_freq = find_closest_solver_action(
                        llm_action_type, llm_bet_size, solver_actions
                    )
                    
                    # If no match found, use frequency 0.0 instead of skipping
                    if closest_action == "no_match":
                        llm_freq = 0.0
                        closest_action = f"no_match_{llm_action_type}"
                    
                    # Calculate scores
                    llm_score = generalized_sigmoid_score(llm_freq)
                    max_freq = get_max_frequency_action(solver_actions)
                    max_score = generalized_sigmoid_score(max_freq)
                    random_score = get_random_action_score(solver_actions)
                    
                    # Accumulate scores
                    total_score += llm_score
                    max_possible_score += max_score
                    random_baseline_score += random_score
                    valid_entries += 1
                    
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find file {results_file}")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    # Calculate final metrics
    if valid_entries == 0:
        print("No valid entries found!")
        return {}
    
    avg_score = total_score / valid_entries
    avg_max_score = max_possible_score / valid_entries
    avg_random_score = random_baseline_score / valid_entries
    relative_performance = (avg_score / avg_max_score * 100) if avg_max_score > 0 else 0
    random_performance = (avg_random_score / avg_max_score * 100) if avg_max_score > 0 else 0
    
    return {
        'system': system,
        'valid_entries': valid_entries,
        'average_score': avg_score,
        'average_max_score': avg_max_score,
        'average_random_score': avg_random_score,
        'relative_performance_pct': relative_performance,
        'random_baseline_pct': random_performance,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate poker decision systems using generalized sigmoid scoring")
    parser.add_argument('system', choices=['A', 'B', 'C'], help='System to evaluate (A, B, or C)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Evaluate the system
    results = evaluate_system(args.system)
    
    if not results:
        print("Evaluation failed!")
        return 1
    
    # Display basic results (without percentages)
    print(f"\nSYSTEM {results['system']} EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Valid entries processed: {results['valid_entries']:,}")
    print(f"Average LLM score:      {results['average_score']:.4f}")
    print(f"Average random score:   {results['average_random_score']:.4f}")
    print(f"Average max score:      {results['average_max_score']:.4f}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 