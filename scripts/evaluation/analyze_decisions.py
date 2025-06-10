#!/usr/bin/env python3
"""
Analyze Poker Decision Results
Shows solver recommendations vs LLM decisions with frequencies and scores
"""

import json
import argparse
import re
import math
import os
from typing import Dict, List, Tuple, Optional

# Import scoring and parsing functions from evaluation.py for consistency
from evaluation import (
    generalized_sigmoid_score, 
    parse_solver_recommendation, 
    parse_llm_action, 
    find_closest_solver_action
)

def format_solver_recommendation(solver_rec: str) -> str:
    """Format solver recommendation to show only 2 decimal places"""
    import re
    
    # Pattern to match action:frequency pairs
    def replace_numbers(match):
        action_part = match.group(1)  # e.g., "CHECK" or "BET 2.000000"
        frequency = float(match.group(2))
        
        # Format bet sizes to 2 decimal places
        if action_part.startswith("BET") or action_part.startswith("RAISE"):
            # Extract and format the bet size
            bet_match = re.search(r'(BET|RAISE)\s+([\d.]+)', action_part)
            if bet_match:
                action_type = bet_match.group(1)
                bet_size = float(bet_match.group(2))
                formatted_action = f"{action_type} {bet_size:.2f}"
            else:
                formatted_action = action_part
        else:
            formatted_action = action_part
        
        return f"{formatted_action}:{frequency:.2f}"
    
    # Replace all action:frequency pairs
    formatted = re.sub(r'([^,]+):([\d.]+)', replace_numbers, solver_rec)
    return formatted

def analyze_system(system: str, max_entries: int = 200) -> None:
    """Analyze decisions for a specific system"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "results", f"system{system}_evaluation_results.jsonl")
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        return
    
    print(f"SYSTEM {system} DECISION ANALYSIS")
    print("=" * 150)
    print(f"{'Entry':<5} {'Solver Recommendation':<70} {'LLM Action':<25} {'Frequency':<10} {'Score':<8}")
    print("-" * 150)
    
    entry_count = 0
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if entry_count >= max_entries:
                    break
                    
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract fields
                    solver_rec = entry.get('solver_recommendation', '')
                    system_data = entry.get(f'system{system}', {})
                    llm_action = system_data.get('recommended_action', '')
                    
                    if not llm_action or not solver_rec:
                        continue
                    
                    # Use evaluation.py functions for consistency
                    solver_actions = parse_solver_recommendation(solver_rec)
                    pot_size = entry.get('metadata', {}).get('pot_size', 0.0)
                    llm_action_type, llm_bet_size = parse_llm_action(llm_action, pot_size)
                    closest_action, llm_freq = find_closest_solver_action(
                        llm_action_type, llm_bet_size, solver_actions
                    )
                    
                    # Handle no match case
                    if closest_action == "no_match":
                        llm_freq = 0.0
                    
                    # Calculate score using evaluation.py function
                    score = generalized_sigmoid_score(llm_freq)
                    
                    # Format solver recommendation with 2 decimal places
                    formatted_solver = format_solver_recommendation(solver_rec)
                    display_llm = llm_action[:22] + "..." if len(llm_action) > 25 else llm_action
                    
                    if len(formatted_solver) > 70:
                        # Show entry info first, then formatted solver rec on next line
                        print(f"{entry_count+1:<5} {'':>70} {display_llm:<25} {llm_freq:<10.3f} {score:<8.3f}")
                        print(f"      {formatted_solver}")
                    else:
                        # Normal display with formatted solver recommendation
                        print(f"{entry_count+1:<5} {formatted_solver:<70} {display_llm:<25} {llm_freq:<10.3f} {score:<8.3f}")
                    
                    entry_count += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find file {results_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print("-" * 150)
    print(f"Showing {entry_count} entries")

def main():
    parser = argparse.ArgumentParser(description="Analyze poker decision results with frequencies and scores")
    parser.add_argument('system', choices=['A', 'B', 'C'], help='System to analyze (A, B, or C)')
    parser.add_argument('--entries', type=int, default=400, help='Maximum number of entries to show (default: 50)')
    
    args = parser.parse_args()
    
    analyze_system(args.system, args.entries)

if __name__ == "__main__":
    main() 