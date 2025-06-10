import eval7
import random
import json
import os
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time
from datetime import datetime

# Add solver directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solver'))

# Import from preflop scenario generator
from preflop_scenario_generator import (
    load_preflop_chart,
    parse_scenario,
    determine_ip_oop,
    POSITION_GROUPS,
    expand_range
)

# Import from equity calculator
from equity_calculator import (
    fix_range_notation,
    parse,
    calculate_range_equity
)


def generate_all_scenarios() -> List[Tuple[str, Dict[str, str]]]:
    """Generate all possible scenarios from the preflop chart.
    
    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        List of (scenario_description, ranges_dict) tuples
    """
    chart = load_preflop_chart()
    scenarios = []
    
    # Process RFI scenarios
    rfi_block = "Cash, 100bb, 8-max, RFI"
    if rfi_block in chart:
        rfi_ranges = chart[rfi_block]
        
        # For each RFI position, generate scenarios with all possible callers
        for raiser_pos, raiser_range in rfi_ranges.items():
            raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
            
            # Look for calling ranges for this raiser
            call_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, call"
            if call_block in chart:
                for caller_pos, caller_range in chart[call_block].items():
                    if caller_pos != raiser_pos:  # Can't call yourself
                        scenario_desc = f"{raiser_pos} raise, {caller_pos} call"
                        ranges = {
                            raiser_pos: raiser_range,
                            caller_pos: caller_range
                        }
                        scenarios.append((scenario_desc, ranges))
            
            # Look for 3betting ranges for this raiser
            threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
            if threbet_block in chart:
                for threebetter_pos, threebetter_range in chart[threbet_block].items():
                    if threebetter_pos != raiser_pos:
                        scenario_desc = f"{raiser_pos} raise, {threebetter_pos} 3bet"
                        ranges = {
                            raiser_pos: raiser_range,
                            threebetter_pos: threebetter_range
                        }
                        scenarios.append((scenario_desc, ranges))
    
    # Process 3bet pot scenarios
    # For 3bet pots, we need to handle the original raiser's response
    for block_name, block_ranges in chart.items():
        if "3bet, OOP, call" in block_name:
            # OOP player called a 3bet
            for pos, calling_range in block_ranges.items():
                # We need to find who 3bet - this is complex without full context
                # For now, we'll create generic scenarios
                scenario_desc = f"3bet pot, {pos} calls OOP"
                scenarios.append((scenario_desc, {pos: calling_range}))
        
        elif "3bet, IP, call" in block_name:
            # IP player called a 3bet
            for pos, calling_range in block_ranges.items():
                scenario_desc = f"3bet pot, {pos} calls IP"
                scenarios.append((scenario_desc, {pos: calling_range}))
    
    # Process 4bet scenarios
    for block_name, block_ranges in chart.items():
        if "4bet" in block_name and ("call" in block_name or "allin" in block_name):
            position_type = "IP" if "IP" in block_name else "OOP"
            action_type = "call" if "call" in block_name else "allin"
            
            if "ALL" in block_ranges:
                # Universal range for this action
                scenario_desc = f"4bet pot, {position_type} {action_type}"
                scenarios.append((scenario_desc, {"Player": block_ranges["ALL"]}))
            else:
                for pos, range_str in block_ranges.items():
                    scenario_desc = f"4bet pot, {pos} {action_type} {position_type}"
                    scenarios.append((scenario_desc, {pos: range_str}))
    
    return scenarios


def build_complete_scenarios() -> List[Dict[str, Any]]:
    """Build complete two-player scenarios with proper range assignments.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of scenario dictionaries with full information
    """
    chart = load_preflop_chart()
    complete_scenarios = []
    
    # Define all positions in order
    all_positions = ['UTG', 'UTG+1', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    
    # RFI positions
    rfi_positions = ['UTG', 'UTG+1', 'LJ', 'HJ', 'CO', 'BTN', 'SB']
    
    # Get RFI block
    rfi_block = "Cash, 100bb, 8-max, RFI"
    if rfi_block not in chart:
        print("Error: RFI block not found in chart")
        return complete_scenarios
    
    # 1. Single Raised Pots (SRP) - 24 scenarios
    for raiser_pos in rfi_positions:
        if raiser_pos not in chart[rfi_block]:
            continue
            
        raiser_range = chart[rfi_block][raiser_pos]
        raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
        
        # Find the appropriate calling block
        call_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, call"
        
        if call_block in chart:
            # Get all possible callers from the block
            for chart_caller, caller_range in chart[call_block].items():
                # MP in the chart means both LJ and HJ can call
                if chart_caller == 'MP' and raiser_group == 'EP':
                    # MP can only call EP raises, and represents both LJ and HJ
                    for actual_caller in ['LJ', 'HJ']:
                        scenario = {
                            'description': f"{raiser_pos} raise, {actual_caller} call",
                            'scenario_type': 'single_raised_pot',
                            'positions': {
                                'raiser': raiser_pos,
                                'caller': actual_caller
                            },
                            'ranges': {
                                raiser_pos: raiser_range,
                                actual_caller: caller_range
                            }
                        }
                        complete_scenarios.append(scenario)
                else:
                    # For all other positions, check if they come after raiser
                    if chart_caller in all_positions:
                        raiser_idx = all_positions.index(raiser_pos)
                        caller_idx = all_positions.index(chart_caller)
                        if caller_idx > raiser_idx:
                            scenario = {
                                'description': f"{raiser_pos} raise, {chart_caller} call",
                                'scenario_type': 'single_raised_pot',
                                'positions': {
                                    'raiser': raiser_pos,
                                    'caller': chart_caller
                                },
                                'ranges': {
                                    raiser_pos: raiser_range,
                                    chart_caller: caller_range
                                }
                            }
                            complete_scenarios.append(scenario)
    
    # 2. 3bet Pots - 24 scenarios
    for raiser_pos in rfi_positions:
        if raiser_pos not in chart[rfi_block]:
            continue
            
        raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
        
        # Find the 3bet block for this raiser
        threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
        
        if threbet_block in chart:
            for chart_3better, threebetter_range in chart[threbet_block].items():
                # Handle MP 3betters (LJ and HJ)
                if chart_3better == 'MP' and raiser_group == 'EP':
                    for actual_3better in ['LJ', 'HJ']:
                        # Determine IP/OOP for calling range
                        ip_pos, oop_pos = determine_ip_oop(raiser_pos, actual_3better)
                        
                        # Get the appropriate calling range for the raiser
                        if raiser_pos == ip_pos:
                            call_block = "Cash, 100bb, 8-max, 3bet, IP, call"
                        else:
                            call_block = "Cash, 100bb, 8-max, 3bet, OOP, call"
                        
                        caller_range = None
                        if call_block in chart:
                            # Try exact position first
                            if raiser_pos in chart[call_block]:
                                caller_range = chart[call_block][raiser_pos]
                            # Then try position group
                            elif raiser_group in chart[call_block]:
                                caller_range = chart[call_block][raiser_group]
                        
                        if caller_range:
                            scenario = {
                                'description': f"{raiser_pos} raise, {actual_3better} 3bet, {raiser_pos} call",
                                'scenario_type': '3bet_pot',
                                'positions': {
                                    'original_raiser': raiser_pos,
                                    '3better': actual_3better
                                },
                                'ranges': {
                                    raiser_pos: caller_range,  # Raiser's calling range
                                    actual_3better: threebetter_range
                                }
                            }
                            complete_scenarios.append(scenario)
                else:
                    # For all other positions
                    if chart_3better in all_positions:
                        raiser_idx = all_positions.index(raiser_pos)
                        threebetter_idx = all_positions.index(chart_3better)
                        if threebetter_idx > raiser_idx:
                            # Determine IP/OOP
                            ip_pos, oop_pos = determine_ip_oop(raiser_pos, chart_3better)
                            
                            # Get calling range
                            if raiser_pos == ip_pos:
                                call_block = "Cash, 100bb, 8-max, 3bet, IP, call"
                            else:
                                call_block = "Cash, 100bb, 8-max, 3bet, OOP, call"
                            
                            caller_range = None
                            if call_block in chart:
                                if raiser_pos in chart[call_block]:
                                    caller_range = chart[call_block][raiser_pos]
                                elif raiser_group in chart[call_block]:
                                    caller_range = chart[call_block][raiser_group]
                            
                            if caller_range:
                                scenario = {
                                    'description': f"{raiser_pos} raise, {chart_3better} 3bet, {raiser_pos} call",
                                    'scenario_type': '3bet_pot',
                                    'positions': {
                                        'original_raiser': raiser_pos,
                                        '3better': chart_3better
                                    },
                                    'ranges': {
                                        raiser_pos: caller_range,
                                        chart_3better: threebetter_range
                                    }
                                }
                                complete_scenarios.append(scenario)
    
    # 3. 4bet Pots - 24 scenarios
    # For each 3bet scenario, create a 4bet scenario
    for raiser_pos in rfi_positions:
        if raiser_pos not in chart[rfi_block]:
            continue
            
        raiser_group = POSITION_GROUPS.get(raiser_pos, raiser_pos)
        threbet_block = f"Cash, 100bb, 8-max, raise, {raiser_group}, 3bet"
        
        if threbet_block in chart:
            for chart_3better, _ in chart[threbet_block].items():
                # Handle MP 3betters for 4bet scenarios
                if chart_3better == 'MP' and raiser_group == 'EP':
                    for actual_3better in ['LJ', 'HJ']:
                        # Determine IP/OOP
                        ip_pos, oop_pos = determine_ip_oop(raiser_pos, actual_3better)
                        
                        # Get 4bet range for the original raiser
                        if raiser_pos == ip_pos:
                            fourbet_block = "Cash, 100bb, 8-max, 3bet, IP, 4bet"
                        else:
                            fourbet_block = "Cash, 100bb, 8-max, 3bet, OOP, 4bet"
                        
                        fourbet_range = None
                        if fourbet_block in chart:
                            if raiser_pos in chart[fourbet_block]:
                                fourbet_range = chart[fourbet_block][raiser_pos]
                            elif raiser_group in chart[fourbet_block]:
                                fourbet_range = chart[fourbet_block][raiser_group]
                        
                        # Get calling range for the 3better
                        if actual_3better == ip_pos:
                            call_block = "Cash, 100bb, 8-max, 4bet, IP, call"
                        else:
                            call_block = "Cash, 100bb, 8-max, 4bet, OOP, call"
                        
                        caller_range = None
                        if call_block in chart:
                            # 4bet calling ranges are usually universal
                            if "ALL" in chart[call_block]:
                                caller_range = chart[call_block]["ALL"]
                            elif actual_3better in chart[call_block]:
                                caller_range = chart[call_block][actual_3better]
                        
                        if fourbet_range and caller_range:
                            scenario = {
                                'description': f"{raiser_pos} raise, {actual_3better} 3bet, {raiser_pos} 4bet, {actual_3better} call",
                                'scenario_type': '4bet_pot',
                                'positions': {
                                    '4better': raiser_pos,
                                    'caller': actual_3better
                                },
                                'ranges': {
                                    raiser_pos: fourbet_range,
                                    actual_3better: caller_range
                                }
                            }
                            complete_scenarios.append(scenario)
                else:
                    # For all other positions
                    if chart_3better in all_positions:
                        raiser_idx = all_positions.index(raiser_pos)
                        threebetter_idx = all_positions.index(chart_3better)
                        if threebetter_idx > raiser_idx:
                            # Determine IP/OOP
                            ip_pos, oop_pos = determine_ip_oop(raiser_pos, chart_3better)
                            
                            # Get 4bet range
                            if raiser_pos == ip_pos:
                                fourbet_block = "Cash, 100bb, 8-max, 3bet, IP, 4bet"
                            else:
                                fourbet_block = "Cash, 100bb, 8-max, 3bet, OOP, 4bet"
                            
                            fourbet_range = None
                            if fourbet_block in chart:
                                if raiser_pos in chart[fourbet_block]:
                                    fourbet_range = chart[fourbet_block][raiser_pos]
                                elif raiser_group in chart[fourbet_block]:
                                    fourbet_range = chart[fourbet_block][raiser_group]
                            
                            # Get calling range
                            if chart_3better == ip_pos:
                                call_block = "Cash, 100bb, 8-max, 4bet, IP, call"
                            else:
                                call_block = "Cash, 100bb, 8-max, 4bet, OOP, call"
                            
                            caller_range = None
                            if call_block in chart:
                                if "ALL" in chart[call_block]:
                                    caller_range = chart[call_block]["ALL"]
                                elif chart_3better in chart[call_block]:
                                    caller_range = chart[call_block][chart_3better]
                            
                            if fourbet_range and caller_range:
                                scenario = {
                                    'description': f"{raiser_pos} raise, {chart_3better} 3bet, {raiser_pos} 4bet, {chart_3better} call",
                                    'scenario_type': '4bet_pot',
                                    'positions': {
                                        '4better': raiser_pos,
                                        'caller': chart_3better
                                    },
                                    'ranges': {
                                        raiser_pos: fourbet_range,
                                        chart_3better: caller_range
                                    }
                                }
                                complete_scenarios.append(scenario)
    
    return complete_scenarios


def calculate_scenario_equity(ranges: Dict[str, str], num_flops: int = 1000, samples_per_flop: int = 500) -> Dict[str, Any]:
    """Calculate average equity for a scenario across random flops.
    
    Parameters
    ----------
    ranges : Dict[str, str]
        Dictionary mapping positions to range strings
    num_flops : int
        Number of random flops to sample
    samples_per_flop : int
        Number of Monte Carlo samples per flop
        
    Returns
    -------
    Dict[str, Any]
        Equity results including average, min, max, and standard deviation
    """
    if len(ranges) != 2:
        return {'error': 'Need exactly 2 players'}
    
    positions = list(ranges.keys())
    pos1, pos2 = positions[0], positions[1]
    
    # Fix range notations
    range1 = fix_range_notation(ranges[pos1])
    range2 = fix_range_notation(ranges[pos2])
    
    equities = []
    
    # Sample random flops
    for _ in range(num_flops):
        # Create new deck for each iteration
        deck = eval7.Deck()
        deck.shuffle()
        flop = deck.deal(3)
        
        try:
            # Calculate equity
            equity = calculate_range_equity(range1, range2, flop, samples=samples_per_flop)
            equities.append(equity)
        except Exception as e:
            # Skip problematic flops
            continue
    
    if not equities:
        return {'error': 'No valid equity calculations'}
    
    # Calculate statistics
    avg_equity = sum(equities) / len(equities)
    min_equity = min(equities)
    max_equity = max(equities)
    
    # Calculate standard deviation
    variance = sum((eq - avg_equity) ** 2 for eq in equities) / len(equities)
    std_dev = variance ** 0.5
    
    return {
        'average_equity': {
            pos1: avg_equity,
            pos2: 1 - avg_equity
        },
        'statistics': {
            'samples': len(equities),
            'min': {pos1: min_equity, pos2: 1 - max_equity},
            'max': {pos1: max_equity, pos2: 1 - min_equity},
            'std_dev': std_dev
        }
    }


def build_equity_database(num_flops: int = 1000, samples_per_flop: int = 500, output_file: str = 'preflop_equity_database.json'):
    """Build comprehensive equity database for all scenarios.
    
    Parameters
    ----------
    num_flops : int
        Number of random flops to sample per scenario
    samples_per_flop : int
        Number of Monte Carlo samples per flop
    output_file : str
        Output filename for the database
    """
    print("Building Preflop Equity Database")
    print("=" * 70)
    
    # Get all scenarios
    scenarios = build_complete_scenarios()
    print(f"Found {len(scenarios)} scenarios to analyze")
    print(f"Settings: {num_flops} flops per scenario, {samples_per_flop} samples per flop")
    
    # Calculate equity for each scenario
    database = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'num_flops_per_scenario': num_flops,
            'samples_per_flop': samples_per_flop,
            'total_scenarios': len(scenarios)
        },
        'scenarios': {}
    }
    
    start_time = time.time()
    
    for i, scenario in enumerate(scenarios):
        desc = scenario['description']
        print(f"\n[{i+1}/{len(scenarios)}] Processing: {desc}")
        
        # Calculate equity
        equity_result = calculate_scenario_equity(scenario['ranges'], num_flops, samples_per_flop)
        
        if 'error' not in equity_result:
            # Store result
            database['scenarios'][desc] = {
                'type': scenario['scenario_type'],
                'positions': scenario['positions'],
                'ranges': scenario['ranges'],
                'equity': equity_result['average_equity'],
                'statistics': equity_result['statistics']
            }
            
            # Print summary
            positions = list(scenario['ranges'].keys())
            pos1, pos2 = positions[0], positions[1]
            eq1 = equity_result['average_equity'][pos1]
            eq2 = equity_result['average_equity'][pos2]
            
            print(f"  {pos1}: {eq1:.1%} | {pos2}: {eq2:.1%}")
            print(f"  Range advantage: {pos1 if eq1 > eq2 else pos2} by {abs(eq1 - eq2):.1%}")
        else:
            print(f"  Error: {equity_result['error']}")
    
    # Calculate overall statistics
    elapsed_time = time.time() - start_time
    database['metadata']['build_time_seconds'] = elapsed_time
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Database built successfully!")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output saved to: {output_file}")
    
    # Print summary statistics
    print_database_summary(database)


def print_database_summary(database: Dict[str, Any]):
    """Print summary statistics for the equity database."""
    print(f"\n{'=' * 70}")
    print("DATABASE SUMMARY")
    print(f"{'=' * 70}")
    
    scenarios = database['scenarios']
    
    # Group by scenario type
    by_type = defaultdict(list)
    for desc, data in scenarios.items():
        by_type[data['type']].append((desc, data))
    
    for scenario_type, scenario_list in by_type.items():
        print(f"\n{scenario_type.upper()} ({len(scenario_list)} scenarios)")
        print("-" * 50)
        
        # Find biggest advantages
        advantages = []
        for desc, data in scenario_list:
            positions = list(data['equity'].keys())
            eq1 = data['equity'][positions[0]]
            eq2 = data['equity'][positions[1]]
            advantage = abs(eq1 - eq2)
            leader = positions[0] if eq1 > eq2 else positions[1]
            advantages.append((advantage, leader, desc, eq1, eq2))
        
        # Sort by advantage size
        advantages.sort(reverse=True)
        
        # Show top 5
        print("Top 5 biggest advantages:")
        for i, (adv, leader, desc, eq1, eq2) in enumerate(advantages[:5]):
            print(f"{i+1}. {desc}: {leader} has {adv:.1%} advantage ({eq1:.1%} vs {eq2:.1%})")


def query_database(database_file: str = 'preflop_equity_database.json'):
    """Interactive query interface for the equity database."""
    # Load database
    with open(database_file, 'r') as f:
        database = json.load(f)
    
    print("\nPreflop Equity Database Query Interface")
    print("=" * 70)
    print(f"Database contains {len(database['scenarios'])} scenarios")
    print("\nExample queries:")
    print("  - UTG raise, BB call")
    print("  - BTN raise, SB 3bet")
    print("  - show all (list all scenarios)")
    print("  - quit")
    
    while True:
        query = input("\nEnter scenario or query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query.lower() == 'show all':
            print("\nAll scenarios:")
            for i, desc in enumerate(sorted(database['scenarios'].keys())):
                print(f"{i+1}. {desc}")
            continue
        
        # Search for matching scenarios
        matches = []
        for desc, data in database['scenarios'].items():
            if query.lower() in desc.lower():
                matches.append((desc, data))
        
        if not matches:
            print(f"No scenarios found matching '{query}'")
        elif len(matches) == 1:
            # Show detailed info
            desc, data = matches[0]
            print(f"\nScenario: {desc}")
            print("-" * 50)
            
            positions = list(data['equity'].keys())
            pos1, pos2 = positions[0], positions[1]
            eq1 = data['equity'][pos1]
            eq2 = data['equity'][pos2]
            
            print(f"\nEquity:")
            print(f"  {pos1}: {eq1:.1%}")
            print(f"  {pos2}: {eq2:.1%}")
            print(f"\nRange advantage: {pos1 if eq1 > eq2 else pos2} by {abs(eq1 - eq2):.1%}")
            
            print(f"\nRanges:")
            print(f"  {pos1}: {data['ranges'][pos1][:60]}..." if len(data['ranges'][pos1]) > 60 else f"  {pos1}: {data['ranges'][pos1]}")
            print(f"  {pos2}: {data['ranges'][pos2][:60]}..." if len(data['ranges'][pos2]) > 60 else f"  {pos2}: {data['ranges'][pos2]}")
            
            stats = data['statistics']
            print(f"\nStatistics (based on {stats['samples']} flops):")
            print(f"  Standard deviation: {stats['std_dev']:.1%}")
            print(f"  {pos1} equity range: {stats['min'][pos1]:.1%} - {stats['max'][pos1]:.1%}")
            
        else:
            # Multiple matches
            print(f"\nFound {len(matches)} matching scenarios:")
            for i, (desc, data) in enumerate(matches):
                positions = list(data['equity'].keys())
                eq1 = data['equity'][positions[0]]
                eq2 = data['equity'][positions[1]]
                print(f"{i+1}. {desc}: {positions[0]} {eq1:.1%} vs {positions[1]} {eq2:.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preflop Equity Database Builder')
    parser.add_argument('--build', action='store_true', help='Build the equity database')
    parser.add_argument('--query', action='store_true', help='Query existing database')
    parser.add_argument('--num-flops', type=int, default=1000, help='Number of flops to sample per scenario')
    parser.add_argument('--samples-per-flop', type=int, default=500, help='Number of Monte Carlo samples per flop')
    parser.add_argument('--output', default='preflop_equity_database.json', help='Output filename')
    
    args = parser.parse_args()
    
    if args.build:
        build_equity_database(num_flops=args.num_flops, samples_per_flop=args.samples_per_flop, output_file=args.output)
    elif args.query:
        if os.path.exists(args.output):
            query_database(args.output)
        else:
            print(f"Database file '{args.output}' not found. Build it first with --build")
    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print("  python preflop_equity_database.py --build")
        print("  python preflop_equity_database.py --build --num-flops 5000")
        print("  python preflop_equity_database.py --query")
