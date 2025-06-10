#!/usr/bin/env python3
"""
Texas Solver Runner

Takes input from solver_inputer.py, combines with default configuration,
creates input file and runs the Texas Solver.
"""

import os
import sys
import subprocess
import random
import argparse
from typing import Dict, Any, List, Tuple

# Add current directory to path for importing solver_inputer
sys.path.append(os.path.dirname(__file__))

from solver_inputer import generate_complete_scenario, format_range_simple, determine_ip_oop, format_flop
from preflop_scenario_generator import (
    PREFLOP_POSITIONS, 
    POSITION_GROUPS,
    get_positions_after,
    choose_action
)


def generate_weighted_preflop_scenario() -> str:
    """Generate a random preflop scenario with weighted RFI excluding slow-solving spots.
    
    This function ensures equal probability for each valid scenario by:
    1. Enumerating all valid scenarios
    2. Randomly selecting one
    
    Valid scenarios exclude:
    - SB raise, BB call (takes too long to solve)
    - BTN raise, SB call (takes too long to solve)
    - BTN raise, BB call (takes too long to solve)
    - Impossible matchups like UTG vs UTG+1 or LJ vs HJ (same position group)
    
    Returns
    -------
    str
        A preflop action string like "UTG raise, BB call" or "UTG raise, BTN 3bet, UTG call"
    """
    # Define all valid single raised pot scenarios
    valid_srp_scenarios = []
    
    # For each raiser position
    for raiser in PREFLOP_POSITIONS:
        # Skip SB as raiser when only BB can respond (we want to exclude SB vs BB)
        if raiser == "SB":
            continue
            
        # Get positions that can respond
        positions_after = get_positions_after(raiser)
        
        # Get raiser's position group
        raiser_group = POSITION_GROUPS.get(raiser, raiser)
        
        for responder in positions_after:
            # Skip if responder is in same position group as raiser
            responder_group = POSITION_GROUPS.get(responder, responder)
            
            # Check for invalid combinations
            if raiser_group == responder_group and raiser_group in ["EP", "MP"]:
                # Can't have EP vs EP or MP vs MP
                continue
            
            # Skip BTN vs SB and BTN vs BB scenarios (slow to solve)
            if raiser == "BTN" and responder in ["SB", "BB"]:
                continue
            
            # Add valid SRP scenario
            valid_srp_scenarios.append((raiser, responder, "call"))
    
    # Define all valid 3bet scenarios (same logic but for 3bets)
    valid_3bet_scenarios = []
    
    for raiser in PREFLOP_POSITIONS:
        if raiser == "SB":
            continue
            
        positions_after = get_positions_after(raiser)
        raiser_group = POSITION_GROUPS.get(raiser, raiser)
        
        for threebetter in positions_after:
            threebetter_group = POSITION_GROUPS.get(threebetter, threebetter)
            
            if raiser_group == threebetter_group and raiser_group in ["EP", "MP"]:
                continue
            
            # Add 3bet scenario with caller response
            valid_3bet_scenarios.append((raiser, threebetter, "3bet", "call"))
    
    # Define all valid 4bet scenarios
    valid_4bet_scenarios = []
    
    for raiser in PREFLOP_POSITIONS:
        if raiser == "SB":
            continue
            
        positions_after = get_positions_after(raiser)
        raiser_group = POSITION_GROUPS.get(raiser, raiser)
        
        for threebetter in positions_after:
            threebetter_group = POSITION_GROUPS.get(threebetter, threebetter)
            
            if raiser_group == threebetter_group and raiser_group in ["EP", "MP"]:
                continue
            
            # Add 4bet scenario
            valid_4bet_scenarios.append((raiser, threebetter, "3bet", "4bet", "call"))
    
    # Weight selection based on typical frequencies
    # ~75% SRP, ~21% 3bet pots, ~4% 4bet pots
    total_scenarios = len(valid_srp_scenarios) + len(valid_3bet_scenarios) + len(valid_4bet_scenarios)
    
    # Calculate weights to achieve desired distribution
    srp_weight = 0.75
    threebet_weight = 0.21
    fourbet_weight = 0.04
    
    # Randomly choose scenario type based on weights
    rand_val = random.random()
    
    if rand_val < srp_weight:
        # Generate SRP
        if valid_srp_scenarios:
            raiser, caller, _ = random.choice(valid_srp_scenarios)
            return f"{raiser} raise, {caller} call"
    elif rand_val < srp_weight + threebet_weight:
        # Generate 3bet pot
        if valid_3bet_scenarios:
            raiser, threebetter, _, _ = random.choice(valid_3bet_scenarios)
            return f"{raiser} raise, {threebetter} 3bet, {raiser} call"
    else:
        # Generate 4bet pot
        if valid_4bet_scenarios:
            raiser, threebetter, _, _, _ = random.choice(valid_4bet_scenarios)
            return f"{raiser} raise, {threebetter} 3bet, {raiser} 4bet, {threebetter} call"
    
    # Fallback (shouldn't reach here)
    return "UTG raise, BB call"


def generate_complete_scenario_weighted() -> Dict[str, Any]:
    """Generate a complete scenario using weighted preflop generation."""
    from solver_inputer import (
        get_ranges_with_frequencies,
        expand_range_notation,
        calculate_pot_and_stacks,
        generate_flop,
        generate_filename
    )
    
    # Generate weighted preflop scenario
    scenario = generate_weighted_preflop_scenario()
    
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


# Import texture analysis functions
from solver_inputer import (
    analyze_suits,
    analyze_pairing,
    analyze_hirank,
    analyze_connectivity
)


def load_default_config() -> str:
    """Load the default configuration from default_config.txt.
    
    Returns
    -------
    str
        The default configuration content
    """
    config_path = os.path.join(os.path.dirname(__file__), "default_config.txt")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Default config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return f.read().strip()


def format_ranges_for_solver(range_data: Dict[str, Any], scenario: str) -> tuple:
    """Format ranges for solver input.
    
    Parameters
    ----------
    range_data : Dict[str, Any]
        Range data from solver_inputer
    scenario : str
        The scenario string
        
    Returns
    -------
    tuple
        (ip_range, oop_range) formatted for solver
    """
    from solver_inputer import parse_scenario
    
    actions = parse_scenario(scenario)
    if len(actions) < 2:
        return "error", "error"
    
    # Get the two main players from the scenario
    acting_pos = range_data.get('acting_position', '')
    prev_acting_pos = range_data.get('prev_acting_position', '')
    
    # Extract positions from scenario if range data is missing
    if not acting_pos or not prev_acting_pos:
        # Get last two players from actions
        if len(actions) >= 2:
            acting_pos = actions[-1][0]
            prev_acting_pos = actions[-2][0]
    
    if not acting_pos or not prev_acting_pos:
        return "error", "error"
    
    # Determine IP/OOP
    ip_pos, oop_pos = determine_ip_oop(acting_pos, prev_acting_pos)
    
    # Get ranges - check for actual range data in basic_ranges first
    basic_ranges = range_data.get('basic_ranges', {})
    
    # For acting position
    acting_frequencies = range_data.get('frequencies', {})
    acting_primary = range_data.get('primary_action', 'unknown')
    
    if acting_frequencies:
        acting_range = format_range_simple(acting_frequencies, acting_primary)
    elif acting_pos in basic_ranges:
        # Use the actual range from basic_ranges
        _, range_str = basic_ranges[acting_pos]
        acting_range = range_str if range_str else "AA"
    else:
        acting_range = "AA"
    
    # For previous acting position
    prev_frequencies = range_data.get('prev_frequencies', {})
    prev_primary = range_data.get('prev_primary_action', 'unknown')
    
    if prev_frequencies:
        prev_range = format_range_simple(prev_frequencies, prev_primary)
    elif prev_acting_pos in basic_ranges:
        # Use the actual range from basic_ranges
        _, range_str = basic_ranges[prev_acting_pos]
        prev_range = range_str if range_str else "AA"
    else:
        prev_range = "AA"
    
    # Remove spaces from range strings for Texas Solver compatibility
    acting_range = acting_range.replace(" ", "")
    prev_range = prev_range.replace(" ", "")
    
    # Assign to IP/OOP
    if acting_pos == ip_pos:
        ip_range = acting_range
        oop_range = prev_range
    else:
        ip_range = prev_range
        oop_range = acting_range
    
    # If ranges are still empty or placeholder, use fallback
    if not ip_range or ip_range == "no range" or ip_range == "full RFI range":
        ip_range = "AA"
    if not oop_range or oop_range == "no range" or oop_range == "full RFI range":
        oop_range = "AA"
    
    return ip_range, oop_range


def create_solver_input_file(scenario_data: Dict[str, Any], filename_base: str, 
                             inputs_dir: str = None, outputs_dir: str = None) -> str:
    """Create the solver input file.
    
    Parameters
    ----------
    scenario_data : Dict[str, Any]
        Complete scenario data from solver_inputer
    filename_base : str
        Base filename (without extension)
    inputs_dir : str, optional
        Directory for input files (default: solver_inputs)
    outputs_dir : str, optional
        Directory for output files (default: solver_outputs)
        
    Returns
    -------
    str
        Path to the created input file
    """
    # Import here to avoid circular imports
    from solver_inputer import RANK_VALUES
    
    # Use default directories if not specified
    if inputs_dir is None:
        inputs_dir = os.path.join(os.path.dirname(__file__), "solver_inputs")
    else:
        # Convert to absolute path if relative to script directory
        if not os.path.isabs(inputs_dir):
            inputs_dir = os.path.join(os.path.dirname(__file__), inputs_dir)
    
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
    else:
        # Convert to absolute path if relative to script directory
        if not os.path.isabs(outputs_dir):
            outputs_dir = os.path.join(os.path.dirname(__file__), outputs_dir)
    
    # Create directories if they don't exist
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Create the input file path
    input_filename = f"{filename_base}.txt"
    input_path = os.path.join(inputs_dir, input_filename)
    
    # Create the full output path for the JSON file
    output_filename = f"{filename_base}.json"
    output_path = os.path.join(outputs_dir, output_filename)
    
    # Extract data
    pot_info = scenario_data['pot_info']
    pot_size = pot_info.get('total_pot', 0)
    effective_stack = pot_info.get('effective_stack', 0)
    flop = scenario_data['flop']  # Use original flop list instead of flop_string
    range_data = scenario_data['range_data']
    scenario = scenario_data['scenario']
    
    # Format board with commas for Texas Solver
    # Sort by rank (high to low) and join with commas
    sorted_flop = sorted(flop, key=lambda card: RANK_VALUES[card[0]], reverse=True)
    board_string = ','.join(sorted_flop)
    
    # Format ranges
    ip_range, oop_range = format_ranges_for_solver(range_data, scenario)
    
    # Load default config
    default_config = load_default_config()
    
    # Create the input file content
    input_content = []
    input_content.append(f"set_pot {pot_size}")
    input_content.append(f"set_effective_stack {effective_stack}")
    input_content.append(f"set_board {board_string}")
    input_content.append(f"set_range_ip {ip_range}")
    input_content.append(f"set_range_oop {oop_range}")
    input_content.append(default_config)
    input_content.append(f"dump_result {output_path}")
    
    # Write the file
    with open(input_path, 'w') as f:
        f.write('\n'.join(input_content))
    
    print(f"Created solver input file: {input_path}")
    return input_path


def run_texas_solver(input_path: str, outputs_dir: str = None) -> bool:
    """Run the Texas Solver with the given input file.
    
    Parameters
    ----------
    input_path : str
        Path to the input file
    outputs_dir : str, optional
        Directory for output files (default: solver_outputs)
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Create solver_outputs directory if it doesn't exist
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
    else:
        # Convert to absolute path if relative to script directory
        if not os.path.isabs(outputs_dir):
            outputs_dir = os.path.join(os.path.dirname(__file__), outputs_dir)
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Get the project root directory (go up two levels from scripts/solver)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Look for texas solver executable in the texas_solver directory
    solver_executable = None
    texas_solver_dir = os.path.join(project_root, "texas_solver")
    possible_names = ["console_solver", "texas_solver", "console_solver.exe", "texas_solver.exe"]
    
    for name in possible_names:
        exe_path = os.path.join(texas_solver_dir, name)
        if os.path.exists(exe_path):
            solver_executable = exe_path
            break
    
    if not solver_executable:
        print("Error: Texas Solver executable not found!")
        print(f"Looking in directory: {texas_solver_dir}")
        print("Please ensure console_solver or texas_solver is in the texas_solver directory")
        print("Available files in texas_solver directory:")
        try:
            if os.path.exists(texas_solver_dir):
                files = os.listdir(texas_solver_dir)
                for f in files:
                    if not f.startswith('.') and not f.startswith('__'):
                        print(f"  {f}")
            else:
                print(f"  Directory {texas_solver_dir} does not exist")
        except Exception as e:
            print(f"  Could not list files: {e}")
        return False
    
    # Make sure the executable is executable (on Unix systems)
    if os.name != 'nt':  # Not Windows
        try:
            os.chmod(solver_executable, 0o755)
        except Exception as e:
            print(f"Warning: Could not make executable executable: {e}")
    
    # Run the solver
    cmd = [solver_executable, "-i", input_path]
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=texas_solver_dir  # Run from texas_solver directory
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("=" * 60)
            print("Solver completed successfully!")
            return True
        else:
            print("=" * 60)
            print(f"Solver failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running solver: {e}")
        return False


def main():
    """Main function to generate scenario and run solver."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Texas Solver Runner - Generate and solve poker scenarios')
    parser.add_argument('--input-dir', '-i', type=str, default='solver_inputs',
                      help='Directory for input .txt files (default: solver_inputs)')
    parser.add_argument('--output-dir', '-o', type=str, default='solver_outputs', 
                      help='Directory for output .json files (default: solver_outputs)')
    
    args = parser.parse_args()
    
    print("Texas Solver Runner - Weighted RFI Mode (Excluding Slow Spots)")
    print("=" * 50)
    print("Generating random scenarios with equal probability...")
    print("Excluding SB vs BB, BTN vs SB, BTN vs BB spots for faster solving")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from solver_inputer import RANK_VALUES
    
    scenario_count = 0
    
    try:
        while True:
            scenario_count += 1
            print(f"\n🎯 SCENARIO #{scenario_count}")
            print("-" * 30)
            
            # Keep generating scenarios until we get a unique filename
            max_attempts = 100  # Prevent infinite loop
            attempts = 0
            
            while attempts < max_attempts:
                attempts += 1
                
                # Generate a complete scenario using weighted generation
                if attempts == 1:
                    print(f"Generating weighted random scenario...")
                else:
                    print(f"Generating weighted random scenario (attempt {attempts})...")
                    
                scenario_data = generate_complete_scenario_weighted()
                
                # Extract filename base (without .json extension)
                filename = scenario_data['filename']
                filename_base = filename.replace('.json', '')
                
                # Extract scenario part (everything after flop cards)
                # Format: FlopCards_Raiser_Caller_PotType_Suits_Pairing_Hirank_Connectivity
                # We want to check collision only for: Raiser_Caller_PotType_Suits_Pairing_Hirank_Connectivity
                parts = filename_base.split('_')
                if len(parts) >= 8:  # New format with flop cards
                    scenario_part = '_'.join(parts[1:])  # Everything after flop cards
                else:  # Old format without flop cards
                    scenario_part = filename_base
                
                # Check if any existing file has the same scenario part
                # Use the command line specified input directory
                inputs_dir = args.input_dir
                if not os.path.isabs(inputs_dir):
                    inputs_dir = os.path.join(os.path.dirname(__file__), inputs_dir)
                
                # Look for existing files with the same scenario part
                scenario_collision = False
                if os.path.exists(inputs_dir):
                    for existing_file in os.listdir(inputs_dir):
                        if existing_file.endswith('.txt'):
                            existing_base = existing_file.replace('.txt', '')
                            existing_parts = existing_base.split('_')
                
                            if len(existing_parts) >= 8:  # New format
                                existing_scenario_part = '_'.join(existing_parts[1:])
                            else:  # Old format
                                existing_scenario_part = existing_base
                            
                            if existing_scenario_part == scenario_part:
                                scenario_collision = True
                                break
                
                if not scenario_collision:
                    # Unique scenario found!
                    break
                else:
                    print(f"  Scenario {scenario_part} already exists, regenerating...")
            
            if attempts >= max_attempts:
                print(f"⚠️  Could not generate unique filename after {max_attempts} attempts, skipping...")
                continue
            
            # Display the scenario
            scenario = scenario_data['scenario']
            pot_info = scenario_data['pot_info']
            flop = scenario_data['flop']
            
            print(f"Generated scenario: {scenario}")
            print(f"Pot: {pot_info.get('total_pot', 0)}bb")
            print(f"Effective stack: {pot_info.get('effective_stack', 0)}bb")
            
            # Format flop for display (without commas)
            sorted_flop_display = sorted(flop, key=lambda card: RANK_VALUES[card[0]], reverse=True)
            flop_display = ''.join(sorted_flop_display)
            print(f"Flop: {flop_display}")
            print(f"Output file: {filename}")
            
            # Create solver input file
            input_path = create_solver_input_file(scenario_data, filename_base, 
                                                 inputs_dir=args.input_dir, 
                                                 outputs_dir=args.output_dir)
            
            # Automatically run Texas Solver
            print(f"\n🚀 Starting Texas Solver...")
            success = run_texas_solver(input_path, outputs_dir=args.output_dir)
            
            if success:
                outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
                output_file = os.path.join(outputs_dir, filename)
                print(f"✅ Results saved to: {filename}")
                print(f"📊 Total scenarios completed: {scenario_count}")
            else:
                print(f"❌ Solver failed for scenario #{scenario_count}")
                print(f"📄 Input file saved at: {input_path}")
            
            print(f"\n{'='*50}")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopping solver runner...")
        print(f"📈 Total scenarios processed: {scenario_count}")
        print(f"📁 Input files saved in: solver_inputs/")
        print(f"📁 Output files saved in: solver_outputs/")
        print(f"👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📈 Scenarios completed before error: {scenario_count}")
        raise


if __name__ == "__main__":
    main() 