#!/usr/bin/env python3
"""
Texas Solver Result Analyzer

Interprets and displays Texas Solver JSON output files in a user-friendly format.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional


def load_json_result(filename: str) -> Optional[Dict[str, Any]]:
    """Load a JSON result file from solver_outputs directory.
    
    Parameters
    ----------
    filename : str
        Name of the JSON file (with or without .json extension)
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed JSON data or None if file not found
    """
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Check in solver_outputs directory
    outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
    file_path = os.path.join(outputs_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded: {filename}")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def extract_scenario_info(filename: str) -> Dict[str, str]:
    """Extract scenario information from filename.
    
    Parameters
    ----------
    filename : str
        The filename to parse
        
    Returns
    -------
    Dict[str, str]
        Dictionary with scenario information
    """
    # Remove .json extension for parsing
    base_name = filename.replace('.json', '')
    
    try:
        # Format: raiser_caller_potType_suits_pairing_hirank_connectivity
        parts = base_name.split('_')
        if len(parts) >= 7:
            return {
                'raiser': parts[0],
                'caller': parts[1], 
                'pot_type': parts[2],
                'suits': parts[3],
                'pairing': parts[4],
                'hirank': parts[5],
                'connectivity': parts[6]
            }
    except Exception:
        pass
    
    return {'filename': base_name}


def format_percentage(value: float) -> str:
    """Format a decimal value as percentage."""
    return f"{value * 100:.1f}%"


def format_bb(value: float) -> str:
    """Format a value in big blinds."""
    return f"{value:.2f}bb"


def display_game_info(data: Dict[str, Any], filename: str) -> None:
    """Display basic game information."""
    print("\n" + "="*60)
    print("🎯 TEXAS SOLVER RESULT ANALYSIS")
    print("="*60)
    
    scenario_info = extract_scenario_info(filename)
    
    print(f"📁 File: {filename}")
    
    if len(scenario_info) > 1:
        print(f"👥 Scenario: {scenario_info.get('raiser', 'Unknown')} vs {scenario_info.get('caller', 'Unknown')}")
        print(f"💰 Pot Type: {scenario_info.get('pot_type', 'Unknown')}")
        
        # Format board texture info
        suits_map = {'mono': 'Monotone', 'tt': 'Two-tone', 'rb': 'Rainbow'}
        pairing_map = {'np': 'Unpaired', 'lowpair': 'Low pair', 'midpair': 'Mid pair', 
                      'broadpair': 'Broadway pair', 'acepair': 'Ace pair', 'trips': 'Trips'}
        hirank_map = {'ah': 'Ace high', 'bh': 'Broadway high', 'mh': 'Mid high', 'low': 'Low'}
        connectivity_map = {'high': 'Highly connected', 'semi': 'Semi-connected', 'dry': 'Dry'}
        
        suits = suits_map.get(scenario_info.get('suits', ''), scenario_info.get('suits', 'Unknown'))
        pairing = pairing_map.get(scenario_info.get('pairing', ''), scenario_info.get('pairing', 'Unknown'))
        hirank = hirank_map.get(scenario_info.get('hirank', ''), scenario_info.get('hirank', 'Unknown'))
        connectivity = connectivity_map.get(scenario_info.get('connectivity', ''), scenario_info.get('connectivity', 'Unknown'))
        
        print(f"🃏 Board: {suits}, {pairing}, {hirank}, {connectivity}")
    
    # Texas Solver specific information
    if 'node_type' in data:
        print(f"🌳 Root node type: {data['node_type']}")
    
    if 'player' in data:
        print(f"🎭 Root player: {data['player']}")
    
    if 'actions' in data:
        actions = data['actions']
        if isinstance(actions, list):
            print(f"🎬 Root actions: {', '.join(actions)}")


def display_exploitability(data: Dict[str, Any]) -> None:
    """Display exploitability information."""
    if 'exploitability' not in data:
        return
    
    print("\n" + "="*60)
    print("📊 EXPLOITABILITY ANALYSIS")
    print("="*60)
    
    exploit = data['exploitability']
    
    if 'player_0' in exploit:
        print(f"🔵 Player 0 (IP): {format_bb(exploit['player_0'])}")
    
    if 'player_1' in exploit:
        print(f"🔴 Player 1 (OOP): {format_bb(exploit['player_1'])}")
    
    if 'total' in exploit:
        total_bb = exploit['total']
        total_pct = (total_bb / data.get('pot', 1)) * 100 if 'pot' in data else 0
        print(f"⚖️  Total: {format_bb(total_bb)} ({total_pct:.1f}% of pot)")


def display_strategy_summary(data: Dict[str, Any]) -> None:
    """Display strategy summary."""
    if 'strategy' not in data:
        return
    
    print("\n" + "="*60)
    print("🎲 STRATEGY SUMMARY")
    print("="*60)
    
    strategy = data['strategy']
    
    for player_name, player_data in strategy.items():
        if isinstance(player_data, dict):
            print(f"\n🎯 {player_name.upper()}:")
            
            # Display ranges if available
            if 'range' in player_data:
                range_info = player_data['range']
                if isinstance(range_info, str):
                    print(f"   📋 Range: {range_info[:100]}...")
                elif isinstance(range_info, dict) and 'hands' in range_info:
                    hands = range_info['hands']
                    print(f"   📋 Range: {len(hands)} combinations")
            
            # Display action frequencies if available
            if 'actions' in player_data:
                actions = player_data['actions']
                print(f"   🎬 Actions available: {', '.join(actions.keys())}")


def display_strategy_analysis(data: Dict[str, Any]) -> None:
    """Display strategy analysis for Texas Solver format."""
    print("\n" + "="*60)
    print("🎲 STRATEGY ANALYSIS")
    print("="*60)
    
    def analyze_node_strategy(node_data, path="root", depth=0):
        """Analyze strategy at a specific node."""
        if not isinstance(node_data, dict) or depth > 3:  # Limit depth for display
            return
        
        # Display node information
        if 'strategy' in node_data and 'actions' in node_data:
            strategy = node_data['strategy']
            actions = node_data['actions']
            player = node_data.get('player', 'Unknown')
            
            if isinstance(strategy, list) and isinstance(actions, list) and len(strategy) == len(actions):
                print(f"\n📍 Node: {path}")
                print(f"   🎭 Player: {player}")
                
                # Show action frequencies
                for action, freq in zip(actions, strategy):
                    if freq > 0:  # Only show actions with positive frequency
                        print(f"   🎬 {action}: {format_percentage(freq)}")
        
        # Recurse through children (limited depth)
        if depth < 2 and 'childrens' in node_data and isinstance(node_data['childrens'], dict):
            for child_name, child_data in list(node_data['childrens'].items())[:3]:  # Limit to first 3 children
                new_path = f"{path} → {child_name}" if path != "root" else child_name
                analyze_node_strategy(child_data, new_path, depth + 1)
    
    analyze_node_strategy(data)


def display_detailed_strategy(data: Dict[str, Any], show_details: bool = False) -> None:
    """Display detailed strategy information for Texas Solver format."""
    if not show_details:
        return
    
    print("\n" + "="*60)
    print("🔍 DETAILED STRATEGY TREE")
    print("="*60)
    
    def show_detailed_node(node_data, path="root", depth=0):
        """Show detailed information for each node."""
        if not isinstance(node_data, dict) or depth > 5:  # Show more depth for detailed view
            return
        
        # Display comprehensive node information
        print(f"\n{'  ' * depth}📍 {path}")
        
        if 'node_type' in node_data:
            print(f"{'  ' * depth}   Type: {node_data['node_type']}")
        
        if 'player' in node_data:
            print(f"{'  ' * depth}   Player: {node_data['player']}")
        
        if 'actions' in node_data and 'strategy' in node_data:
            actions = node_data['actions']
            strategy = node_data['strategy']
            
            if isinstance(actions, list) and isinstance(strategy, list) and len(actions) == len(strategy):
                print(f"{'  ' * depth}   Strategy:")
                for action, freq in zip(actions, strategy):
                    if freq > 0.01:  # Show actions with >1% frequency
                        print(f"{'  ' * depth}     {action}: {format_percentage(freq)}")
        
        # Recurse through children
        if 'childrens' in node_data and isinstance(node_data['childrens'], dict):
            children_count = len(node_data['childrens'])
            if children_count > 0:
                print(f"{'  ' * depth}   Children: {children_count}")
                
                # Show first few children in detail
                for i, (child_name, child_data) in enumerate(node_data['childrens'].items()):
                    if i < 5:  # Limit to first 5 children
                        show_detailed_node(child_data, child_name, depth + 1)
                    elif i == 5:
                        print(f"{'  ' * (depth + 1)}... and {children_count - 5} more children")
                        break
    
    show_detailed_node(data)


def analyze_game_tree_structure(data: Dict[str, Any]) -> None:
    """Analyze the Texas Solver game tree structure."""
    print("\n" + "="*60)
    print("🌳 GAME TREE ANALYSIS")
    print("="*60)
    
    # Count nodes recursively
    def count_nodes(node_data, depth=0):
        """Recursively count nodes in the tree."""
        if not isinstance(node_data, dict):
            return 0, 0, 0
        
        total_nodes = 1
        max_depth = depth
        terminal_nodes = 0
        
        # Check if this is a terminal node (no children)
        if 'childrens' not in node_data or not node_data['childrens']:
            terminal_nodes = 1
        
        # Recurse through children
        if 'childrens' in node_data and isinstance(node_data['childrens'], dict):
            for child_name, child_data in node_data['childrens'].items():
                child_total, child_terminals, child_depth = count_nodes(child_data, depth + 1)
                total_nodes += child_total
                terminal_nodes += child_terminals
                max_depth = max(max_depth, child_depth)
        
        return total_nodes, terminal_nodes, max_depth
    
    total_nodes, terminal_nodes, max_depth = count_nodes(data)
    decision_nodes = total_nodes - terminal_nodes
    
    print(f"📊 Total nodes: {total_nodes:,}")
    print(f"🎬 Decision nodes: {decision_nodes:,}")
    print(f"🏁 Terminal nodes: {terminal_nodes:,}")
    print(f"📏 Tree depth: {max_depth}")


def list_available_files() -> List[str]:
    """List all available JSON files in solver_outputs directory."""
    outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
    
    if not os.path.exists(outputs_dir):
        return []
    
    json_files = []
    for filename in os.listdir(outputs_dir):
        if filename.endswith('.json'):
            json_files.append(filename)
    
    return sorted(json_files)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze Texas Solver JSON output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 result_analyzer.py filename.json      # Analyze specific file
  python3 result_analyzer.py filename -d        # Show detailed strategy
  python3 result_analyzer.py --list             # List available files
        """
    )
    
    parser.add_argument(
        'filename', 
        nargs='?',
        help='JSON filename to analyze (with or without .json extension)'
    )
    
    parser.add_argument(
        '-d', '--detailed',
        action='store_true',
        help='Show detailed strategy information'
    )
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all available JSON files'
    )
    
    args = parser.parse_args()
    
    # List available files
    if args.list:
        print("📁 Available result files:")
        print("=" * 40)
        
        files = list_available_files()
        if files:
            for i, filename in enumerate(files, 1):
                file_path = os.path.join(os.path.dirname(__file__), "solver_outputs", filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"{i:2d}. {filename} ({file_size:.1f} MB)")
        else:
            print("No JSON files found in solver_outputs directory")
        return
    
    # Check if filename provided
    if not args.filename:
        print("❌ Error: Please provide a filename or use --list to see available files")
        print("Usage: python3 result_analyzer.py filename.json")
        return
    
    # Load and analyze the file
    data = load_json_result(args.filename)
    if data is None:
        return
    
    # Display analysis
    display_game_info(data, args.filename)
    analyze_game_tree_structure(data)
    display_strategy_analysis(data)
    
    if args.detailed:
        display_detailed_strategy(data, show_details=True)
    else:
        print(f"\n💡 Use -d flag for detailed strategy tree analysis")
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main() 