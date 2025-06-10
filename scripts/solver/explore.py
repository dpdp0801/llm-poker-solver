#!/usr/bin/env python3
"""
Poker Solver Strategy Explorer

Explore solver strategies with focus on overall frequencies and hand-by-hand recommendations.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


class StrategyExplorer:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = self.load_json_file(filename)
        self.current_path = []
        self.current_node = self.data
        self.history = []  # Track action history
        
        # Parse filename for positions and pot type
        self.parse_filename_info(filename)
        
        # Get initial pot size from solver input file
        self.initial_pot = self.get_initial_pot_size()
        
        # Get effective stack size from solver input file
        self.effective_stack = self.get_effective_stack_size()
        
        # Track current street bet for raise calculations
        self.current_street_bet = 0.0
        
    def parse_filename_info(self, filename: str):
        """Parse filename to extract positions and pot type."""
        # Remove path and extension
        base_name = os.path.basename(filename).replace('.json', '')
        
        try:
            # Format: flop_pos1_pos2_pottype_...
            parts = base_name.split('_')
            if len(parts) >= 4:
                self.flop = parts[0]
                self.pos1 = parts[1]  # Aggressor/Raiser
                self.pos2 = parts[2]  # Caller
                self.pot_type = parts[3]
                
                # Determine who is IP/OOP based on positions
                position_order = ['SB', 'BB', 'UTG', 'UTG+1', 'LJ', 'HJ', 'CO', 'BTN']
                
                # Handle generalized positions
                if self.pos1 == 'EP':
                    self.pos1 = 'UTG'  # Default EP to UTG for display
                elif self.pos1 == 'MP':
                    self.pos1 = 'HJ'   # Default MP to HJ for display
                    
                if self.pos2 == 'EP':
                    self.pos2 = 'UTG+1'
                elif self.pos2 == 'MP':
                    self.pos2 = 'LJ'
                elif self.pos2 == 'IP':
                    # For 3bet/4bet, IP could be CO or BTN
                    self.pos2 = 'BTN'
                elif self.pos2 == 'OOP':
                    # For 3bet/4bet, OOP is usually BB
                    self.pos2 = 'BB'
                
                # Determine IP/OOP based on postflop position order
                # SB always acts first postflop (is OOP)
                try:
                    pos1_idx = position_order.index(self.pos1)
                    pos2_idx = position_order.index(self.pos2)
                    
                    # Lower index = acts first postflop = OOP
                    if pos1_idx < pos2_idx:
                        self.oop_pos = self.pos1
                        self.ip_pos = self.pos2
                    else:
                        self.oop_pos = self.pos2
                        self.ip_pos = self.pos1
                except ValueError:
                    # Default if position not found
                    self.ip_pos = self.pos2
                    self.oop_pos = self.pos1
            else:
                self.pos1 = "Unknown"
                self.pos2 = "Unknown"
                self.pot_type = "Unknown"
                self.ip_pos = "Unknown"
                self.oop_pos = "Unknown"
        except Exception:
            self.pos1 = "Unknown"
            self.pos2 = "Unknown"
            self.pot_type = "Unknown"
            self.ip_pos = "Unknown"
            self.oop_pos = "Unknown"
    
    def get_initial_pot_size(self) -> float:
        """Get initial pot size from solver input file."""
        # Try to load corresponding input file
        input_filename = self.filename.replace('solver_outputs', 'solver_inputs').replace('.json', '.txt')
        
        try:
            if os.path.exists(input_filename):
                with open(input_filename, 'r') as f:
                    for line in f:
                        if line.startswith('set_pot'):
                            parts = line.split()
                            if len(parts) >= 2:
                                return float(parts[1])
        except Exception:
            pass
        
        # Default pot sizes based on pot type
        if self.pot_type == 'SRP':
            return 6.0  # 2.5bb raise + 2.5bb call + 0.5bb + 0.5bb blinds
        elif self.pot_type == '3bet':
            return 18.5  # Approximate 3bet pot
        elif self.pot_type == '4bet':
            return 45.0  # Approximate 4bet pot
        
        return 6.0  # Default
    
    def get_effective_stack_size(self) -> float:
        """Get effective stack size from solver input file."""
        # Try to load corresponding input file
        input_filename = self.filename.replace('solver_outputs', 'solver_inputs').replace('.json', '.txt')
        
        try:
            if os.path.exists(input_filename):
                with open(input_filename, 'r') as f:
                    for line in f:
                        if line.startswith('set_effective_stack'):
                            parts = line.split()
                            if len(parts) >= 2:
                                return float(parts[1])
        except Exception:
            pass
        
        # Default effective stacks based on pot type
        if self.pot_type == 'SRP':
            return 97.5  # 100bb - 2.5bb from preflop
        elif self.pot_type == '3bet':
            return 92.5  # 100bb - 7.5bb from preflop
        elif self.pot_type == '4bet':
            return 78.0  # 100bb - 22bb from preflop
        
        return 97.5  # Default
    
    def get_current_pot(self) -> float:
        """Get current pot size from node or calculate it."""
        # Start with initial pot
        pot = self.initial_pot
        
        # Add bets from history
        current_bet = 0.0
        for pos, action in self.history:
            action_upper = action.upper()
            if action_upper.startswith('BET') or action_upper.startswith('RAISE'):
                # Extract bb amount
                parts = action.split()
                if len(parts) > 1:
                    try:
                        bet_amount = float(parts[1])
                        pot += bet_amount
                        current_bet = bet_amount
                    except:
                        pass
            elif action_upper == 'CALL':
                # Caller matches the current bet
                pot += current_bet
                current_bet = 0.0  # Reset after both players have acted
            elif action_upper == 'FOLD':
                # Pot stays the same
                pass
        
        return pot
        
    def load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from solver_outputs directory."""
        if os.path.exists(filename):
            file_path = filename
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
            file_path = os.path.join(outputs_dir, filename)
            
            if not file_path.endswith('.json'):
                file_path += '.json'
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded: {os.path.basename(file_path)}\n")
            return data
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)
    
    def get_player_position(self, player_num: int) -> str:
        """Get position name for player number."""
        # Player 1 is OOP (acts first postflop)
        # Player 0 is IP (acts second postflop)
        if player_num == 1:
            return self.oop_pos
        else:
            return self.ip_pos
    
    def get_preflop_history(self) -> str:
        """Construct preflop history based on pot type and positions."""
        if self.pot_type == 'SRP':
            # Single raised pot
            return f"{self.pos1} raises 2.5bb, {self.pos2} calls 2.5bb"
        elif self.pot_type == '3bet':
            # 3bet pot - pos1 is 3better, pos2 is initial raiser who called
            return f"{self.pos2} raises 2.5bb, {self.pos1} 3bets to 7.5bb, {self.pos2} calls 7.5bb"
        elif self.pot_type == '4bet':
            # 4bet pot - pos1 is 4better, pos2 is 3better who called
            return f"{self.pos1} 4bets to 22bb, {self.pos2} calls 22bb"
        else:
            return "Preflop action unknown"
    
    def get_action_bb_amount(self, action: str, pot: float) -> float:
        """Get the bb amount for a bet or raise action."""
        action_upper = action.upper()
        
        if action_upper == 'CHECK' or action_upper == 'FOLD':
            return 0.0
        elif action_upper == 'CALL':
            return self.current_street_bet
        elif action_upper.startswith('BET'):
            # Extract amount (already in bb)
            parts = action.split()
            if len(parts) > 1:
                try:
                    bb_amount = float(parts[1])
                    return bb_amount
                except:
                    return 0.0
            return 0.0
        elif action_upper.startswith('RAISE'):
            # Raise amount is already in bb
            parts = action.split()
            if len(parts) > 1:
                try:
                    bb_amount = float(parts[1])
                    return bb_amount
                except:
                    return 0.0
            return 0.0
        elif action_upper == 'ALLIN':
            return self.effective_stack
        
        return 0.0
    
    def format_action_name(self, action: str, include_bb: bool = True) -> str:
        """Format action name for display with optional bb amount."""
        action_upper = action.upper()
        pot = self.get_current_pot()
        
        if action_upper == 'CHECK':
            return 'Check'
        elif action_upper == 'FOLD':
            return 'Fold'
        elif action_upper == 'CALL':
            if include_bb and self.current_street_bet > 0:
                return f'Call ({self.current_street_bet:.1f}bb)'
            return 'Call'
        elif action_upper.startswith('BET'):
            # Extract amount (in bb)
            parts = action.split()
            if len(parts) > 1:
                try:
                    bb_amount = float(parts[1])
                    # Check if it's all-in (close to effective stack)
                    if bb_amount >= self.effective_stack - 0.5:
                        return f'All-in ({self.effective_stack:.1f}bb)'
                    else:
                        # Calculate percentage of pot
                        pct = (bb_amount / pot) * 100
                        
                        # Detect standard bet sizes (accounting for rounding)
                        # Check if it's close to 33%, 50%, or 100%
                        if 30 <= pct <= 40:
                            display_pct = 33
                        elif 45 <= pct <= 55:
                            display_pct = 50
                        elif 95 <= pct <= 115:
                            display_pct = 100
                        else:
                            # Use actual percentage for non-standard sizes
                            display_pct = int(pct)
                        
                        if include_bb:
                            return f'Bet {display_pct}% ({bb_amount:.1f}bb)'
                        return f'Bet {display_pct}%'
                except:
                    return action.title()
        elif action_upper.startswith('RAISE'):
            parts = action.split()
            if len(parts) > 1:
                try:
                    bb_amount = float(parts[1])
                    # Check if it's all-in
                    if bb_amount >= self.effective_stack - 0.5:
                        return f'All-in ({self.effective_stack:.1f}bb)'
                    else:
                        # Calculate percentage of pot after call
                        pot_after_call = pot + self.current_street_bet
                        # The raise amount is what's added on top of the call
                        raise_amount = bb_amount - self.current_street_bet
                        pct = (raise_amount / pot_after_call) * 100
                        
                        # Detect standard raise sizes
                        if 70 <= pct <= 80:
                            display_pct = 75
                        elif 95 <= pct <= 105:
                            display_pct = 100
                        else:
                            display_pct = int(pct)
                        
                        if include_bb:
                            return f'Raise {display_pct}% to {bb_amount:.1f}bb'
                        return f'Raise {display_pct}%'
                except:
                    return action.title()
        elif action_upper == 'ALLIN':
            return f'All-in ({self.effective_stack:.1f}bb)'
        
        return action.title()
    
    def get_overall_strategy(self) -> Dict[str, float]:
        """Calculate overall strategy frequencies across all hands."""
        if 'strategy' not in self.current_node or 'strategy' not in self.current_node['strategy']:
            return {}
        
        hand_strategies = self.current_node['strategy']['strategy']
        if not hand_strategies:
            return {}
        
        # Get actions from current node
        actions = self.current_node.get('actions', [])
        if not actions:
            return {}
        
        # Initialize counters
        action_weights = defaultdict(float)
        total_weight = 0
        
        # Sum up frequencies across all hands
        for hand, strategy in hand_strategies.items():
            if isinstance(strategy, list) and len(strategy) == len(actions):
                for i, freq in enumerate(strategy):
                    action_weights[actions[i]] += freq
                total_weight += 1
        
        # Calculate averages
        if total_weight > 0:
            return {action: weight / total_weight for action, weight in action_weights.items()}
        return {}
    
    def display_action_history(self):
        """Display preflop and postflop action history."""
        print("\n📜 ACTION HISTORY:")
        print("-" * 80)
        
        # Preflop
        print(f"Preflop: {self.get_preflop_history()}")
        print(f"Flop: {self.flop}")
        
        # Postflop actions
        if self.history:
            print("\nPostflop:")
            # Reset street bet tracking for history display
            temp_street_bet = 0.0
            temp_pot = self.initial_pot
            
            for i, (pos, action) in enumerate(self.history):
                # Update street bet for proper formatting
                self.current_street_bet = temp_street_bet
                formatted_action = self.format_action_name(action)
                print(f"  {i+1}. {pos}: {formatted_action}")
                
                # Update temp values for next action
                action_bb = self.get_action_bb_amount(action, temp_pot)
                if action.upper().startswith('BET') or action.upper().startswith('RAISE'):
                    temp_street_bet = action_bb
                elif action.upper() == 'CALL':
                    temp_pot += temp_street_bet * 2  # Both players contributed
                    temp_street_bet = 0.0
                elif action.upper() == 'CHECK' and i > 0 and self.history[i-1][1].upper() == 'CHECK':
                    # Both checked, reset street bet
                    temp_street_bet = 0.0
        else:
            print("\nPostflop: (No actions yet)")
        
        # Restore current street bet
        self.update_current_street_bet()
    
    def update_current_street_bet(self):
        """Update current street bet based on action history."""
        self.current_street_bet = 0.0
        
        # Look for the last bet/raise in the current betting round
        for pos, action in reversed(self.history):
            if action.upper().startswith('BET') or action.upper().startswith('RAISE'):
                # Extract bb amount directly from action
                parts = action.split()
                if len(parts) > 1:
                    try:
                        self.current_street_bet = float(parts[1])
                        break
                    except:
                        pass
            elif action.upper() == 'CALL':
                # If someone called, the betting round might be over
                self.current_street_bet = 0.0
                break
            elif action.upper() == 'CHECK':
                # Keep looking back
                continue
            elif action.upper() == 'FOLD':
                break
    
    def display_node_overview(self):
        """Display comprehensive overview of current node."""
        print("\n" + "="*100)
        
        # Always show action history
        self.display_action_history()
        
        print("="*100)
        
        # Basic node info
        player = self.current_node.get('player', 0)
        position = self.get_player_position(player)
        pot = self.get_current_pot()
        
        print(f"\n🎭 {position} to Act")
        print(f"💰 Pot Size: {pot:.1f}bb")
        
        # Update current street bet
        self.update_current_street_bet()
        
        # Overall strategy frequencies
        overall_strategy = self.get_overall_strategy()
        if overall_strategy:
            print(f"\n📊 OVERALL STRATEGY FREQUENCIES:")
            print("-" * 50)
            sorted_actions = sorted(overall_strategy.items(), key=lambda x: x[1], reverse=True)
            for action, freq in sorted_actions:
                formatted_action = self.format_action_name(action)
                bar = "█" * int(freq * 40)
                print(f"  {formatted_action:<20} {freq:6.1%} {bar}")
        
        # Available actions
        if 'childrens' in self.current_node and self.current_node['childrens']:
            children = list(self.current_node['childrens'].keys())
            print(f"\n🎬 Available Actions:")
            for i, child in enumerate(children, 1):
                formatted = self.format_action_name(child)
                print(f"  {i}. {formatted}")
            print(f"\n💡 Enter number (1-{len(children)}) or action name")
        else:
            print("\n🏁 Terminal node - no more actions")
    
    def display_hands_table(self):
        """Display all hands in table format."""
        if 'strategy' not in self.current_node or 'strategy' not in self.current_node['strategy']:
            print("❌ No hand-specific strategies at this node")
            return
        
        hand_strategies = self.current_node['strategy']['strategy']
        actions = self.current_node.get('actions', [])
        
        if not actions or not hand_strategies:
            print("❌ No strategies available")
            return
        
        # Update current street bet for proper formatting
        self.update_current_street_bet()
        
        # Format action names (without bb for table headers to save space)
        formatted_actions = [self.format_action_name(a, include_bb=False) for a in actions]
        
        # Create header
        print(f"\n🃏 HAND STRATEGIES ({len(hand_strategies)} hands):")
        print("=" * 100)
        
        # Calculate column widths
        hand_width = 8
        action_widths = [max(len(a), 7) for a in formatted_actions]
        
        # Print header
        header = f"{'Hand':<{hand_width}}"
        for i, action in enumerate(formatted_actions):
            header += f" | {action:^{action_widths[i]}}"
        print(header)
        print("-" * len(header))
        
        # Sort hands for consistent display
        sorted_hands = sorted(hand_strategies.items())
        
        # Print each hand
        for hand, strategy in sorted_hands:
            if isinstance(strategy, list) and len(strategy) == len(actions):
                row = f"{hand:<{hand_width}}"
                for i, freq in enumerate(strategy):
                    freq_str = f"{freq:.1%}" if freq > 0 else "-"
                    row += f" | {freq_str:^{action_widths[i]}}"
                print(row)
    
    def navigate_to_action(self, action: str) -> bool:
        """Navigate to a specific action using number or action name."""
        available_actions = list(self.current_node.get('childrens', {}).keys())
        if not available_actions:
            print("❌ No actions available")
            return False
        
        matched_action = None
        
        # First, try as a number (1-based index)
        try:
            action_num = int(action)
            if 1 <= action_num <= len(available_actions):
                matched_action = available_actions[action_num - 1]
            else:
                print(f"❌ Invalid number. Choose 1-{len(available_actions)}")
                return False
        except ValueError:
            # Not a number, try to match action name
            action_upper = action.upper()
            
            # Direct match with raw action names
            for avail_action in available_actions:
                if action_upper == avail_action.upper():
                    matched_action = avail_action
                    break
            
            # If no direct match, try fuzzy matching
            if not matched_action:
                # Common action mappings
                if action_upper in ['CHECK', 'C']:
                    for avail in available_actions:
                        if 'CHECK' in avail.upper():
                            matched_action = avail
                            break
                elif action_upper in ['BET', 'B']:
                    # Find any bet action
                    for avail in available_actions:
                        if 'BET' in avail.upper():
                            matched_action = avail
                            break
                elif action_upper in ['CALL']:
                    for avail in available_actions:
                        if 'CALL' in avail.upper():
                            matched_action = avail
                            break
                elif action_upper in ['FOLD', 'F']:
                    for avail in available_actions:
                        if 'FOLD' in avail.upper():
                            matched_action = avail
                            break
                elif action_upper in ['RAISE', 'R']:
                    for avail in available_actions:
                        if 'RAISE' in avail.upper():
                            matched_action = avail
                            break
                elif action_upper in ['ALLIN', 'ALL-IN', 'ALL IN', 'A']:
                    for avail in available_actions:
                        if 'ALLIN' in avail.upper():
                            matched_action = avail
                            break
                        # Check if it's a bet/raise close to effective stack
                        elif avail.upper().startswith('BET') or avail.upper().startswith('RAISE'):
                            parts = avail.split()
                            if len(parts) > 1:
                                try:
                                    bb_amount = float(parts[1])
                                    if bb_amount >= self.effective_stack - 0.5:
                                        matched_action = avail
                                        break
                                except:
                                    pass
        
        if not matched_action:
            print(f"❌ Action '{action}' not available")
            self.show_available_actions_help(available_actions)
            return False
        
        # Record history with position name
        player = self.current_node.get('player', 0)
        position = self.get_player_position(player)
        self.history.append((position, matched_action))
        
        # Navigate
        self.current_path.append(matched_action)
        self.current_node = self.current_node['childrens'][matched_action]
        
        # Update current street bet
        self.update_current_street_bet()
        
        formatted_action = self.format_action_name(matched_action)
        print(f"✅ {position} takes action: {formatted_action}")
        return True
    
    def show_available_actions_help(self, available_actions: List[str]):
        """Show help for available actions with numbers."""
        print(f"\n💡 Available actions:")
        # Update current street bet for proper formatting
        self.update_current_street_bet()
        for i, action in enumerate(available_actions, 1):
            formatted = self.format_action_name(action)
            print(f"  {i}. {formatted}")
        print(f"\nYou can enter the number (1-{len(available_actions)}) or action name")
    
    def reset(self):
        """Reset to root node."""
        self.current_path = []
        self.current_node = self.data
        self.history = []
        print("🔄 Reset to starting position")
    
    def go_back(self):
        """Go back one action."""
        if not self.current_path:
            print("❌ Already at starting position")
            return
        
        self.current_path.pop()
        self.history.pop()
        
        # Navigate back
        node = self.data
        for action in self.current_path:
            node = node['childrens'][action]
        self.current_node = node
        
        print("⬅️  Went back one action")
    
    def search_hand(self, hand: str):
        """Search for a specific hand and show its strategy."""
        if 'strategy' not in self.current_node or 'strategy' not in self.current_node['strategy']:
            print("❌ No hand-specific strategies at this node")
            return
        
        hand_strategies = self.current_node['strategy']['strategy']
        hand = hand.upper()
        
        if hand not in hand_strategies:
            # Try to find similar hands
            similar = [h for h in hand_strategies.keys() if hand[:2] in h or hand[-2:] in h]
            print(f"❌ Hand {hand} not found")
            if similar:
                print(f"💡 Similar hands: {', '.join(similar[:10])}")
            return
        
        strategy = hand_strategies[hand]
        actions = self.current_node.get('actions', [])
        
        print(f"\n🃏 Strategy for {hand}:")
        print("-" * 40)
        
        # Update current street bet for proper formatting
        self.update_current_street_bet()
        
        if isinstance(strategy, list) and len(strategy) == len(actions):
            sorted_actions = sorted(zip(actions, strategy), key=lambda x: x[1], reverse=True)
            for action, freq in sorted_actions:
                if freq > 0:
                    formatted_action = self.format_action_name(action)
                    bar = "█" * int(freq * 30)
                    print(f"  {formatted_action:<20} {freq:6.1%} {bar}")
    
    def show_help(self):
        """Show help information."""
        print("\n📚 HELP - Available Commands:")
        print("="*60)
        print("  1, 2, 3...   - Select action by number")
        print("  check        - Take check action (shortcut: c)")
        print("  bet          - Take any bet action (shortcut: b)")
        print("  call         - Take call action")
        print("  fold         - Take fold action (shortcut: f)")
        print("  raise        - Take raise action (shortcut: r)")
        print("  allin        - Take all-in action (shortcut: a)")
        print("  all          - Show all hands in table format")
        print("  hand [HAND]  - Show strategy for specific hand (e.g., 'hand AhKh')")
        print("  back         - Go back one action")
        print("  reset        - Go back to starting position")
        print("  help         - Show this help")
        print("  quit         - Exit")
        print("\n💡 Tips:")
        print("  - Easiest way: just enter the number (1, 2, 3, etc.)")
        print("  - Actions are case-insensitive")
        print("  - Hand format: rank+suit (e.g., 'AhKh', 'JdJc', 'Ts9s')")
    
    def run(self):
        """Main interactive loop."""
        print("🎮 POKER SOLVER STRATEGY EXPLORER")
        print(f"📄 {self.pos1} vs {self.pos2} - {self.pot_type}")
        print("Type 'help' for commands\n")
        
        while True:
            # Display current node
            self.display_node_overview()
            
            # Get user input
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'q', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif command.lower() in ['help', 'h', '?']:
                    self.show_help()
                    
                elif command.lower() == 'reset':
                    self.reset()
                    
                elif command.lower() == 'back':
                    self.go_back()
                    
                elif command.lower() == 'all':
                    self.display_hands_table()
                    
                elif command.lower().startswith('hand '):
                    hand = command[5:].strip()
                    self.search_hand(hand)
                    
                else:
                    # Try to navigate to this action
                    self.navigate_to_action(command)
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python explore.py <solver_output.json>")
        print("\nExample:")
        print("  python explore.py Qc5h4s_UTG_SB_SRP_rb_np_bh_semi.json")
        print("  python explore.py solver_outputs/Tc8s2s_LJ_BTN_SRP_tt_np_mh_dry.json")
        
        # List some available files
        outputs_dir = os.path.join(os.path.dirname(__file__), "solver_outputs")
        if os.path.exists(outputs_dir):
            files = [f for f in os.listdir(outputs_dir) if f.endswith('.json')]
            if files:
                print(f"\nAvailable files ({len(files)} total):")
                for f in sorted(files)[:10]:
                    size_mb = os.path.getsize(os.path.join(outputs_dir, f)) / (1024 * 1024)
                    print(f"  {f} ({size_mb:.1f} MB)")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
        sys.exit(1)
    
    filename = sys.argv[1]
    explorer = StrategyExplorer(filename)
    explorer.run()


if __name__ == "__main__":
    main() 