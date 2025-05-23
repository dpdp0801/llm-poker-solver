#!/usr/bin/env python3
"""Python bridge for the TexasSolver integration.

This module provides a high-level interface to the TexasSolver, allowing
for more realistic strategy generation, game tree navigation, and
proper handling of sequential decision points in poker.
"""

import os
import json
import subprocess
import tempfile
import sys
from typing import Dict, List, Tuple, Any, Optional, Set, Union


class SolverConfig:
    """Configuration for the TexasSolver."""

    def __init__(
        self,
        iterations: int = 1000,
        accuracy: float = 0.001,
        board: str = None,
        pot_sizes: List[int] = None,
        effective_stack: int = 200,
        range_oop: str = None,
        range_ip: str = None,
        bet_sizes: Dict[str, List[float]] = None,
        raise_sizes: Dict[str, List[float]] = None,
        donk_sizing: Dict[str, List[float]] = None,
        allin_threshold: float = 0.8,
    ):
        """Initialize solver configuration.

        Parameters
        ----------
        iterations : int
            Maximum number of iterations (default: 1000)
        accuracy : float
            Target accuracy for early stopping (default: 0.001)
        board : str
            Board cards in string format (e.g., "Ah7d2c")
        pot_sizes : List[int]
            Starting pot sizes for both players (default: [100, 100])
        effective_stack : int
            The effective stack size in big blinds (default: 200)
        range_oop : str
            Hand range for out-of-position player
        range_ip : str
            Hand range for in-position player
        bet_sizes : Dict[str, List[float]]
            Betting sizes as percentages of the pot for each street and position
        raise_sizes : Dict[str, List[float]]
            Raise sizes as percentages of the pot (e.g. ``250`` means 2.5x pot)
        donk_sizing : Dict[str, List[float]]
            Donk bet sizes as % of pot for each street
        """
        self.iterations = iterations
        self.accuracy = accuracy
        self.board = board
        self.pot_sizes = pot_sizes or [100, 100]
        self.effective_stack = effective_stack
        self.range_oop = range_oop
        self.range_ip = range_ip
        
        # Default bet sizing if not provided. Each list may include the string
        # "allin" to allow the solver to consider shoving.
        self.bet_sizes = bet_sizes or {
            'ip_flop': [33, 50, 75, 125, 'allin'],
            'ip_turn': [33, 50, 75, 125, 'allin'],
            'ip_river': [33, 50, 75, 125, 'allin'],
            'oop_flop': [33, 50, 75, 125, 'allin'],
            'oop_turn': [33, 50, 75, 125, 'allin'],
            'oop_river': [33, 50, 75, 125, 'allin'],
        }

        # Raise sizes are also percentages of the pot. Values like 250 mean
        # a 2.5× raise size.
        self.raise_sizes = raise_sizes or {
            'ip_flop': [50, 100, 'allin'],
            'ip_turn': [50, 100, 'allin'],
            'ip_river': [50, 100, 'allin'],
            'oop_flop': [50, 100, 'allin'],
            'oop_turn': [50, 100, 'allin'],
            'oop_river': [50, 100, 'allin'],
        }
        
        self.donk_sizing = donk_sizing or {
            'turn': [33, 50],
            'river': [33, 50],
        }

        self.allin_threshold = allin_threshold

    def format_board(self) -> str:
        """Format board cards for the solver."""
        if not self.board:
            return ""
        board = self.board.strip()
        cards = []
        for i in range(0, len(board), 2):
            if i+1 < len(board):
                cards.append(f"{board[i:i+2]}")
        return ",".join(cards)

    def to_commands(self) -> List[str]:
        """Convert config to solver commands."""
        commands = []
        
        # Set basic parameters
        commands.append(f"set_pot {self.pot_sizes[0]}")
        commands.append(f"set_effective_stack {self.effective_stack}")
        
        # Set board if available
        if self.board:
            commands.append(f"set_board {self.format_board()}")
        
        # Set ranges
        if self.range_ip:
            commands.append(f"set_range_ip {self.range_ip}")
        if self.range_oop:
            commands.append(f"set_range_oop {self.range_oop}")
        
        # Set bet sizes for each street and position
        for pos in ['ip', 'oop']:
            for street in ['flop', 'turn', 'river']:
                key = f"{pos}_{street}"
                if key in self.bet_sizes:
                    sizes = self.bet_sizes[key]
                    nums = [str(s) for s in sizes if s != 'allin']
                    if nums:
                        bet_str = ",".join(nums)
                        commands.append(f"set_bet_sizes {pos},{street},bet,{bet_str}")
                    if 'allin' in sizes:
                        commands.append(f"set_bet_sizes {pos},{street},allin")
                
                if key in self.raise_sizes:
                    sizes = self.raise_sizes[key]
                    nums = [str(s) for s in sizes if s != 'allin']
                    if nums:
                        raise_str = ",".join(nums)
                        commands.append(f"set_bet_sizes {pos},{street},raise,{raise_str}")
                    if 'allin' in sizes:
                        commands.append(f"set_bet_sizes {pos},{street},allin")
        
        # Set donk bet sizes
        for street in ['turn', 'river']:
            if street in self.donk_sizing:
                donk_str = ",".join(str(size) for size in self.donk_sizing[street])
                commands.append(f"set_donk_sizes {street},{donk_str}")

        commands.append(f"set_allin_threshold {self.allin_threshold}")

        # Build tree and set solver parameters
        commands.append("build_tree")
        commands.append(f"set_accuracy {self.accuracy}")
        commands.append(f"set_max_iterations {self.iterations}")
        
        return commands


class TexasSolverBridge:
    """Bridge to the TexasSolver CLI."""

    def __init__(self, solver_path: Optional[str] = None):
        """Initialize the solver bridge.

        Parameters
        ----------
        solver_path : str, optional
            Path to the TexasSolver binary
        """
        if solver_path is None:
            # Default path relative to project root - use the bundled release
            # shipped under ``texas_solver``. This avoids requiring users to
            # build the solver themselves under ``external``.
            solver_path = os.path.join('texas_solver', 'console_solver')
        self.solver_path = solver_path
        self._result_path = None
        self._result_data = None
        self._config = None

    def set_config(self, config: SolverConfig) -> None:
        """Set the solver configuration.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver
        """
        self._config = config

    def run_solver(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the solver with the current configuration.

        Parameters
        ----------
        output_path : str, optional
            Path to save the solver results. If None, a temporary file is used.

        Returns
        -------
        Dict[str, Any]
            The solver results
        """
        if self._config is None:
            raise ValueError("Config not set, call set_config first")

        if not os.path.exists(self.solver_path):
            raise FileNotFoundError(f"Solver binary not found at {self.solver_path}")

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.path.join(root_dir, 'solver_inputs')
        output_dir = os.path.join(root_dir, 'solver_outputs')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if output_path is None:
            output_path = os.path.join(output_dir, 'output_result.json')

        self._result_path = output_path
        commands = self._config.to_commands()
        commands.append("start_solve")
        commands.append(f"dump_result {output_path}")

        fd, input_path = tempfile.mkstemp(dir=input_dir, suffix='.txt')
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(commands))

        process = subprocess.run(
            [self.solver_path, '-i', input_path],
            text=True,
            capture_output=True
        )

        # Display any output the solver produced so the user can see progress
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr, file=sys.stderr)

        if process.returncode != 0:
            raise RuntimeError(f"Solver failed: {process.stderr}")

        # Read results
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                self._result_data = json.load(f)
            return self._result_data
        else:
            raise FileNotFoundError(f"Solver did not produce output at {output_path}")

    def get_ev(self) -> float:
        """Get the expected value from the solver results.

        Returns
        -------
        float
            The expected value
        """
        if self._result_data is None:
            raise ValueError("No solver results available")
        return self._result_data.get("ev", 0.0)
    
    def get_strategy(self, player: str, node_path: str) -> Dict[str, Any]:
        """Get the strategy for a specific player at a specific node.

        Parameters
        ----------
        player : str
            'ip' for in-position player or 'oop' for out-of-position player
        node_path : str
            Path to the node, e.g., 'flop' for the flop betting round

        Returns
        -------
        Dict[str, Any]
            The strategy at the specified node
        """
        if self._result_data is None:
            raise ValueError("No solver results available")
        
        # Get strategy based on player and node path
        if player.lower() == 'ip':
            strategy = self._navigate_ip_strategy(node_path)
        elif player.lower() == 'oop':
            strategy = self._navigate_oop_strategy(node_path)
        else:
            raise ValueError(f"Invalid player: {player}, must be 'ip' or 'oop'")
            
        return strategy
    
    def _navigate_ip_strategy(self, node_path: str) -> Dict[str, Any]:
        """Navigate to IP player's strategy at the specified node path.

        Parameters
        ----------
        node_path : str
            Path to the node, e.g., 'flop.bet.call'

        Returns
        -------
        Dict[str, Any]
            The strategy at the specified node
        """
        # This is a simplified implementation - in a real system we'd have 
        # proper game tree navigation
        if not self._result_data or 'strategy' not in self._result_data:
            raise ValueError("No strategy data available")
            
        # Basic implementation for flop betting
        if node_path == 'flop':
            return self._result_data.get('ip_flop_strategy', {})
        
        return {}
    
    def _navigate_oop_strategy(self, node_path: str) -> Dict[str, Any]:
        """Navigate to OOP player's strategy at the specified node path.

        Parameters
        ----------
        node_path : str
            Path to the node, e.g., 'flop.check'

        Returns
        -------
        Dict[str, Any]
            The strategy at the specified node
        """
        # This is a simplified implementation - in a real system we'd have 
        # proper game tree navigation
        if not self._result_data or 'strategy' not in self._result_data:
            raise ValueError("No strategy data available")
            
        # Basic implementation for flop betting
        if node_path == 'flop':
            return self._result_data.get('oop_flop_strategy', {})
        
        return {}

    def analyze_strategy(self, player: str, node_path: str) -> Dict[str, float]:
        """Analyze the strategy at a specific node.

        Parameters
        ----------
        player : str
            'ip' for in-position player or 'oop' for out-of-position player
        node_path : str
            Path to the node, e.g., 'flop' for the flop betting round

        Returns
        -------
        Dict[str, float]
            Frequency of each action at the node
        """
        if not self._result_data:
            return {}
            
        # For default root node strategies
        if node_path == "flop":
            if player.lower() == "oop":
                # OOP player is typically player 1 in the root node
                if "strategy" in self._result_data:
                    strategy = self._result_data["strategy"]
                else:
                    return {}
            else:
                # IP player acts in response to OOP in the root children
                if "childrens" in self._result_data and "CHECK" in self._result_data["childrens"]:
                    check_node = self._result_data["childrens"]["CHECK"]
                    if "strategy" in check_node:
                        strategy = check_node["strategy"]
                    else:
                        return {}
                else:
                    return {}
        else:
            # Try to navigate to the specified node path
            strategy = self._navigate_to_node(player, node_path)
            if not strategy:
                return {}
        
        # Extract action frequencies
        if "actions" not in strategy or "strategy" not in strategy:
            return {}
            
        actions = strategy.get("actions", [])
        combos = strategy.get("strategy", {})
        if not actions or not combos:
            return {}
            
        # Get frequencies for each action
        totals = {a: 0.0 for a in actions}
        n = len(combos)
        
        if n == 0:
            return {}
            
        # Check if this is a uniform strategy (all hands have the same action)
        uniform_strategy = True
        first_values = None
        
        for hand, probs in combos.items():
            if first_values is None:
                first_values = probs
            elif probs != first_values:
                uniform_strategy = False
                break
        
        if uniform_strategy and first_values:
            # If all hands have the same strategy, just return that
            return dict(zip(actions, first_values))
        
        # Otherwise calculate the average frequencies
        for hand, probs in combos.items():
            for i, action in enumerate(actions):
                if i < len(probs):
                    totals[action] += probs[i]
        
        for a in totals:
            totals[a] /= n
            
        return totals
    
    def _navigate_to_node(self, player: str, node_path: str) -> Dict[str, Any]:
        """Navigate to a specific node in the game tree.
        
        Parameters
        ----------
        player : str
            'ip' or 'oop'
        node_path : str
            Path to navigate, e.g., 'flop.check.bet'
            
        Returns
        -------
        Dict[str, Any]
            Strategy at the node, or empty dict if not found
        """
        if not self._result_data:
            return {}
            
        parts = node_path.split('.')
        current = self._result_data
        
        for i, part in enumerate(parts):
            if part.lower() == 'flop':
                continue  # Already at the root node for flop
                
            # Navigate based on the action
            if "childrens" not in current:
                return {}
                
            # Try exact match
            action_key = None
            for key in current["childrens"]:
                if key.lower() == part.lower():
                    action_key = key
                    break
                    
            # Try partial match
            if action_key is None:
                for key in current["childrens"]:
                    if part.lower() in key.lower():
                        action_key = key
                        break
            
            if action_key is None:
                return {}
                
            # Move to the next node
            current = current["childrens"][action_key]
            
            # If we're at the target node, return the strategy
            if i == len(parts) - 1:
                if "strategy" in current:
                    return current["strategy"]
                    
        # If we reached a node with a strategy, return it
        if "strategy" in current:
            return current["strategy"]
            
        return {}

    def get_optimal_play(self, player: str, node_path: str, hand: str) -> Dict[str, float]:
        """Get the optimal play for a specific hand at a specific node.

        Parameters
        ----------
        player : str
            'ip' for in-position player or 'oop' for out-of-position player
        node_path : str
            Path to the node, e.g., 'flop' for the flop betting round
        hand : str
            The hand in format like "AcKd"

        Returns
        -------
        Dict[str, float]
            Probability of each action for the specific hand
        """
        if not self._result_data:
            return {}
            
        # Normalize the hand format if needed (e.g., "AcKs" -> "AcKs")
        hand = hand.strip()
        if len(hand) == 4:
            # Check if the hand is in the correct format
            if not (hand[0].upper() in "AKQJT98765432" and 
                   hand[1].lower() in "cdhs" and
                   hand[2].upper() in "AKQJT98765432" and
                   hand[3].lower() in "cdhs"):
                return {}
        else:
            return {}
            
        # Also check for the reversed version of the hand (e.g., "KsAc")
        reverse_hand = hand[2:] + hand[:2]
            
        # Get the strategy at the node
        if node_path == "flop":
            if player.lower() == "oop":
                # OOP player is typically player 1 in the root node
                if "strategy" in self._result_data:
                    strategy = self._result_data["strategy"]
                else:
                    return {}
            else:
                # IP player acts in response to OOP in the root children
                if "childrens" in self._result_data and "CHECK" in self._result_data["childrens"]:
                    check_node = self._result_data["childrens"]["CHECK"]
                    if "strategy" in check_node:
                        strategy = check_node["strategy"]
                    else:
                        return {}
                else:
                    return {}
        else:
            # Try to navigate to the specified node path
            strategy = self._navigate_to_node(player, node_path)
            if not strategy:
                return {}
        
        # Check structure
        if "actions" not in strategy or "strategy" not in strategy:
            return {}
            
        actions = strategy.get("actions", [])
        combos = strategy.get("strategy", {})
        
        if not actions or not combos:
            return {}
            
        # Look for the hand in the combos
        if hand in combos:
            return dict(zip(actions, combos[hand]))
        elif reverse_hand in combos:
            return dict(zip(actions, combos[reverse_hand]))
        else:
            # If the exact hand isn't found, see if there are any hands with the same cards
            # but different suits that match the board better
            if len(hand) == 4:
                # Extract ranks
                rank1, rank2 = hand[0].upper(), hand[2].upper()
                for combo_hand in combos:
                    if len(combo_hand) == 4:
                        combo_rank1, combo_rank2 = combo_hand[0].upper(), combo_hand[2].upper()
                        if (rank1 == combo_rank1 and rank2 == combo_rank2) or \
                           (rank1 == combo_rank2 and rank2 == combo_rank1):
                            return dict(zip(actions, combos[combo_hand]))
            
        return {} 