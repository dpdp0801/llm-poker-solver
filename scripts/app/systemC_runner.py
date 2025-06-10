#!/opt/miniconda3/envs/poker-llm/bin/python
"""
SystemC Poker Model Runner

Interactive runner for the SystemC model that provides GTO poker decisions
with TOOL_TAGS analysis including board texture, ranges, and advantages.

SystemC uses TOOL_TAGS which provide:
- board_texture: Texture analysis (dry, semi-wet, wet)
- hero_range/villain_range: Preflop ranges
- range_adv: Range advantage analysis
- nut_adv: Nut advantage analysis
- hero_hand_category: Hand category analysis
- hero_hand_ranking: Percentile ranking

REQUIREMENTS:
- Python 3.8+
- PyTorch, Transformers, PEFT libraries
- Internet connection (first run downloads Llama-3-8B base model ~16GB)
- 16GB+ RAM recommended, GPU optional but faster

USAGE:
    python systemC_runner.py                    # Interactive mode
    python systemC_runner.py --gpu-info         # Show device info
    python systemC_runner.py --model-path PATH  # Custom model path
    python systemC_runner.py --prompt-file FILE # Process prompt from file

The model expects prompts in the format with TOOL_TAGS:

### HAND_META
game: cash
seats: 8-max
hero_pos: BTN
hero_hand: Ad 8d
villain_profile: balanced

### HISTORY_PREFLOP
preflop: UTG folds, ... BTN raises 2.5bb, SB folds, BB calls

### HISTORY 1
flop: (Kc Qs 6s)    pot: 5.5bb
stacks: 97.5bb
actions: BB checks

### DECISION 1
street: flop
pot: 5.5bb
to_act: HERO
legal: [check,bet 33%,bet 50%,bet 100%,allin]

### TOOL_TAGS
board_texture: dry
hero_range: [22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,87s,76s,A2o+,K8o+,Q9o+,J9o+,T9o]
villain_range: [22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,65s,A5o+,K9o+,Q9o+,J9o+,T9o]
range_adv: hero    # eq_gap = +2.3%
nut_adv: hero    # nut_gap = +1.5%
hero_hand_category: top_pair
hero_hand_ranking: top 15%

Choose the best action...
"""

import os
import sys
from typing import List, Dict, Optional
import argparse
import tempfile
import json
import warnings

# Suppress PyTorch LoRA loading warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*Did you mean to pass `assign=True`.*")

# Try to import required ML libraries
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM
    )
    from peft import PeftModel, PeftConfig
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)

# Add paths for SystemB functions - use absolute paths to work from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up 2 levels from scripts/app
sys.path.append(os.path.join(project_root, 'scripts', 'training_data', 'systemB'))
sys.path.append(os.path.join(project_root, 'scripts', 'training_data', 'systemC'))
sys.path.append(os.path.join(project_root, 'scripts', 'solver'))
sys.path.append(os.path.join(project_root, 'scripts', 'tools'))

try:
    # Import SystemB functions for TOOL_TAGS generation
    from generate_poker_training_data import (
        get_ranges_from_preflop_chart,
        convert_board_texture,
        analyze_hand_category,
        calculate_equity_and_advantages,
        determine_ip_oop
    )
    SYSTEMB_AVAILABLE = True
except ImportError as e:
    SYSTEMB_AVAILABLE = False
    SYSTEMB_MISSING = str(e)



class SystemCRunner:
    """Interactive runner for SystemC model with TOOL_TAGS."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """Initialize the model runner.
        
        Parameters
        ----------
        model_path : str
            Path to the fine-tuned model directory
        base_model : str
            Base model name for inference
        """
        self.model_path = model_path
        self.base_model_name = base_model
        self.model = None
        self.tokenizer = None
        
        # Standard poker values
        self.positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
        self.villain_profiles = [
            "balanced",
            "tag (tight-aggressive)", 
            "lag (loose-aggressive)",
            "nit (super-tight)",
            "station (calling-station)",
            "maniac (loose, hyper-aggressive)",
            "whale (loose-passive)"
        ]
        
    def load_model(self):
        """Load model - copied EXACTLY from evaluation/create_evaluation.py"""
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
        print(f"Loading LoRA adapter: {self.model_path}")
        try:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            print(f"✅ Successfully loaded SystemC model")
        except Exception as e:
            print(f"❌ Error loading LoRA adapter {self.model_path}: {e}")
            print("Continuing with base model only...")
    
    def show_gpu_info(self):
        """Display detailed device information."""
        import platform
        
        print("\n🖥️  DEVICE INFORMATION:")
        print("=" * 40)
        
        # Show system info
        print(f"System: {platform.system()} {platform.machine()}")
        print(f"Python: {platform.python_version()}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print("\n🚀 CUDA DEVICES:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                if i == torch.cuda.current_device():
                    print(f"    Status: ACTIVE ✅")
                    print(f"    Memory Used: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    print(f"    Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                else:
                    print(f"    Status: Available")
        else:
            print("\n❌ CUDA: Not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("\n🍎 MPS: Available ✅")
                print("  Apple Silicon GPU acceleration enabled")
            else:
                print("\n🍎 MPS: Not available")
        
        # Show recommended device
        if torch.cuda.is_available():
            print(f"\n✅ Recommended device: CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"\n✅ Recommended device: MPS (Apple Silicon)")
        else:
            print(f"\n⚠️  Recommended device: CPU (slower)")
        
        print("=" * 40)
    
    def get_user_input(self, prompt: str, options: List[str] = None, allow_empty: bool = False) -> str:
        """Get validated user input."""
        while True:
            if options:
                print(f"\n{prompt}")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                print("  0. Custom input")
                
                try:
                    choice = input("\nEnter choice (number or custom): ").strip()
                    
                    if choice == "0":
                        custom = input("Enter custom value: ").strip()
                        if custom or allow_empty:
                            return custom
                        print("❌ Custom input cannot be empty")
                        continue
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(options):
                        return options[choice_num - 1]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(options)}, or 0 for custom")
                        continue
                        
                except ValueError:
                    # Try treating as direct input
                    if choice and (choice in options or allow_empty):
                        return choice
                    print("❌ Invalid input. Please enter a number or valid option.")
                    continue
            else:
                response = input(f"\n{prompt}: ").strip()
                if response or allow_empty:
                    return response
                print("❌ Input cannot be empty")
    
    def generate_tool_tags(self, scenario: Dict[str, str]) -> str:
        """Generate TOOL_TAGS for the given scenario."""
        tool_tags = []
        
        try:
            # 1. BOARD TEXTURE ANALYSIS
            flop = scenario.get('flop', '')
            
            # Clean up flop format
            flop_clean = flop.replace('(', '').replace(')', '').strip()
            flop_cards = flop_clean.split()
            flop_str = ' '.join(flop_cards)
            
            if len(flop_cards) >= 3:
                try:
                    # Import board analysis functions (path already added at module level)
                    from solver_inputer import analyze_suits, analyze_pairing, analyze_hirank, analyze_connectivity
                    
                    suits = analyze_suits(flop_cards)
                    pairing = analyze_pairing(flop_cards)
                    hirank = analyze_hirank(flop_cards)
                    connectivity = analyze_connectivity(flop_cards)
                    
                    # Convert to full descriptions
                    board_texture = convert_board_texture(suits, pairing, hirank, connectivity)
                except Exception as bt_error:
                    board_texture = f"ERROR: {str(bt_error)}"
            else:
                board_texture = "ERROR: insufficient board cards"
            
            # 2. GET HERO/VILLAIN POSITIONS AND RANGES
            hero_pos = scenario['hero_pos']
            
            # Determine villain position based on preflop history
            preflop_history = scenario.get('preflop_history', '')
            
            participating_positions = []
            positions = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
            for pos in positions:
                if pos in preflop_history and 'folds' not in preflop_history.split(pos)[1].split(',')[0]:
                    participating_positions.append(pos)
            
            # Villain is the other participating position
            villain_pos = None
            for pos in participating_positions:
                if pos != hero_pos:
                    villain_pos = pos
                    break
            
            if not villain_pos:
                villain_pos = "BB" if hero_pos != "BB" else "BTN"
            
            # Get ranges from preflop scenario
            try:
                # Import the equity calculator directly (path already added at module level)
                from equity_calculator import get_ranges_from_scenario
                
                # Convert preflop history to simple scenario format
                simple_scenario = ""
                actions = preflop_history.split(',')
                active_actions = []
                
                # Track raise sequence to detect 3bets/4bets
                raise_count = 0
                
                for i, action in enumerate(actions):
                    action = action.strip()
                    
                    # Skip folds
                    if 'folds' in action:
                        continue
                    
                    # Find position and action
                    positions = ["UTG+1", "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"]  # UTG+1 before UTG
                    for pos in positions:
                        if action.startswith(pos):
                            action_part = action[len(pos):].strip()
                            
                            if 'raise' in action_part or 'open' in action_part:
                                raise_count += 1
                                if raise_count == 1:
                                    # First raise
                                    active_actions.append(f"{pos} raise")
                                elif raise_count == 2:
                                    # Second raise = 3bet
                                    active_actions.append(f"{pos} 3bet")
                                elif raise_count == 3:
                                    # Third raise = 4bet
                                    active_actions.append(f"{pos} 4bet")
                                else:
                                    # Beyond 4bet, just call it raise
                                    active_actions.append(f"{pos} raise")
                            elif 'call' in action_part:
                                active_actions.append(f"{pos} call")
                            elif '3bet' in action_part:
                                # Explicit 3bet mentioned
                                raise_count = 2  # Update raise count
                                active_actions.append(f"{pos} 3bet")
                            elif '4bet' in action_part:
                                # Explicit 4bet mentioned
                                raise_count = 3  # Update raise count
                                active_actions.append(f"{pos} 4bet")
                            break
                
                simple_scenario = ', '.join(active_actions)
                
                # Use equity calculator to get ranges directly
                ranges = get_ranges_from_scenario(simple_scenario)
                
                # Extract hero and villain ranges
                hero_range = ranges.get(hero_pos, f"ERROR: {hero_pos} not found")
                villain_range = ranges.get(villain_pos, f"ERROR: {villain_pos} not found")
                
            except Exception as range_error:
                error_msg = str(range_error)
                hero_range = f"ERROR: {error_msg}"
                villain_range = f"ERROR: {error_msg}"
            
            # 3. HERO HAND ANALYSIS
            hero_hand = scenario['hero_hand'].replace(' ', '')
            
            hero_cards = []
            
            # Parse hero hand (e.g., "Ad 8d" -> ["Ad", "8d"])
            if len(hero_hand) >= 4:
                # Handle different formats
                if len(hero_hand) == 4:  # "AdKs"
                    hero_cards = [hero_hand[:2], hero_hand[2:]]
                else:  # "Ad Kh" with space
                    hero_cards = scenario['hero_hand'].split()
            
            # 4. CALCULATE EQUITY AND ADVANTAGES
            range_adv_str = "neutral    # eq_gap = +0.0%"
            nut_adv_str = "neutral    # nut_gap = +0.0%"
            hero_hand_ranking = "unknown"
            
            if len(hero_cards) == 2 and len(flop_cards) >= 3:
                try:
                    equity_results = calculate_equity_and_advantages(
                        hero_pos, villain_pos, hero_range, villain_range, flop_str, hero_cards
                    )
                    
                    # Range advantage
                    range_margin = equity_results['range_adv_value']
                    range_margin_pct = abs(range_margin) * 100
                    
                    if range_margin_pct < 1.0:
                        range_adv_str = "neutral    # eq_gap = +0.0%"
                    else:
                        # Determine edge size based on thresholds from SystemB training data
                        if range_margin_pct < 10.0:
                            size = "small"
                        elif range_margin_pct < 20.0:
                            size = "medium"
                        else:
                            size = "large"
                        
                        if range_margin > 0:
                            range_adv_str = f"{size} edge for hero    # eq_gap = +{range_margin_pct:.1f}%"
                        else:
                            range_adv_str = f"{size} edge for villain    # eq_gap = -{range_margin_pct:.1f}%"
                    
                    # Nut advantage
                    nut_margin = equity_results['nut_adv_value']
                    
                    nut_margin_abs = abs(nut_margin)
                    
                    if nut_margin_abs < 1.0:
                        nut_adv_str = "neutral    # nut_gap = +0.0%"
                    else:
                        # Determine edge size based on thresholds from SystemB training data
                        if nut_margin_abs < 7.0:
                            size = "small"
                        elif nut_margin_abs < 14.0:
                            size = "medium"
                        else:
                            size = "large"
                        
                        if nut_margin > 0:
                            nut_adv_str = f"{size} edge for hero    # nut_gap = +{nut_margin:.1f}%"
                        else:
                            nut_adv_str = f"{size} edge for villain    # nut_gap = -{nut_margin_abs:.1f}%"
                    
                    # Hero hand ranking - Use consistent format with training data
                    percentile = equity_results['hero_percentile']
                    # Always use "top X%" format where X = 100 - percentile (consistent with training data)
                    hero_hand_ranking = f"top {100-percentile:.0f}%"
                    
                except Exception as eq_error:
                    pass  # Keep default values
            
            # 5. HERO HAND CATEGORY
            hero_hand_category = "unknown"
            if len(hero_cards) == 2 and len(flop_cards) >= 3:
                try:
                    # Check if function is actually available
                    if not SYSTEMB_AVAILABLE:
                        hero_hand_category = f"ERROR: SystemB functions not available - {SYSTEMB_MISSING}"
                    else:
                        # Try importing eval7 directly
                        try:
                            import eval7
                        except ImportError as eval_error:
                            hero_hand_category = f"ERROR: eval7 not available - {eval_error}"
                        
                        if hero_hand_category == "unknown":
                            # Use fixed version of SystemB function with correct eval7 strings
                            hero_hand_category = self.analyze_hand_category_fixed(hero_cards, flop_cards)
                
                except Exception as cat_error:
                    hero_hand_category = f"ERROR: analyze_hand_category failed - {str(cat_error)[:30]}..."
            else:
                hero_hand_category = f"ERROR: Invalid cards - hero:{len(hero_cards)} flop:{len(flop_cards)}"
            
            # Build TOOL_TAGS
            tool_tags.extend([
                f"board_texture: {board_texture}",
                f"hero_range: {hero_range}",
                f"villain_range: {villain_range}",
                f"range_adv: {range_adv_str}",
                f"nut_adv: {nut_adv_str}",
                f"hero_hand_category: {hero_hand_category}",
                f"hero_hand_ranking: {hero_hand_ranking}"
            ])
            
        except Exception as e:
            # Show error messages instead of fake values
            tool_tags.extend([
                f"board_texture: ERROR - {str(e)[:50]}...",
                "hero_range: [ERROR: TOOL_TAGS generation failed]",
                "villain_range: [ERROR: TOOL_TAGS generation failed]",
                "range_adv: ERROR - calculation failed",
                "nut_adv: ERROR - calculation failed", 
                "hero_hand_category: ERROR - analysis failed",
                "hero_hand_ranking: ERROR - calculation failed"
            ])
        
        return "\n".join(tool_tags)
    
    def analyze_hand_category_fixed(self, hero_cards: List[str], board_cards: List[str]) -> str:
        """Fixed version of SystemB analyze_hand_category with correct eval7 strings."""
        import eval7
        
        # Convert to eval7 cards for evaluation
        hero_eval = [eval7.Card(c) for c in hero_cards]
        board_eval = [eval7.Card(c) for c in board_cards]
        
        # Get hand strength
        hand_strength = eval7.evaluate(hero_eval + board_eval)
        hand_type = eval7.handtype(hand_strength)
        
        # Extract ranks and suits
        hero_ranks = [c[0] for c in hero_cards]
        hero_suits = [c[1] for c in hero_cards]
        board_ranks = [c[0] for c in board_cards]
        board_suits = [c[1] for c in board_cards]
        all_ranks = hero_ranks + board_ranks
        all_suits = hero_suits + board_suits
        
        # Count occurrences
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        suit_counts = {}
        for suit in all_suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        # Check for specific hand types
        categories = []
        
        # 1. Quads - FIX: eval7 returns "Quads" not "Four of a Kind"
        if hand_type == 'Quads':
            return "quads"
        
        # 2. Full house
        if hand_type == 'Full House':
            return "full house"
        
        # 3. Flush/flush draws
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        
        if flush_suit:
            return "flush"
        
        # Check for flush draws
        hero_flush_draw = False
        backdoor_flush_draw = False
        for suit in ['h', 'd', 'c', 's']:
            hero_suit_count = sum(1 for s in hero_suits if s == suit)
            board_suit_count = sum(1 for s in board_suits if s == suit)
            total_suit = hero_suit_count + board_suit_count
            
            if total_suit == 4:
                hero_flush_draw = True
            elif total_suit == 3 and hero_suit_count >= 1:
                backdoor_flush_draw = True
        
        # 4. Straight/straight draws
        if hand_type == 'Straight':
            if backdoor_flush_draw:
                return "straight with backdoor flush draw"
            else:
                return "straight"
        
        # Check for straight draws
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                       '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        # Get rank values for hero cards and board
        hero_rank_vals = [rank_values[r] for r in hero_ranks]
        board_rank_vals = [rank_values[r] for r in board_ranks]
        all_rank_values = sorted(hero_rank_vals + board_rank_vals)
        unique_ranks = sorted(list(set(all_rank_values)))
        
        # Check for OESD, gutshot, and backdoor straight draws
        oesd = False
        gutshot = False
        backdoor_straight = False
        
        # Check for all possible 5-card straights
        for start in range(1, 11):  # Check all possible straights (A-5 through T-A)
            straight_vals = list(range(start, start + 5))
            # Handle A-5 straight (wheel)
            if start == 1:
                straight_vals = [14, 2, 3, 4, 5]
            
            # Count how many cards we have for this straight
            have = sum(1 for val in straight_vals if val in all_rank_values)
            # Count how many cards we have that include at least one hero card
            hero_contributes = any(val in hero_rank_vals for val in straight_vals if val in all_rank_values)
            
            # Check if this straight involves an Ace
            has_ace_in_straight = 14 in straight_vals and 14 in all_rank_values
            
            if have == 4 and hero_contributes:
                # Need 1 card for straight - check if it's OESD or gutshot
                missing = [val for val in straight_vals if val not in all_rank_values][0]
                
                # Check if we have 4 consecutive cards (OESD)
                temp_unique = sorted(list(set([v for v in all_rank_values if v in straight_vals])))
                is_connected = len(temp_unique) == 4 and temp_unique[-1] - temp_unique[0] == 3
                
                if is_connected and not has_ace_in_straight:
                    # True OESD - 4 connected cards without Ace
                    oesd = True
                else:
                    # Either not connected or involves Ace - it's a gutshot
                    gutshot = True
                    
            elif have == 3 and hero_contributes:
                # Need 2 cards - check if it's backdoor
                # For backdoor straight, we need exactly 3 consecutive cards
                cards_in_straight = sorted([v for v in all_rank_values if v in straight_vals])
                if len(cards_in_straight) == 3:
                    # Check if these 3 cards are consecutive
                    if cards_in_straight[-1] - cards_in_straight[0] == 2 and not has_ace_in_straight:
                        # True backdoor straight - 3 consecutive cards without Ace
                        backdoor_straight = True
                    # If has Ace, no backdoor straight draw (downgrades to nothing)
        
        # If we have straight draws, continue checking other categories
        if hero_flush_draw:
            categories.append("flush draw")
        elif backdoor_flush_draw:
            categories.append("backdoor flush draw")
        
        if oesd:
            categories.append("open ended straight draw")
        elif gutshot:
            categories.append("gutshot")
        elif backdoor_straight:
            categories.append("backdoor straight draw")
        
        # 5. Set/trips - FIX: eval7 returns "Trips" not "Three of a Kind"
        if hand_type == 'Trips':
            # Check if it's a set (pocket pair) or trips
            if hero_ranks[0] == hero_ranks[1]:
                # Pocket pair - it's a set
                if categories:
                    return "set with " + " with ".join(categories)
                else:
                    return "set"
            else:
                # It's trips - need to determine kicker
                kicker = None
                for rank in hero_ranks:
                    if rank_counts[rank] == 1:
                        kicker = rank
                        break
                
                if kicker:
                    kicker_val = rank_values[kicker]
                    if kicker == 'A':
                        kicker_cat = "ace kicker"
                    elif kicker_val >= 10:  # T, J, Q, K
                        kicker_cat = "broadway kicker"
                    else:
                        kicker_cat = "low kicker"
                    
                    if categories:
                        return f"trips with {kicker_cat} with " + " with ".join(categories)
                    else:
                        return f"trips with {kicker_cat}"
                else:
                    if categories:
                        return "trips with " + " with ".join(categories)
                    else:
                        return "trips"
        
        # 6. Two pair
        if hand_type == 'Two Pair':
            board_paired = len([r for r in board_ranks if board_ranks.count(r) >= 2]) > 0
            
            if board_paired and hero_ranks[0] == hero_ranks[1]:
                # This is actually just one pair (pocket pair on paired board)
                # Treat it as one pair based on pocket pair's rank
                # Get distinct board ranks for comparison
                distinct_board_ranks = []
                seen = set()
                for r in board_ranks:
                    if r not in seen:
                        distinct_board_ranks.append(r)
                        seen.add(r)
                
                board_vals = sorted([rank_values[r] for r in distinct_board_ranks], reverse=True)
                pair_val = rank_values[hero_ranks[0]]
                
                # Determine pair position
                if pair_val > board_vals[0]:
                    pair_pos = "overpair"
                elif pair_val == board_vals[0]:
                    pair_pos = "top pair"
                elif len(board_vals) > 1 and pair_val == board_vals[1]:
                    pair_pos = "2nd pair"
                else:
                    pair_pos = "bottom pair"
                
                if categories:
                    return f"{pair_pos} on a paired board with " + " with ".join(categories)
                else:
                    return f"{pair_pos} on a paired board"
            elif board_paired:
                # One card makes two pair - has kicker
                kicker = None
                for rank in hero_ranks:
                    if rank_counts[rank] == 1:
                        kicker = rank
                        break
                
                if kicker:
                    kicker_val = rank_values[kicker]
                    if kicker == 'A':
                        kicker_cat = "ace kicker"
                    elif kicker_val >= 10:
                        kicker_cat = "broadway kicker"
                    else:
                        kicker_cat = "low kicker"
                    
                    # Determine which pair we made
                    board_vals = sorted([rank_values[r] for r in set(board_ranks)], reverse=True)
                    our_pair_rank = None
                    for rank in hero_ranks:
                        if rank in board_ranks:
                            our_pair_rank = rank
                            break
                    
                    if our_pair_rank:
                        our_pair_val = rank_values[our_pair_rank]
                        if our_pair_val == board_vals[0]:
                            pair_pos = "top pair"
                        elif len(board_vals) > 1 and our_pair_val == board_vals[1]:
                            pair_pos = "2nd pair"
                        else:
                            pair_pos = "bottom pair"
                    else:
                        pair_pos = ""
                    
                    result = f"two pair with {kicker_cat} on a paired board"
                    if pair_pos:
                        result += f" ({pair_pos})"
                    
                    if categories:
                        return result + " with " + " with ".join(categories)
                    else:
                        return result
            else:
                # Unpaired board - both cards play, no kicker
                if categories:
                    return "two pair with " + " with ".join(categories)
                else:
                    return "two pair"
        
        # 7. One pair
        if hand_type == 'Pair':
            board_paired = len([r for r in board_ranks if board_ranks.count(r) >= 2]) > 0
            
            # Check if we have a pocket pair
            is_pocket_pair = (hero_ranks[0] == hero_ranks[1])
            
            # Find which card makes the pair
            pair_rank = None
            kicker_rank = None
            
            for rank in hero_ranks:
                if rank_counts[rank] >= 2:
                    pair_rank = rank
                elif not is_pocket_pair:
                    kicker_rank = rank
            
            # Determine pair position
            board_vals = sorted([rank_values[r] for r in set(board_ranks)], reverse=True)
            
            if pair_rank:
                pair_val = rank_values[pair_rank]
                
                # Check if it's overpair (pocket pair higher than board)
                if is_pocket_pair and pair_val > board_vals[0]:
                    pair_pos = "overpair"
                elif pair_val == board_vals[0]:
                    pair_pos = "top pair"
                elif len(board_vals) > 1 and pair_val == board_vals[1]:
                    pair_pos = "2nd pair"  
                elif len(board_vals) > 2 and pair_val == board_vals[2]:
                    pair_pos = "bottom pair"
                else:
                    # For pocket pairs below board or between board cards
                    if is_pocket_pair:
                        # Determine position based on where the pocket pair would rank
                        if len(board_vals) == 1:
                            # Only one board rank, pocket pair is below it
                            pair_pos = "2nd pair"
                        elif len(board_vals) == 2:
                            # Two board ranks
                            if pair_val > board_vals[1]:
                                pair_pos = "2nd pair"  # Between highest and second
                            else:
                                pair_pos = "bottom pair"  # Below both
                        else:
                            # Three board ranks  
                            if pair_val > board_vals[1]:
                                pair_pos = "2nd pair"  # Between highest and second
                            elif pair_val > board_vals[2]:
                                pair_pos = "bottom pair"  # Between second and third
                            else:
                                pair_pos = "bottom pair"  # Below all
                    else:
                        pair_pos = "bottom pair"  # Default for made pairs
                
                # Only add kicker if not a pocket pair
                if kicker_rank and not is_pocket_pair:
                    kicker_val = rank_values[kicker_rank]
                    if kicker_rank == 'A':
                        kicker_cat = "ace kicker"
                    elif kicker_val >= 10:
                        kicker_cat = "broadway kicker"
                    else:
                        kicker_cat = "low kicker"
                    
                    result = f"{pair_pos} with {kicker_cat}"
                else:
                    result = pair_pos
                
                if board_paired:
                    result += " on a paired board"
                
                if categories:
                    return result + " with " + " with ".join(categories)
                else:
                    return result
        
        # 8. No pair (and draws without made hands)
        if hand_type == 'High Card':
            # Count overcards
            board_vals = sorted([rank_values[r] for r in board_ranks], reverse=True)
            highest_board = board_vals[0]
            
            overcards = 0
            has_ace = False
            for rank in hero_ranks:
                rank_val = rank_values[rank]
                if rank_val > highest_board:
                    overcards += 1
                if rank == 'A':
                    has_ace = True
            
            if overcards == 2:
                result = "two overcards"
            elif overcards == 1:
                result = "one overcard"
            else:
                result = "no overcards"
            
            if has_ace:
                result += " with ace-high"
            
            if categories:
                return result + " with " + " with ".join(categories)
            else:
                return result

        # Final fallback - should not reach here but handle gracefully  
        return "high card"
    
    def parse_existing_prompt(self, prompt_text: str) -> Dict[str, str]:
        """Parse an existing prompt to extract scenario data."""
        scenario = {}
        
        lines = prompt_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Section headers
            if line.startswith('###'):
                current_section = line.replace('#', '').strip()
                continue
            
            # Extract data based on section
            if current_section == "HAND_META":
                if line.startswith('hero_pos:'):
                    scenario['hero_pos'] = line.split(':', 1)[1].strip()
                elif line.startswith('hero_hand:'):
                    scenario['hero_hand'] = line.split(':', 1)[1].strip()
                elif line.startswith('villain_profile:'):
                    scenario['villain_profile'] = line.split(':', 1)[1].strip()
            elif current_section == "HISTORY_PREFLOP":
                if line.startswith('preflop:'):
                    scenario['preflop_history'] = line.split(':', 1)[1].strip()
            elif current_section == "HISTORY 1":
                if line.startswith('flop:'):
                    # Extract flop: (Kc Qs 6s)    pot: 5.5bb
                    flop_part = line.split('pot:')[0].replace('flop:', '').strip()
                    scenario['flop'] = flop_part
                elif line.startswith('stacks:'):
                    scenario['stacks'] = line.split(':', 1)[1].strip().replace('bb', '').strip()
                elif line.startswith('actions:'):
                    scenario['actions'] = line.split(':', 1)[1].strip()
            elif current_section == "DECISION 1":
                if line.startswith('pot:'):
                    scenario['pot_size'] = line.split(':', 1)[1].strip().replace('bb', '').strip()
                elif line.startswith('legal:'):
                    scenario['legal_actions'] = line.split(':', 1)[1].strip()
        
        return scenario
    
    def collect_scenario_data(self) -> Dict[str, str]:
        """Collect all scenario data from user input."""
        print("\n" + "="*60)
        print("🃏 SYSTEMC POKER MODEL RUNNER")
        print("="*60)
        print("Let's build your poker scenario step by step...\n")
        
        scenario = {}
        
        # Hero position
        scenario['hero_pos'] = self.get_user_input(
            "Select hero's position:", 
            self.positions
        )
        
        # Hero hand
        scenario['hero_hand'] = self.get_user_input(
            "Enter hero's hand (e.g., 'Ad 8d', 'Kh Qc')"
        )
        
        # Villain profile
        scenario['villain_profile'] = self.get_user_input(
            "Select villain profile:",
            self.villain_profiles
        )
        
        # Preflop history
        print(f"\n📋 PREFLOP HISTORY")
        print("Describe the preflop action sequence.")
        print("Example: 'UTG folds, UTG+1 folds, LJ folds, HJ folds, CO folds, BTN raises 2.5bb, SB folds, BB calls 2.5bb'")
        scenario['preflop_history'] = self.get_user_input(
            "Enter preflop history"
        )
        
        # Flop cards
        print(f"\n🎲 FLOP")
        print("Enter the three flop cards.")
        print("Example: 'Kc Qs 6s', 'Ah 4h 3c'")
        scenario['flop'] = self.get_user_input(
            "Enter flop cards"
        )
        
        # Pot size
        scenario['pot_size'] = self.get_user_input(
            "Enter pot size (in bb, e.g., '22.5', '6.5')"
        )
        
        # Effective stacks
        scenario['stacks'] = self.get_user_input(
            "Enter effective stacks (in bb, e.g., '89', '97.5')"
        )
        
        # Previous actions on flop (if any)
        print(f"\n🎯 FLOP ACTION")
        print("Enter any actions that occurred before hero's decision.")
        print("Examples: 'BB bets 11.0bb (50% pot)', leave empty if hero acts first")
        scenario['actions'] = self.get_user_input(
            "Enter previous actions (or press Enter if hero acts first)",
            allow_empty=True
        )
        
        # Legal actions
        print(f"\n⚖️ LEGAL ACTIONS")
        print("Enter the legal actions available to hero.")
        print("Example: '[fold,call,raise 100%,allin]', '[check,bet 33% (1.7bb),bet 50% (2.5bb),bet 100% (5.0bb),allin]'")
        scenario['legal_actions'] = self.get_user_input(
            "Enter legal actions"
        )
        
        return scenario
    
    def format_prompt(self, scenario: Dict[str, str]) -> str:
        """Format the scenario into the SystemC prompt format with TOOL_TAGS."""
        
        # Build the prompt sections
        prompt_lines = [
            "### HAND_META",
            "game: cash",
            "seats: 8-max", 
            "stacks: 100bb",
            f"hero_pos: {scenario['hero_pos']}",
            f"hero_hand: {scenario['hero_hand']}",
            f"villain_profile: {scenario['villain_profile']}",
            "",
            "### HISTORY_PREFLOP",
            f"preflop: {scenario['preflop_history']}",
            "",
            "### HISTORY 1",
            f"flop: ({scenario['flop']})    pot: {scenario['pot_size']}bb",
            f"stacks: {scenario['stacks']}bb",
        ]
        
        # Add actions line
        if scenario['actions']:
            prompt_lines.append(f"actions: {scenario['actions']}")
        else:
            prompt_lines.append("actions:")
        
        prompt_lines.extend([
            "",
            "### DECISION 1", 
            "street: flop",
            f"pot: {scenario['pot_size']}bb",
            "to_act: HERO",
            f"legal: {scenario['legal_actions']}",
            "",
            "### TOOL_TAGS"
        ])
        
        # Generate and add TOOL_TAGS
        tool_tags = self.generate_tool_tags(scenario)
        prompt_lines.append(tool_tags)
        
        prompt_lines.extend([
            "",
            "Choose the best action from the legal options and provide a justification under 5 sentences. State the best action first by starting with \"**Best Action: \". Examples are **Best Action: Call**, **Best Action: Bet 33% (2.1bb)**, etc. Consider position (IP/OOP), villain profile for possible exploits, and the given tools in tool_tags about hand ranges, board texture, range advantage, nut advantage."
        ])
        
        prompt_content = "\n".join(prompt_lines)
        
        # Format with chat template
        formatted_prompt = f"<|user|>\n{prompt_content}\n<|assistant|>\n"
        
        return formatted_prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a response from the model - exact same approach as working evaluation code."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print("🔄 Generating response...")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate - EXACT same approach as evaluation script
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        completion = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return completion.strip()
    
    def run_interactive(self):
        """Run the interactive session."""
        # Show GPU info before starting
        self.show_gpu_info()
        
        # Note: Model should be loaded before calling this method
        
        while True:
            try:
                # Collect scenario data
                scenario = self.collect_scenario_data()
                
                # Format prompt
                prompt = self.format_prompt(scenario)
                
                # Display formatted prompt
                print("\n" + "="*60)
                print("📝 FORMATTED PROMPT WITH TOOL_TAGS:")
                print("="*60)
                print(prompt)
                print("="*60)
                
                # Ask if user wants to proceed
                proceed = input("\n🚀 Generate response? (y/n): ").strip().lower()
                if proceed != 'y':
                    print("Skipping generation...")
                    continue
                
                # Generate response
                response = self.generate_response(prompt)
                
                # Display response
                print("\n" + "="*60)
                print("🤖 SYSTEMC MODEL RESPONSE:")
                print("="*60)
                print(response)
                print("="*60)
                
                # Ask if user wants to continue
                continue_session = input("\n🔄 Generate another scenario? (y/n): ").strip().lower()
                if continue_session != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def process_prompt_file(self, prompt_file: str):
        """Process a prompt from a file (for Flask app integration)."""
        try:
            with open(prompt_file, 'r') as f:
                prompt_text = f.read().strip()
            
            print("📄 Processing prompt from file...")
            
            # Check if this already has TOOL_TAGS
            if "### TOOL_TAGS" in prompt_text:
                # Already has TOOL_TAGS, use as-is
                if not prompt_text.startswith("<|user|>"):
                    formatted_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
                else:
                    formatted_prompt = prompt_text
            else:
                # Parse and add TOOL_TAGS
                scenario = self.parse_existing_prompt(prompt_text)
                
                # If parsing failed, try to use the prompt as-is
                if not scenario:
                    print("⚠️  Could not parse prompt, using as-is...")
                    if not prompt_text.startswith("<|user|>"):
                        formatted_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
                    else:
                        formatted_prompt = prompt_text
                else:
                    # Generate TOOL_TAGS and format properly
                    formatted_prompt = self.format_prompt(scenario)
            
            # Note: Model should be loaded before calling this method
            
            # Generate response
            response = self.generate_response(formatted_prompt)
            
            print("="*50)
            print(response)
            
        except Exception as e:
            print(f"❌ Error processing prompt file: {e}")
            sys.exit(1)

    def cleanup_memory(self):
        """Clean up GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("🧹 CUDA memory cleaned")
            
            if torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                print("🧹 MPS memory cleaned")
        except Exception as e:
            print(f"⚠️  Memory cleanup failed: {e}")
    
    def recover_from_crash(self):
        """Attempt to recover from model crashes."""
        print("🔄 Attempting to recover from crash...")
        
        try:
            # Clear all GPU caches
            self.cleanup_memory()
            
            # Reset model and tokenizer
            self.model = None
            self.tokenizer = None
            
            # Reload model with same GPU settings
            self.load_model()
            
            print("✅ Recovery successful!")
            return True
            
        except Exception as recovery_error:
            print(f"❌ Recovery failed: {recovery_error}")
            return False

def main():
    """Main function."""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available!")
        print(f"Missing: {MISSING_DEPS}")
        print("\nPlease install required packages:")
        print("pip install torch transformers peft accelerate")
        return
    
    parser = argparse.ArgumentParser(description='SystemC Poker Model Runner')
    parser.add_argument('--model-path', type=str, 
                       default=None,
                       help='Path to the SystemC model directory')
    parser.add_argument('--gpu-info', action='store_true',
                       help='Show GPU information and exit')
    parser.add_argument('--prompt-file', type=str,
                       help='Process prompt from file (for Flask app)')
    
    args = parser.parse_args()
    
    # Intelligent model path detection
    if args.model_path:
        model_path = args.model_path
    else:
        # Try multiple possible paths
        possible_paths = [
            "out/stageB/systemC_bf16_r256",           # If run from project root
            "../../out/stageB/systemC_bf16_r256",     # If run from scripts/app
            "../../../out/stageB/systemC_bf16_r256",  # If run from deeper
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ SystemC model not found in any of these locations:")
            for path in possible_paths:
                print(f"  ❌ {path}")
            print("\nPlease ensure the SystemC model is trained and available.")
            print("Expected structure:")
            print("  out/stageB/systemC_bf16_r256/")
            print("    ├── adapter_config.json")
            print("    ├── adapter_model.safetensors")
            print("    └── ...")
            print("\nOr specify path with: --model-path /path/to/model")
            return
    
    # Check if the found/specified model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("\nPlease ensure the SystemC model is trained and available.")
        return
    
    # Initialize runner
    runner = SystemCRunner(model_path)
    
    if args.gpu_info:
        runner.show_gpu_info()
        return
    
    if args.prompt_file:
        runner.process_prompt_file(args.prompt_file)
        return
    
    print("🔧 Initializing SystemC model...")
    if not SYSTEMB_AVAILABLE:
        print(f"⚠️  Warning: SystemB functions not available: {SYSTEMB_MISSING}")
        print("TOOL_TAGS will use simplified analysis")
    
    # Run interactive session
    runner.run_interactive()

if __name__ == "__main__":
    main() 