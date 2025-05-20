#!/usr/bin/env python3
import sys
import os

# Add the src/llm_poker_solver directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "llm_poker_solver"))

# Import directly from preflop.py
from preflop import PreflopLookup, canonize_hand, expand_range

# Run examples and display outputs
def display_method(name, call_str, result):
    """Helper to show method call and result."""
    print(f"Method: {name}")
    print(f"Call: {call_str}")
    print(f"Output: {result}")
    print("-" * 70)

def display_ranges(name, call_str, result, description=""):
    """Helper to show both hero and villain ranges."""
    print(f"Method: {name}")
    if description:
        print(f"Scenario: {description}")
    print(f"Call: {call_str}")
    print(f"Hero range: {result.get('hero', 'None')}")
    print(f"Villain range: {result.get('villain', 'None')}")
    print("-" * 70)

def main():
    # Initialize the preflop lookup
    lookup = PreflopLookup()
    
    print("PREFLOP.PY METHOD OUTPUTS\n" + "=" * 50 + "\n")
    
    # Example 1: get_ranges for UTG RFI (default hero perspective)
    action = "UTG raise"
    result = lookup.get_ranges(action)
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}")', result, "UTG open-raising range (RFI)")
    
    # Example 2: get_ranges with specified hero position - Facing a raise
    action = "CO raise, BTN call"
    # Using BTN as hero (default)
    result_btn = lookup.get_ranges(action)
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}")', result_btn, "BTN calling CO's raise (BTN as hero)")
    
    # Using CO as hero
    result_co = lookup.get_ranges(action, hero_position="CO")
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}", hero_position="CO")', result_co, "CO raising with BTN calling (CO as hero)")
    
    # Example 3: get_ranges for 3bet scenario with different hero positions
    action = "CO raise, BTN 3bet"
    # Using BTN as hero (default)
    result_btn = lookup.get_ranges(action)
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}")', result_btn, "BTN 3-betting CO's raise (BTN as hero)")
    
    # Using CO as hero
    result_co = lookup.get_ranges(action, hero_position="CO")
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}", hero_position="CO")', result_co, "CO facing a 3-bet from BTN (CO as hero)")
    
    # Example 4: get_ranges for 4-bet scenario with different positions
    action = "BTN raise, SB 3bet, BTN 4bet"
    # Using BTN as hero (default)
    result_btn = lookup.get_ranges(action)
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}")', result_btn, "BTN 4-betting vs SB's 3bet (BTN as hero)")
    
    # Using SB as hero
    result_sb = lookup.get_ranges(action, hero_position="SB")
    display_ranges("PreflopLookup.get_ranges", f'get_ranges("{action}", hero_position="SB")', result_sb, "SB 3-betting BTN, then facing a 4-bet (SB as hero)")
    
    # Example 5: demonstrate the fixed expand_range for non-pocket pairs
    print("\nExample 5: Fixed expand_range for non-pocket pairs")
    print("The updated expand_range no longer includes pocket pairs when expanding suited/offsuit ranges")
    
    range_text1 = "AQs+"
    expanded1 = expand_range(range_text1)
    display_method("expand_range", f'expand_range("{range_text1}")', sorted(expanded1))
    
    range_text2 = "KTo+"
    expanded2 = expand_range(range_text2)
    display_method("expand_range", f'expand_range("{range_text2}")', sorted(expanded2))
    
    # Example 6: contrast with pocket pairs expansion
    range_text3 = "TT+"
    expanded3 = expand_range(range_text3)
    display_method("expand_range", f'expand_range("{range_text3}")', sorted(expanded3))
    
    # Example 7: canonize hand
    hand = "AhKs"
    result = canonize_hand(hand)
    display_method("canonize_hand", f'canonize_hand("{hand}")', result)
    
    # Example 8: recommend action with different hero positions
    action = "CO raise, BTN 3bet"
    hand = "AhKs"
    
    # Default hero (BTN)
    result = lookup.recommend(action, hand)
    display_method("PreflopLookup.recommend", f'recommend("{action}", "{hand}")', result)
    
    # CO as hero
    result_co = lookup.recommend(action, hand, hero_position="CO")
    display_method("PreflopLookup.recommend", f'recommend("{action}", "{hand}", hero_position="CO")', result_co)

if __name__ == "__main__":
    main() 