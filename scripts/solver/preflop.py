#!/usr/bin/env python3
"""
Preflop module - provides missing PreflopLookup and expand_range functions.

This module was missing from the project but is imported by other files.
Provides stub implementations to prevent import errors.
"""

import os
import sys
from typing import Dict, List, Set


class PreflopLookup:
    """Stub implementation of PreflopLookup for compatibility."""
    
    def __init__(self):
        """Initialize the lookup system."""
        self.chart_path = os.path.join(os.path.dirname(__file__), "preflop_chart.txt")
    
    def get_ranges(self, scenario: str) -> Dict[str, str]:
        """Get ranges for a given scenario.
        
        Parameters
        ----------
        scenario : str
            Preflop scenario string like "UTG raise, BB call"
            
        Returns
        -------
        Dict[str, str]
            Dictionary with 'hero' and 'villain' ranges
        """
        # Stub implementation - return basic ranges
        return {
            'hero': 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J2s,T9s-T2s,98s-92s,87s-82s,76s-72s,65s-62s,54s-52s,43s-42s,32s,AKo-A2o,KQo-K2o,QJo-Q2o,JTo-J2o,T9o-T2o,98o-92o,87o-82o,76o-72o,65o-62o,54o-52o,43o-42o,32o',
            'villain': 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-A5o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
        }


def expand_range(range_str: str) -> List[str]:
    """Expand a poker range string into individual hands.
    
    Parameters
    ----------
    range_str : str
        Range string like "22+,A2s+,KQo-KJo"
        
    Returns
    -------
    List[str]
        List of individual poker hands
    """
    if not range_str or range_str in ["no range", "ERROR"]:
        return []
    
    # For now, use a simple implementation
    # This is a stub - for full functionality, a complete range parser would be needed
    hands = []
    
    # Split by comma and process each part
    parts = [part.strip() for part in range_str.split(',')]
    
    for part in parts:
        if not part:
            continue
            
        # Handle specific hands (e.g., "AKs", "22")
        if len(part) == 3 and part[2] in 'so':
            hands.append(part)
        elif len(part) == 2:
            hands.append(part)
        # Handle ranges with "+" (e.g., "22+", "A2s+")
        elif '+' in part:
            base = part.replace('+', '')
            # This is a simplified expansion - full implementation would need more logic
            hands.append(base)
        # Handle ranges with "-" (e.g., "KQo-KJo")
        elif '-' in part:
            # Simplified - just add the endpoints
            start, end = part.split('-')
            hands.extend([start.strip(), end.strip()])
        else:
            hands.append(part)
    
    # Remove duplicates and return
    return list(set(hands))


# For compatibility with existing code
def get_ranges_from_preflop_chart(scenario: str) -> Dict[str, str]:
    """Compatibility function that redirects to PreflopLookup."""
    lookup = PreflopLookup()
    return lookup.get_ranges(scenario)


if __name__ == "__main__":
    # Test the functions
    print("Testing PreflopLookup...")
    lookup = PreflopLookup()
    ranges = lookup.get_ranges("UTG raise, BB call")
    print(f"Hero range: {ranges['hero'][:50]}...")
    print(f"Villain range: {ranges['villain'][:50]}...")
    
    print("\nTesting expand_range...")
    test_range = "22+,A2s+,KQo"
    expanded = expand_range(test_range)
    print(f"Expanded '{test_range}' to: {expanded[:10]}...") 