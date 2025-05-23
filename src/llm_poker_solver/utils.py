import os
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def get_hf_token():
    """
    Get the Hugging Face token from environment variables.
    Loads from .env file if available.
    """
    # Load environment variables from .env file
    if HAS_DOTENV:
        load_dotenv()

    # Get token from environment
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN not found. Please create a .env file with your token or "
            "set the HF_TOKEN environment variable."
        )

    return token


# Poker-specific utility functions

def parse_cards(cards: str) -> List[Tuple[str, str]]:
    """
    Parse a string of cards into a list of (rank, suit) tuples.
    
    Parameters
    ----------
    cards : str
        A string of cards, e.g., "AhKs2c"
    
    Returns
    -------
    List[Tuple[str, str]]
        A list of (rank, suit) tuples
    """
    if not cards or len(cards) % 2 != 0:
        raise ValueError(f"Invalid cards string: {cards}")
    
    result = []
    for i in range(0, len(cards), 2):
        if i+1 < len(cards):
            rank, suit = cards[i], cards[i+1]
            result.append((rank.upper(), suit.lower()))
    return result


def format_hand_for_display(hand: str) -> str:
    """
    Format a hand string for display, with proper spacing and suit symbols.
    
    Parameters
    ----------
    hand : str
        A hand string like "AhKs" or "AsAd"
    
    Returns
    -------
    str
        Formatted hand string
    """
    if not hand or len(hand) % 2 != 0:
        return hand
    
    # Unicode symbols for suits
    suit_symbols = {
        'c': '♣',
        'd': '♦',
        'h': '♥',
        's': '♠'
    }
    
    formatted = ""
    for i in range(0, len(hand), 2):
        if i+1 < len(hand):
            rank, suit = hand[i], hand[i+1].lower()
            symbol = suit_symbols.get(suit, suit)
            formatted += f"{rank}{symbol} "
    
    return formatted.strip()


def analyze_board_texture(board: str) -> Dict[str, Any]:
    """
    Analyze the texture of a board.
    
    Parameters
    ----------
    board : str
        A string representing the board, e.g., "AhKs2c"
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with board texture features
    """
    if not board or len(board) < 6 or len(board) % 2 != 0:
        return {"texture": "unknown"}
    
    # Parse the board
    cards = parse_cards(board)
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    
    # Convert ranks to numerical values
    rank_values = []
    for rank in ranks:
        if rank == 'T':
            rank_values.append(10)
        elif rank == 'J':
            rank_values.append(11)
        elif rank == 'Q':
            rank_values.append(12)
        elif rank == 'K':
            rank_values.append(13)
        elif rank == 'A':
            rank_values.append(14)
        else:
            try:
                rank_values.append(int(rank))
            except ValueError:
                rank_values.append(0)
    
    # Check for flush possibilities
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    # Check for straight possibilities
    rank_values.sort()
    
    # Calculate various texture features
    flush_draw = any(count >= 3 for count in suit_counts.values())
    paired_board = len(set(ranks)) < len(ranks)
    trips_board = any(ranks.count(rank) >= 3 for rank in set(ranks))
    two_pair_board = sum(1 for rank in set(ranks) if ranks.count(rank) >= 2) >= 2
    high_card = max(rank_values) if rank_values else 0
    
    # Straight possibilities (very simplified)
    straight_draw = False
    if len(rank_values) >= 3:
        # Check for connected cards
        for i in range(len(rank_values) - 2):
            if rank_values[i+2] - rank_values[i] <= 4:
                straight_draw = True
                break
    
    # Assign a texture category
    if trips_board:
        texture = "trips"
    elif two_pair_board:
        texture = "two pair"
    elif paired_board:
        if flush_draw:
            texture = "paired with flush draw"
        else:
            texture = "paired"
    elif flush_draw:
        if straight_draw:
            texture = "draw heavy"
        else:
            texture = "flush draw"
    elif straight_draw:
        texture = "straight draw"
    elif high_card >= 13:  # K or A high
        texture = "high card"
    else:
        texture = "dry"
    
    return {
        "texture": texture,
        "high_card": high_card,
        "paired": paired_board,
        "trips": trips_board,
        "two_pair": two_pair_board,
        "flush_draw": flush_draw,
        "straight_draw": straight_draw,
        "rainbow": len(set(suits)) == len(suits),
        "monotone": len(set(suits)) == 1
    }


def suggest_bet_sizes(board_texture: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Suggest appropriate bet sizes based on board texture.
    
    Parameters
    ----------
    board_texture : Dict[str, Any]
        Board texture analysis from analyze_board_texture
    
    Returns
    -------
    Dict[str, List[float]]
        Suggested bet sizes for IP and OOP players. All values are
        percentages of the pot.
    """
    # Recommended bet sizes are the same for both players by default and
    # expressed as percentages of the pot. These mirror the defaults used in
    # ``SolverConfig``.
    base_sizes = [33, 50, 75, 125]
    return {"ip_flop": base_sizes, "oop_flop": base_sizes}


def generate_realistic_strategy(hand_value: float, board_texture: Dict[str, Any], 
                               is_ip: bool = True) -> Dict[str, float]:
    """
    Generate a more realistic mixed strategy based on hand strength and board texture.
    
    Parameters
    ----------
    hand_value : float
        Value between 0 and 1 representing relative hand strength (1 = nuts, 0 = worst hand)
    board_texture : Dict[str, Any]
        Board texture analysis from analyze_board_texture
    is_ip : bool
        Whether the player is in position
    
    Returns
    -------
    Dict[str, float]
        Strategy dictionary with probabilities for each action
    """
    # Base strategy components
    value_bet_freq = min(1.0, hand_value * 1.3)  # Higher for strong hands
    bluff_freq = max(0.0, 0.7 - hand_value)      # Higher for weak hands
    
    # Adjustments based on board texture
    texture = board_texture.get("texture", "dry")
    
    # More value betting on dynamic boards
    if texture in ["draw heavy", "flush draw", "straight draw"]:
        value_bet_freq = min(1.0, value_bet_freq * 1.2)
        
    # More checking on static boards with medium strength
    if texture in ["dry", "high card"] and 0.3 <= hand_value <= 0.7:
        value_bet_freq *= 0.8
        
    # OOP tends to check more
    if not is_ip:
        value_bet_freq *= 0.85
        bluff_freq *= 0.85
    
    # Polarized strategy
    if hand_value > 0.7:  # Strong hands
        bet_freq = value_bet_freq
        check_freq = 1.0 - bet_freq
        return {"BET": bet_freq, "CHECK": check_freq}
    elif hand_value < 0.3:  # Weak hands
        bet_freq = bluff_freq
        check_freq = 1.0 - bet_freq
        return {"BET": bet_freq, "CHECK": check_freq}
    else:  # Medium hands
        # Medium hands mostly check but occasionally bet for protection
        bet_freq = 0.2
        check_freq = 0.8
        return {"BET": bet_freq, "CHECK": check_freq}


def calculate_hand_strength(hand: str, board: str) -> float:
    """
    Calculate the relative strength of a hand on a given board.
    
    Parameters
    ----------
    hand : str
        Hand in format like "AcKd"
    board : str
        Board in format like "Ah7d2c"
    
    Returns
    -------
    float
        Value between 0 and 1 representing hand strength
    """
    if not hand or len(hand) != 4 or not board or len(board) < 6:
        return 0.5  # Default to medium strength if input is invalid
    
    # Extract ranks from hand
    hand_ranks = [hand[0].upper(), hand[2].upper()]
    hand_suits = [hand[1].lower(), hand[3].lower()]
    
    # Extract ranks from board
    board_ranks = [board[i].upper() for i in range(0, len(board), 2)]
    board_suits = [board[i+1].lower() for i in range(0, len(board), 2)]
    
    # Convert ranks to numeric values
    rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                  "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    
    hand_values = [rank_values.get(r, 0) for r in hand_ranks]
    board_values = [rank_values.get(r, 0) for r in board_ranks]
    
    # Basic strength indicators
    strength = 0.0
    
    # High card strength
    max_hand = max(hand_values)
    relative_high = (max_hand - 2) / 12.0  # Scale from 0 to 1
    strength += 0.2 * relative_high
    
    # Pair with board
    pair_strength = 0.0
    for hr in hand_ranks:
        if hr in board_ranks:
            pair_value = rank_values.get(hr, 0)
            pair_strength = 0.3 + 0.3 * (pair_value - 2) / 12.0
            break
    
    # Pocket pair
    pocket_pair = hand_ranks[0] == hand_ranks[1]
    if pocket_pair:
        pair_value = rank_values.get(hand_ranks[0], 0)
        if pair_value > max(board_values):
            pocket_strength = 0.4 + 0.4 * (pair_value - 2) / 12.0
            pair_strength = max(pair_strength, pocket_strength)
        else:
            pocket_strength = 0.2 + 0.2 * (pair_value - 2) / 12.0
            pair_strength = max(pair_strength, pocket_strength)
    
    strength = max(strength, pair_strength)
    
    # Two pair or better (simplified)
    if pair_strength > 0 and pocket_pair:
        strength = max(strength, 0.7)  # Set of trips
    elif pair_strength > 0 and hand_ranks[0] in board_ranks and hand_ranks[1] in board_ranks:
        strength = max(strength, 0.75)  # Two pair
        
    # Check for flush possibilities
    flush_strength = 0.0
    if hand_suits[0] == hand_suits[1]:  # Suited hand
        suit = hand_suits[0]
        board_suit_count = board_suits.count(suit)
        if board_suit_count >= 3:  # Flush
            flush_strength = 0.8
        elif board_suit_count == 2:  # Flush draw
            flush_strength = 0.5
    
    strength = max(strength, flush_strength)
    
    # Check for straight possibilities (simplified)
    all_values = sorted(hand_values + board_values)
    unique_values = sorted(set(all_values))
    
    straight = False
    for i in range(len(unique_values) - 4):
        if unique_values[i+4] - unique_values[i] == 4:
            straight = True
            break
    
    if straight:
        strength = max(strength, 0.85)
    
    # Adjust for potential (gut-shot straight draws, backdoor flushes, etc)
    all_suits = hand_suits + board_suits
    max_suit_count = max(all_suits.count(s) for s in set(all_suits))
    if max_suit_count >= 4:
        strength = max(strength, 0.3)  # Flush draw potential
    
    # Adjust for position in the range
    if max_hand >= 13:  # K or A high
        strength += 0.1
    
    return min(1.0, strength)
