import os
import re
from typing import Dict, List, Tuple, Optional, Set

# Card rank order helper
RANKS = "23456789TJQKA"
RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}

# Postflop acting order used to determine who is in position
POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"]

# Mapping of positions for chart lookups
POSITION_CATEGORY_RFI = {
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "LJ": "LJ",
    "HJ": "HJ",
    "CO": "CO",
    "BTN": "BTN",
    "SB": "SB",
}

POSITION_CATEGORY_OTHER = {
    "UTG": "EP",
    "UTG+1": "EP",
    "LJ": "MP",
    "HJ": "MP",
    "CO": "CO",
    "BTN": "BTN",
    "SB": "SB",
    "BB": "BB",
}


def _normalize_position(pos: str, rfi: bool = False) -> str:
    """Map raw position to chart category."""
    pos = pos.upper()
    if rfi:
        return POSITION_CATEGORY_RFI.get(pos, pos)
    return POSITION_CATEGORY_OTHER.get(pos, pos)


def _is_villain_ip(villain: str, hero: str) -> bool:
    """Return True if villain is in position relative to hero."""
    return POSTFLOP_ORDER.index(villain) > POSTFLOP_ORDER.index(hero)


class PreflopChart:
    """Load and access the preflop chart."""

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "solver", "preflop_chart.txt")
        self.scenarios = self._parse_chart(path)

    @staticmethod
    def _parse_chart(path: str) -> Dict[str, Dict[str, str]]:
        scenarios: Dict[str, Dict[str, str]] = {}
        current: Optional[str] = None
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("Cash"):
                    current = line
                    scenarios[current] = {}
                    continue
                if current is None:
                    continue
                if ":" in line:
                    pos, rng = line.split(":", 1)
                    scenarios[current][pos.strip()] = rng.strip()
                else:
                    scenarios[current]["range"] = line.strip()
        return scenarios

    def get_range_text(self, scenario: str, position: str) -> Optional[str]:
        return self.scenarios.get(scenario, {}).get(position)

    def get_range_combos(self, scenario: str, position: str) -> Optional[Set[str]]:
        text = self.get_range_text(scenario, position)
        if text is None:
            return None
        return expand_range(text)


def expand_range(desc: str) -> Set[str]:
    """Expand a range description into a set of hand notations."""
    hands: Set[str] = set()
    for part in [p.strip() for p in desc.split(',')]:
        if not part:
            continue
        if '-' in part and '+' not in part:
            start, end = part.split('-')
            hands.update(_expand_between(start.strip(), end.strip()))
        elif part.endswith('+'):
            hands.update(_expand_plus(part[:-1]))
        elif part.endswith('-'):
            hands.update(_expand_minus(part[:-1]))
        else:
            hands.add(part)
    return hands


def _expand_between(start: str, end: str) -> List[str]:
    """Expand ranges like 55-JJ or A4s-A5s."""
    if start[0] == start[1]:
        # pair range
        i_start = RANK_TO_INDEX[start[0]]
        i_end = RANK_TO_INDEX[end[0]]
        step = 1 if i_start <= i_end else -1
        return [RANKS[i] * 2 for i in range(i_start, i_end + step, step)]
    else:
        prefix = start[0]
        suited = start.endswith('s')
        i_start = RANK_TO_INDEX[start[1]]
        i_end = RANK_TO_INDEX[end[1]]
        step = 1 if i_start <= i_end else -1
        return [f"{prefix}{RANKS[i]}{'s' if suited else 'o'}" for i in range(i_start, i_end + step, step)]


def _expand_plus(base: str) -> List[str]:
    if len(base) == 2:  # pair
        i_start = RANK_TO_INDEX[base[0]]
        return [RANKS[i] * 2 for i in range(i_start, len(RANKS))]
    prefix = base[0]
    suited = base[2] == 's'
    start = RANK_TO_INDEX[base[1]]
    high = RANK_TO_INDEX[prefix]
    return [f"{prefix}{RANKS[i]}{'s' if suited else 'o'}" for i in range(start, high)] + [f"{prefix}{prefix}{'s' if suited else 'o'}"]


def _expand_minus(base: str) -> List[str]:
    if len(base) == 2:
        end_i = RANK_TO_INDEX[base[0]]
        return [RANKS[i] * 2 for i in range(0, end_i + 1)]
    prefix = base[0]
    suited = base[2] == 's'
    end = RANK_TO_INDEX[base[1]]
    return [f"{prefix}{RANKS[i]}{'s' if suited else 'o'}" for i in range(2, end + 1)]


def canonize_hand(hand: str) -> str:
    """Convert raw cards like AhKs to range notation (AKs/AKo/KK)."""
    hand = hand.strip()
    if len(hand) == 2:
        return hand.upper()
    if len(hand) != 4:
        raise ValueError(f"Invalid hand: {hand}")
    r1, s1, r2, s2 = hand[0].upper(), hand[1].lower(), hand[2].upper(), hand[3].lower()
    if r1 == r2:
        return r1 + r2
    ranks = sorted([r1, r2], key=lambda r: RANK_TO_INDEX[r], reverse=True)
    suited = s1 == s2
    return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"


def parse_action_string(action: str) -> List[Tuple[str, str]]:
    """Parse a user action string into list of (position, action)."""
    actions: List[Tuple[str, str]] = []
    for part in re.split(r'[;,]', action):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'(\S+)\s+(\S+)', part)
        if not m:
            raise ValueError(f"Cannot parse action segment: {part}")
        pos, act = m.group(1), m.group(2)
        actions.append((pos.upper(), act.lower()))
    return actions


class PreflopLookup:
    """High level lookup interface."""

    def __init__(self, chart: Optional[PreflopChart] = None) -> None:
        self.chart = chart or PreflopChart()

    def _scenario_for_actions(self, actions: List[Tuple[str, str]]) -> Tuple[str, str, str]:
        hero_pos, hero_act = actions[-1]
        prev_pos, prev_act = actions[-2] if len(actions) > 1 else (None, None)

        if hero_act == 'raise' and prev_act is None:
            scenario = 'Cash, 100bb, 8-max, RFI'
            position = _normalize_position(hero_pos, rfi=True)
            return scenario, position, hero_pos

        if prev_act == 'raise' and hero_act in {'call', '3bet'}:
            villain_cat = _normalize_position(prev_pos)
            scenario = f'Cash, 100bb, 8-max, raise, {villain_cat}, {hero_act}'
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == '3bet' and hero_act in {'call', '4bet'}:
            ip = 'IP' if _is_villain_ip(prev_pos, hero_pos) else 'OOP'
            scenario = f'Cash, 100bb, 8-max, 3bet, {ip}, {hero_act}'
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == '4bet' and hero_act in {'call', 'allin'}:
            ip = 'IP' if _is_villain_ip(prev_pos, hero_pos) else 'OOP'
            scenario = f'Cash, 100bb, 8-max, 4bet, {ip}, {hero_act}'
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == 'allin' and hero_act == 'call':
            ip = 'IP' if _is_villain_ip(prev_pos, hero_pos) else 'OOP'
            scenario = f'Cash, 100bb, 8-max, allin, {ip}, call'
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        raise ValueError('Unsupported action sequence')

    def get_ranges(self, action: str) -> Dict[str, str]:
        """Return hero and villain ranges as text for given action string."""
        acts = parse_action_string(action)
        scenario, hero_pos, villain_pos = self._scenario_for_actions(acts)
        res: Dict[str, str] = {}
        hero_range = self.chart.get_range_text(scenario, hero_pos)
        if hero_range:
            res['hero'] = hero_range

        if len(acts) >= 2:
            prev_pos, prev_act = acts[-2]
            if prev_act == 'raise' and acts[-1][1] in {'call', '3bet'}:
                villain_scenario = f'Cash, 100bb, 8-max, raise, {_normalize_position(prev_pos)}, {acts[-1][1]}'
                res['villain'] = self.chart.get_range_text(villain_scenario, _normalize_position(prev_pos))
            elif prev_act == '3bet' and acts[-1][1] in {'call', '4bet'}:
                ip = 'IP' if _is_villain_ip(prev_pos, hero_pos) else 'OOP'
                villain_scenario = f'Cash, 100bb, 8-max, 3bet, {ip}, {prev_act}'
                res['villain'] = self.chart.get_range_text(villain_scenario, _normalize_position(prev_pos))
        return res

    def recommend(self, action: str, hero_hand: str) -> str:
        """Return recommended action (fold/call/raise) for hero_hand."""
        acts = parse_action_string(action)
        scenario, hero_pos, _ = self._scenario_for_actions(acts)
        hand = canonize_hand(hero_hand)

        call_range = self.chart.get_range_combos(scenario, hero_pos) or set()
        alt_action = '3bet' if scenario.endswith('call') else '4bet'
        alt_scenario = scenario.rsplit(', ', 1)[0] + f', {alt_action}'
        raise_range = self.chart.get_range_combos(alt_scenario, hero_pos) or set()

        in_call = hand in call_range
        in_raise = hand in raise_range

        if in_raise and in_call:
            return 'raise or call'
        if in_raise:
            return 'raise'
        if in_call:
            return 'call'
        return 'fold'
