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
            path = os.path.join(
                os.path.dirname(__file__), "..", "..", "solver", "preflop_chart.txt"
            )
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
    for part in [p.strip() for p in desc.split(",")]:
        if not part:
            continue
        if part.endswith("+"):
            hands.update(_expand_plus(part[:-1]))
        elif part.endswith("-"):
            hands.update(_expand_minus(part[:-1]))
        elif "-" in part and "+" not in part:
            start, end = part.split("-")
            hands.update(_expand_between(start.strip(), end.strip()))
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
        suited = start.endswith("s")
        i_start = RANK_TO_INDEX[start[1]]
        i_end = RANK_TO_INDEX[end[1]]
        step = 1 if i_start <= i_end else -1
        return [
            f"{prefix}{RANKS[i]}{'s' if suited else 'o'}"
            for i in range(i_start, i_end + step, step)
        ]


def _expand_plus(base: str) -> List[str]:
    if len(base) == 2:  # pair
        i_start = RANK_TO_INDEX[base[0]]
        return [RANKS[i] * 2 for i in range(i_start, len(RANKS))]
    prefix = base[0]
    suited = base[2] == "s"
    start = RANK_TO_INDEX[base[1]]
    high = RANK_TO_INDEX[prefix]
    return [f"{prefix}{RANKS[i]}{'s' if suited else 'o'}" for i in range(start, high)]


def _expand_minus(base: str) -> List[str]:
    if len(base) == 2:
        end_i = RANK_TO_INDEX[base[0]]
        return [RANKS[i] * 2 for i in range(0, end_i + 1)]
    prefix = base[0]
    suited = base[2] == "s"
    end = RANK_TO_INDEX[base[1]]
    return [
        f"{prefix}{RANKS[i]}{'s' if suited else 'o'}" for i in range(0, end + 1)
    ]


def canonize_hand(hand: str) -> str:
    """Convert raw cards like AhKs to range notation (AKs/AKo/KK)."""
    hand = hand.strip()
    if len(hand) == 2:
        # Pair notation (e.g. "QQ")
        return hand.upper()

    if len(hand) == 3:
        # Already canonical form like "AKs" or "JTo"
        r1, r2, suited_flag = hand[0].upper(), hand[1].upper(), hand[2].lower()
        if suited_flag not in {"s", "o"}:
            raise ValueError(f"Invalid hand: {hand}")
        ranks = sorted([r1, r2], key=lambda r: RANK_TO_INDEX[r], reverse=True)
        return f"{ranks[0]}{ranks[1]}{suited_flag}"

    if len(hand) == 4:
        # Raw card notation (e.g. "AhKs")
        r1, s1, r2, s2 = (
            hand[0].upper(),
            hand[1].lower(),
            hand[2].upper(),
            hand[3].lower(),
        )
        if r1 == r2:
            return r1 + r2
        ranks = sorted([r1, r2], key=lambda r: RANK_TO_INDEX[r], reverse=True)
        suited = s1 == s2
        return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"

    raise ValueError(f"Invalid hand: {hand}")


def parse_action_string(action: str) -> List[Tuple[str, str]]:
    """Parse a user action string into list of (position, action)."""
    actions: List[Tuple[str, str]] = []
    for part in re.split(r"[;,]", action):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\S+)\s+(\S+)", part)
        if not m:
            raise ValueError(f"Cannot parse action segment: {part}")
        pos, act = m.group(1), m.group(2)
        actions.append((pos.upper(), act.lower()))
    return actions


class PreflopLookup:
    """High level lookup interface."""

    def __init__(self, chart: Optional[PreflopChart] = None) -> None:
        self.chart = chart or PreflopChart()

    def _scenario_for_actions(
        self, actions: List[Tuple[str, str]]
    ) -> Tuple[str, str, str]:
        hero_pos, hero_act = actions[-1]
        prev_pos, prev_act = actions[-2] if len(actions) > 1 else (None, None)

        if hero_act == "raise" and prev_act is None:
            scenario = "Cash, 100bb, 8-max, RFI"
            position = _normalize_position(hero_pos, rfi=True)
            return scenario, position, hero_pos

        if prev_act == "raise" and hero_act in {"call", "3bet"}:
            villain_cat = _normalize_position(prev_pos)
            scenario = f"Cash, 100bb, 8-max, raise, {villain_cat}, {hero_act}"
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == "3bet" and hero_act in {"call", "4bet"}:
            ip = "IP" if _is_villain_ip(prev_pos, hero_pos) else "OOP"
            scenario = f"Cash, 100bb, 8-max, 3bet, {ip}, {hero_act}"
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == "4bet" and hero_act in {"call", "allin"}:
            ip = "IP" if _is_villain_ip(prev_pos, hero_pos) else "OOP"
            scenario = f"Cash, 100bb, 8-max, 4bet, {ip}, {hero_act}"
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        if prev_act == "allin" and hero_act == "call":
            ip = "IP" if _is_villain_ip(prev_pos, hero_pos) else "OOP"
            scenario = f"Cash, 100bb, 8-max, allin, {ip}, call"
            position = _normalize_position(hero_pos)
            return scenario, position, prev_pos

        raise ValueError("Unsupported action sequence")

    def get_ranges(
        self, action: str, hero_position: Optional[str] = None
    ) -> Dict[str, str]:
        """Return hero and villain ranges as text for given action string.

        Parameters
        ----------
        action:
            Comma separated action string like ``"CO raise, BTN call"``.
        hero_position:
            Which seat the hero occupies. If ``None`` the last actor in the
            action string is assumed to be the hero.
        """

        acts = parse_action_string(action)

        if hero_position is None:
            hero_position = acts[-1][0]
        hero_position = hero_position.upper()

        # locate the final action from the hero
        hero_index = None
        for i in range(len(acts) - 1, -1, -1):
            if acts[i][0] == hero_position:
                hero_index = i
                break

        if hero_index is None:
            raise ValueError("Hero position not found in action string")

        hero_scenario, hero_chart_pos, _ = self._scenario_for_actions(
            acts[: hero_index + 1]
        )
        res: Dict[str, str] = {}
        hero_range = self.chart.get_range_text(hero_scenario, hero_chart_pos)
        if hero_range:
            res["hero"] = hero_range

        if len(acts) > 1:
            if hero_index < len(acts) - 1:
                villain_index = hero_index + 1
            else:
                villain_index = hero_index - 1

            if 0 <= villain_index < len(acts):
                villain_scenario, villain_chart_pos, _ = self._scenario_for_actions(
                    acts[: villain_index + 1]
                )
                villain_range = self.chart.get_range_text(
                    villain_scenario, villain_chart_pos
                )
                if villain_range:
                    res["villain"] = villain_range

        return res

    def recommend(
        self, action: str, hero_hand: str, hero_position: Optional[str] = None
    ) -> str:
        """Return recommended action (fold/call/raise) for hero_hand."""
        acts = parse_action_string(action)

        if hero_position is None:
            hero_position = acts[-1][0]
        hero_position = hero_position.upper()

        # locate the most recent action from the hero
        hero_index = None
        for i in range(len(acts) - 1, -1, -1):
            if acts[i][0] == hero_position:
                hero_index = i
                break

        if hero_index is None:
            raise ValueError("Hero position not found in action string")

        hand = canonize_hand(hero_hand)

        # Determine the sequence prior to the hero's decision
        if hero_index == len(acts) - 1:
            pending = acts[:hero_index]
        else:
            pending = acts[:]

        villain_act = pending[-1][1] if pending else None
        villain_pos = pending[-1][0] if pending else None

        call_range: Set[str] = set()
        raise_range: Set[str] = set()

        if villain_act is None:
            # Hero is first to act (RFI)
            scenario, hero_pos, _ = self._scenario_for_actions([(hero_position, "raise")])
            raise_range = self.chart.get_range_combos(scenario, hero_pos) or set()
        elif villain_act == "raise":
            sc_call, pos_call, _ = self._scenario_for_actions(pending + [(hero_position, "call")])
            sc_raise, pos_raise, _ = self._scenario_for_actions(pending + [(hero_position, "3bet")])
            call_range = self.chart.get_range_combos(sc_call, pos_call) or set()
            raise_range = self.chart.get_range_combos(sc_raise, pos_raise) or set()
        elif villain_act == "3bet":
            ip = "IP" if _is_villain_ip(villain_pos, hero_position) else "OOP"
            base = f"Cash, 100bb, 8-max, 3bet, {ip}"
            call_range = (
                self.chart.get_range_combos(base + ", call", _normalize_position(hero_position))
                or set()
            )
            raise_range = (
                self.chart.get_range_combos(base + ", 4bet", _normalize_position(hero_position))
                or set()
            )
        elif villain_act == "4bet":
            ip = "IP" if _is_villain_ip(villain_pos, hero_position) else "OOP"
            base = f"Cash, 100bb, 8-max, 4bet, {ip}"
            call_range = (
                self.chart.get_range_combos(base + ", call", _normalize_position(hero_position))
                or set()
            )
            raise_range = (
                self.chart.get_range_combos(base + ", allin", _normalize_position(hero_position))
                or set()
            )
        elif villain_act == "allin":
            ip = "IP" if _is_villain_ip(villain_pos, hero_position) else "OOP"
            scenario = f"Cash, 100bb, 8-max, allin, {ip}, call"
            call_range = (
                self.chart.get_range_combos(scenario, _normalize_position(hero_position))
                or set()
            )

        in_call = hand in call_range
        in_raise = hand in raise_range

        if in_raise and in_call:
            return "raise or call"
        if in_raise:
            return "raise"
        if in_call:
            return "call"
        return "fold"


def compute_pot_and_effective_stack(
    action: str,
    stack_size: float = 100.0,
    raise_size: float = 2.5,
    sb: float = 0.5,
    bb: float = 1.0,
) -> Tuple[float, float]:
    """Return pot size and effective stack after the preflop actions.

    Parameters
    ----------
    action : str
        Preflop action string like ``"UTG raise, BTN call"``.
    stack_size : float, optional
        Starting stack size for each player (default 100bb).
    raise_size : float, optional
        Open raise size in big blinds (default 2.5bb).
    sb : float, optional
        Small blind amount (default 0.5bb).
    bb : float, optional
        Big blind amount (default 1bb).

    Returns
    -------
    Tuple[float, float]
        ``(pot, effective_stack)`` expressed in big blinds.
    """

    acts = parse_action_string(action)

    # Track committed chips for each seat
    committed: Dict[str, float] = {"SB": sb, "BB": bb}
    pot = sb + bb

    last_raise = bb
    last_raiser = "BB"

    def ensure_pos(pos: str) -> None:
        if pos not in committed:
            committed[pos] = sb if pos == "SB" else bb if pos == "BB" else 0.0

    for pos, act in acts:
        ensure_pos(pos)
        if act == "raise":
            amount = raise_size
        elif act == "3bet":
            amount = 11 if pos in {"SB", "BB"} else 7.5
        elif act == "4bet":
            ip = _is_villain_ip(last_raiser, pos)
            amount = 25 if ip else 22
        elif act in {"allin", "5bet"}:
            amount = stack_size
        elif act == "call":
            amount = last_raise
        else:
            continue

        addition = max(0.0, amount - committed.get(pos, 0.0))
        committed[pos] = amount
        pot += addition

        if act != "call":
            last_raise = amount
            last_raiser = pos

    # Determine players that saw the flop (last two non-fold positions)
    players: List[str] = []
    for pos, act in acts:
        if act == "fold":
            if pos in players:
                players.remove(pos)
            continue
        if pos not in players:
            players.append(pos)

    if len(players) >= 2:
        hero, villain = players[-2], players[-1]
    elif players:
        hero = players[0]
        villain = players[0]
    else:
        hero = villain = "BB"

    remaining_hero = stack_size - committed.get(hero, 0.0)
    remaining_villain = stack_size - committed.get(villain, 0.0)
    effective_stack = min(remaining_hero, remaining_villain)

    return pot, effective_stack
