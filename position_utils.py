from __future__ import annotations

import re
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

CANONICAL_SLOTS: Set[str] = {
    "GK",
    "CB", "LB", "RB", "LWB", "RWB",
    "DM", "CM", "AM",
    "LW", "RW", "ST",
    "LM", "RM",
}

def parse_positions_cell(x) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(p).strip() for p in x if str(p).strip()]

    s = str(x).strip()
    if not s:
        return []

    s2 = re.sub(r"^[\[\(]\s*", "", s)
    s2 = re.sub(r"\s*[\]\)]$", "", s2)
    s2 = re.sub(r"[\"']", "", s2)

    parts = re.split(r"\s*,\s*|\s*;\s*", s2)
    return [p.strip() for p in parts if p.strip()]

def map_impect_position_to_slots(pos: str) -> Set[str]:
    p = pos.upper()

    if "GOALKEEPER" in p or p.strip() == "GK":
        return {"GK"}

    slots: Set[str] = set()

    # Defenders
    if any(k in p for k in ["CENTRAL_DEFENDER", "CENTRE_BACK", "CENTER_BACK", "CENTRAL_DEFENCE"]):
        slots.add("CB")

    if any(k in p for k in ["LEFT_BACK", "LEFT_FULLBACK"]):
        slots.add("LB")
    if any(k in p for k in ["RIGHT_BACK", "RIGHT_FULLBACK"]):
        slots.add("RB")

    if any(k in p for k in ["LEFT_WINGBACK", "LEFT_WING_BACK", "LEFT_WINGBACK_DEFENDER"]):
        slots.update({"LWB", "LB"})
    if any(k in p for k in ["RIGHT_WINGBACK", "RIGHT_WING_BACK", "RIGHT_WINGBACK_DEFENDER"]):
        slots.update({"RWB", "RB"})

    # Midfield
    if any(k in p for k in ["DEFENSIVE_MIDFIELD", "DEFENSE_MIDFIELD", "HOLDING_MIDFIELD"]):
        slots.update({"DM", "CM"})

    if "CENTRAL_MIDFIELD" in p:
        slots.update({"CM", "DM", "AM"})

    if "ATTACKING_MIDFIELD" in p:
        slots.update({"AM", "CM"})

    # Wide roles
    if "LEFT_WINGER" in p or ("WINGER" in p and "LEFT" in p):
        slots.update({"LW", "LM"})
    if "RIGHT_WINGER" in p or ("WINGER" in p and "RIGHT" in p):
        slots.update({"RW", "RM"})
    if "WINGER" in p and "LEFT" not in p and "RIGHT" not in p:
        slots.update({"LW", "RW", "LM", "RM"})

    # Forwards
    if any(k in p for k in ["CENTER_FORWARD", "CENTRE_FORWARD", "STRIKER", "FORWARD"]):
        slots.add("ST")

    return slots

def build_player_slot_eligibility(nodes: pd.DataFrame, positions_col: str = "positions") -> Tuple[List[Set[str]], bool]:
    if positions_col not in nodes.columns:
        return [set(CANONICAL_SLOTS) for _ in range(len(nodes))], False

    raw_lists = nodes[positions_col].apply(parse_positions_cell).tolist()
    elig: List[Set[str]] = []
    for plist in raw_lists:
        s: Set[str] = set()
        for p in plist:
            s |= map_impect_position_to_slots(p)
        elig.append(s)
    return elig