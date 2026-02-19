from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


# =============================
# File / team resolution helpers
# =============================

NODES_DIRNAME = "Fully_connected_team_networks_with_kpis_and_netmetrics"
EDGES_DIRNAME = "Fully_connected_team_networks"

NODES_SUFFIX = "__nodes.csv"
EDGES_SUFFIX = "__fully_connected.csv"


def _normalize_team_query(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def find_team_csvs(repo_root: Path, team_query: str) -> Dict[str, Path]:
    """
    Finds the nodes and edges CSV files for a team query by searching:
      repo_root/Fully_connected_team_networks_with_kpis_and_netmetrics/*__nodes.csv
      repo_root/Fully_connected_team_networks/*__fully_connected.csv

    Supports prefixes like "1_" in filenames.
    Returns dict: {"nodes_csv": Path, "edges_csv": Path, "team_resolved": str}
    """
    repo_root = Path(repo_root).resolve()
    nodes_dir = repo_root / NODES_DIRNAME
    edges_dir = repo_root / EDGES_DIRNAME

    if not nodes_dir.exists():
        raise FileNotFoundError(f"Missing directory: {nodes_dir}")
    if not edges_dir.exists():
        raise FileNotFoundError(f"Missing directory: {edges_dir}")

    q = _normalize_team_query(team_query)

    node_files = sorted(nodes_dir.glob(f"*{NODES_SUFFIX}"))
    edge_files = sorted(edges_dir.glob(f"*{EDGES_SUFFIX}"))

    if not node_files:
        raise FileNotFoundError(f"No nodes files found in {nodes_dir}")
    if not edge_files:
        raise FileNotFoundError(f"No edges files found in {edges_dir}")

    def score_match(path: Path, suffix: str) -> Tuple[int, int]:
        """
        Higher score is better.
        Returns (primary_score, secondary_score)
        """
        name = path.name.lower()
        core = name
        if core.endswith(suffix):
            core = core[: -len(suffix)]
        core = re.sub(r"^\d+_", "", core)  # remove leading "1_" etc
        core_norm = _normalize_team_query(core)

        if core_norm == q:
            return (100, len(core_norm))
        if q in core_norm:
            return (50, len(q))
        q_tokens = set(q.split("_"))
        core_tokens = set(core_norm.split("_"))
        overlap = len(q_tokens.intersection(core_tokens))
        return (overlap, len(core_norm))

    best_node = max(node_files, key=lambda p: score_match(p, NODES_SUFFIX))
    best_edge = max(edge_files, key=lambda p: score_match(p, EDGES_SUFFIX))

    resolved = best_node.name.lower()
    resolved = resolved[: -len(NODES_SUFFIX)]
    resolved = re.sub(r"^\d+_", "", resolved)

    return {"nodes_csv": best_node, "edges_csv": best_edge, "team_resolved": resolved}


# =============================
# Positions: parse + map IMPECT labels to canonical formation slots
# =============================

CANONICAL_SLOTS: Set[str] = {
    "GK", "CB", "LB", "RB", "LWB", "RWB",
    "DM", "CM", "AM",
    "LW", "RW", "ST",
    "LM", "RM",
}

def parse_positions_cell(x) -> List[str]:
    """
    Supports:
      - python-like lists: "[A, B, C]" (with or without quotes)
      - actual python lists
      - comma/semicolon separated strings
    Returns raw strings (not mapped yet).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(p).strip() for p in x if str(p).strip()]
    s = str(x).strip()
    if not s:
        return []
    # strip brackets/quotes, then split by comma/semicolon
    s2 = re.sub(r"[\[\]'\"]", "", s)
    parts = re.split(r"\s*,\s*|\s*;\s*", s2)
    return [p.strip() for p in parts if p.strip()]

def map_impect_position_to_slots(pos: str) -> Set[str]:
    """
    Maps IMPECT strings like:
      'ATTACKING_MIDFIELD (Centre-Right)'
    into canonical slot families.

    Notes:
    - GK is ONLY from explicit goalkeeping tokens.
    - Mapping is intentionally conservative.
    """
    p = pos.upper()
    slots: Set[str] = set()

    # GK
    if "GOALKEEPER" in p or p == "GK":
        return {"GK"}

    # Center backs
    if "CENTRE_BACK" in p or "CENTER_BACK" in p or "CENTRAL_DEFENCE" in p or "CENTRAL_DEFENSE" in p:
        slots.add("CB")

    # Fullbacks / wingbacks
    if "LEFT_BACK" in p or "LEFT_FULLBACK" in p:
        slots.add("LB")
    if "RIGHT_BACK" in p or "RIGHT_FULLBACK" in p:
        slots.add("RB")

    if "LEFT_WING_BACK" in p or "LEFT_WINGBACK" in p:
        slots.update({"LWB", "LB"})
    if "RIGHT_WING_BACK" in p or "RIGHT_WINGBACK" in p:
        slots.update({"RWB", "RB"})
    if "WING_BACK" in p and "LEFT" in p:
        slots.update({"LWB", "LB"})
    if "WING_BACK" in p and "RIGHT" in p:
        slots.update({"RWB", "RB"})

    # Midfield
    if "DEFENSIVE_MIDFIELD" in p:
        slots.update({"DM", "CM"})
    if "CENTRAL_MIDFIELD" in p:
        slots.update({"CM", "DM", "AM"})
    if "ATTACKING_MIDFIELD" in p:
        slots.update({"AM", "CM"})

    # Wings
    if "LEFT_WINGER" in p or ("WINGER" in p and "LEFT" in p):
        slots.update({"LW", "LM"})
    if "RIGHT_WINGER" in p or ("WINGER" in p and "RIGHT" in p):
        slots.update({"RW", "RM"})
    if "WINGER" in p and "LEFT" not in p and "RIGHT" not in p:
        slots.update({"LW", "RW", "LM", "RM"})

    # Forwards
    if "STRIKER" in p or "CENTRE_FORWARD" in p or "CENTER_FORWARD" in p or "FORWARD" in p:
        slots.add("ST")

    return slots

def build_player_slot_eligibility(nodes: pd.DataFrame, positions_col: str = "positions") -> Tuple[List[Set[str]], bool]:
    """
    Returns:
      - list of eligible canonical slot sets per player
      - whether positions_col existed
    """
    if positions_col not in nodes.columns:
        return [set(CANONICAL_SLOTS) for _ in range(len(nodes))], False

    raw_lists = nodes[positions_col].apply(parse_positions_cell).tolist()
    elig: List[Set[str]] = []
    for plist in raw_lists:
        s: Set[str] = set()
        for p in plist:
            s |= map_impect_position_to_slots(p)
        elig.append(s)
    return elig, True


# =============================
# KPI + Network objective components
# =============================

NEGATIVE_KW = (
    "LOSS", "UNSUCCESSFUL", "OWNGOAL", "OWNGOALS", "ERROR",
    "FOUL", "YELLOW", "RED", "CONCEDED"
)

def is_negative_kpi(colname: str) -> bool:
    up = colname.upper()
    return any(k in up for k in NEGATIVE_KW)

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd

def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


DEFAULT_KPI_CATEGORIES = {
    "attacking": [
        "GOALS_mean", "SHOT_XG_mean", "POSTSHOT_XG_mean", "EXPECTED_GOAL_ASSISTS_mean",
        "ASSISTS_mean", "PACKING_XG_mean"
    ],
    "progression": [
        "PXT_PASS_mean", "OPP_PXT_PASS_mean", "DEF_PXT_PASS_mean",
        "BYPASSED_OPPONENTS_mean", "BYPASSED_DEFENDERS_mean"
    ],
    "passing": [
        "SUCCESSFUL_PASSES_mean", "EXPECTED_PASSES_mean", "NEUTRAL_PASSES_mean",
    ],
    "defending": [
        "BALL_WIN_ADDED_TEAMMATES_mean", "BALL_WIN_NUMBER_mean", "BALL_WIN_REMOVED_OPPONENTS_mean",
        "DEF_PXT_BALL_WIN_mean",
    ],
    "security": [
        "BALL_LOSS_NUMBER_mean", "BALL_LOSS_ADDED_OPPONENTS_mean",
        "BALL_LOSS_REMOVED_TEAMMATES_mean", "DEF_PXT_BALL_LOSS_mean",
        "UNSUCCESSFUL_PASSES_mean",
    ],
}

def select_kpis_balanced(
    nodes: pd.DataFrame,
    categories: Dict[str, List[str]] = DEFAULT_KPI_CATEGORIES,
    per_category: int = 1,
    extra: int = 2,
    nan_thresh: float = 0.40,
    corr_thresh: float = 0.92,
) -> List[str]:
    candidates = []
    for cols in categories.values():
        for c in cols:
            if c in nodes.columns:
                candidates.append(c)

    zmap: Dict[str, np.ndarray] = {}
    varmap: Dict[str, float] = {}
    for c in dict.fromkeys(candidates):
        if nodes[c].isna().mean() > nan_thresh:
            continue
        z = zscore(nodes[c]).fillna(0.0).to_numpy(dtype=float)
        if is_negative_kpi(c):
            z = -z
        zmap[c] = z
        varmap[c] = float(np.var(z))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        aa = a - a.mean()
        bb = b - b.mean()
        den = (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)
        return float(np.dot(aa, bb) / den)

    selected: List[str] = []
    for cols in categories.values():
        cols = [c for c in cols if c in zmap]
        cols = sorted(cols, key=lambda c: varmap[c], reverse=True)
        taken = 0
        for c in cols:
            if taken >= per_category:
                break
            if all(abs(corr(zmap[c], zmap[s])) < corr_thresh for s in selected):
                selected.append(c)
                taken += 1

    remaining = [c for c in zmap.keys() if c not in selected]
    remaining = sorted(remaining, key=lambda c: varmap[c], reverse=True)
    for c in remaining:
        if len(selected) >= per_category * len(categories) + extra:
            break
        if all(abs(corr(zmap[c], zmap[s])) < corr_thresh for s in selected):
            selected.append(c)

    return selected


# =============================
# Network cohesion / matrix
# =============================

def build_undirected_pass_matrix(
    edges: pd.DataFrame,
    player_ids: List[int],
    weight_col: str = "passes_per90_shared",
) -> np.ndarray:
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    W = np.zeros((n, n), dtype=float)

    for _, r in edges.iterrows():
        i = id_to_idx.get(int(r["from_id"]), None)
        j = id_to_idx.get(int(r["to_id"]), None)
        if i is None or j is None:
            continue
        w = r.get(weight_col, np.nan)
        if pd.isna(w):
            continue
        W[i, j] += float(w)

    return (W + W.T) / 2.0


def cohesion_of_team(W_und: np.ndarray, idx: List[int]) -> float:
    if len(idx) < 2:
        return 0.0
    sub = W_und[np.ix_(idx, idx)]
    coh = (sub.sum() - np.trace(sub)) / 2.0
    pairs = len(idx) * (len(idx) - 1) / 2.0
    return float(coh / (pairs + 1e-12))


# =============================
# Minutes extraction (avoid GK bias)
# =============================

def extract_minutes(nodes: pd.DataFrame, edges: pd.DataFrame, player_ids: List[int]) -> np.ndarray:
    """
    Priority:
      1) nodes minutes column if present
      2) else max(shared_minutes) over BOTH from_id and to_id
    """
    minutes_cols = [c for c in nodes.columns if c.lower() in {
        "minutes", "minutes_played", "played_minutes", "time_played", "mins_played"
    }]
    if minutes_cols:
        col = minutes_cols[0]
        return nodes[col].fillna(0.0).to_numpy(dtype=float)

    mins = {int(pid): 0.0 for pid in player_ids}
    for _, r in edges.iterrows():
        m = r.get("shared_minutes", np.nan)
        if pd.isna(m):
            continue
        m = float(m)
        a = int(r["from_id"]); b = int(r["to_id"])
        if a in mins: mins[a] = max(mins[a], m)
        if b in mins: mins[b] = max(mins[b], m)
    return np.array([mins[int(pid)] for pid in player_ids], dtype=float)


# =============================
# Formation templates
# =============================

def formation_to_slots_433() -> List[str]:
    return ["GK",
            "LB", "CB", "CB", "RB",
            "DM", "CM", "AM",
            "LW", "ST", "RW"]


# =============================
# Objective definitions (STANDARD scalarization)
# =============================

@dataclass
class ObjectiveWeights:
    """
    Team objective is scalarized as:
      total = w_kpi * KPI_term + w_net * NET_term + w_cohesion * COH_term
    """
    w_kpi: float = 0.55
    w_net: float = 0.25
    w_cohesion: float = 0.20


@dataclass
class PlayerScoreConfig:
    std_penalty: float = 0.35
    min_minutes: float = 600.0
    reliability_floor: float = 0.25


def build_kpi_score(nodes: pd.DataFrame, kpis_mean_cols: List[str], cfg: PlayerScoreConfig) -> np.ndarray:
    score = np.zeros(len(nodes), dtype=float)
    for mean_col in kpis_mean_cols:
        if mean_col not in nodes.columns:
            continue
        std_col = mean_col.replace("_mean", "_std")

        zm = zscore(nodes[mean_col]).fillna(0.0).to_numpy(dtype=float)
        if is_negative_kpi(mean_col):
            zm = -zm
        score += zm

        if std_col in nodes.columns:
            zs = zscore(nodes[std_col]).fillna(0.0).to_numpy(dtype=float)
            score -= cfg.std_penalty * zs

    score = (score - score.mean()) / (score.std() + 1e-9)
    return score


def build_network_composite(
    nodes: pd.DataFrame,
    metrics: List[str],
    metric_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    NET(i) = sum_m v_m z(m_i). If weights not provided, equal weights.
    """
    if not metrics:
        return np.zeros(len(nodes), dtype=float)

    ws = metric_weights or {m: 1.0 for m in metrics}
    net = np.zeros(len(nodes), dtype=float)

    for m in metrics:
        if m not in nodes.columns:
            raise ValueError(f"Network metric '{m}' not found in nodes.csv.")
        z = zscore(nodes[m]).fillna(0.0).to_numpy(dtype=float)
        net += float(ws.get(m, 1.0)) * z

    net = (net - net.mean()) / (net.std() + 1e-9)
    return net


# =============================
# Optimizer (greedy + local swaps)
# =============================

def optimize_lineup(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    formation_slots: List[str],
    positions_col: str = "positions",
    kpis: Optional[List[str]] = None,
    mobility_metrics: Optional[List[str]] = None,
    mobility_weights: Optional[Dict[str, float]] = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    seed: int = 7,
    max_local_iters: int = 2000,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    for col in ("player_id", "player"):
        if col not in nodes.columns:
            raise ValueError(f"nodes.csv must include '{col}' column.")

    player_ids = nodes["player_id"].astype(int).tolist()
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}

    # slot eligibility
    elig_slots, has_positions = build_player_slot_eligibility(nodes, positions_col=positions_col)

    def fits_slot(pid: int, slot: str) -> bool:
        if not has_positions:
            return True
        i = id_to_idx[pid]
        return slot.upper() in elig_slots[i]

    # cohesion matrix
    W_und = build_undirected_pass_matrix(edges, player_ids, weight_col="passes_per90_shared")

    # minutes + reliability
    minutes = extract_minutes(nodes, edges, player_ids)
    reliability = np.sqrt(minutes / (minutes.max() + 1e-9))
    reliability = normalize01(reliability)
    reliability = score_cfg.reliability_floor + (1.0 - score_cfg.reliability_floor) * reliability

    # eligible by minutes
    eligible_mask = minutes >= float(score_cfg.min_minutes)
    eligible_ids = [pid for pid, ok in zip(player_ids, eligible_mask) if ok]
    if len(eligible_ids) < 11:
        eligible_ids = player_ids[:]  # relax automatically

    # KPI set
    if kpis is None:
        kpis = select_kpis_balanced(nodes)

    # mobility metrics default: first few net_* columns if present
    if mobility_metrics is None:
        candidates = [c for c in nodes.columns if c.lower().startswith("net_")]
        mobility_metrics = candidates[:3] if candidates else []

    kpi_vec = build_kpi_score(nodes, kpis, score_cfg) * reliability
    net_vec = build_network_composite(nodes, mobility_metrics, mobility_weights) * reliability

    def team_objective(selected_ids: List[int]) -> Dict[str, float]:
        idx = [id_to_idx[pid] for pid in selected_ids]
        kpi_term = float(np.mean(kpi_vec[idx])) if idx else 0.0
        net_term = float(np.mean(net_vec[idx])) if idx else 0.0
        coh_term = cohesion_of_team(W_und, idx)

        total = (weights.w_kpi * kpi_term +
                 weights.w_net * net_term +
                 weights.w_cohesion * coh_term)

        return {
            "total": total,
            "kpi_term": kpi_term,
            "net_term": net_term,
            "cohesion_term": coh_term,
        }

    # --- Greedy build with hard slot constraints ---
    lineup: Dict[str, int] = {}
    chosen: set[int] = set()

    for slot in formation_slots:
        best_pid, best_val = None, -1e18
        for pid in eligible_ids:
            if pid in chosen:
                continue
            if not fits_slot(pid, slot):
                continue
            trial = list(lineup.values()) + [pid]
            val = team_objective(trial)["total"]
            if val > best_val:
                best_val = val
                best_pid = pid

        if best_pid is None:
            raise ValueError(
                f"No eligible player found for slot '{slot}'. "
                f"Check positions mapping or the '{positions_col}' column."
            )

        lineup[slot] = best_pid
        chosen.add(best_pid)

    # --- Local search swaps that preserve eligibility ---
    current_ids = list(lineup.values())
    current_val = team_objective(current_ids)["total"]
    unselected = [pid for pid in eligible_ids if pid not in chosen]

    for _ in range(max_local_iters):
        if not unselected:
            break
        slot = random.choice(formation_slots)
        out_pid = lineup[slot]

        improved = False
        for in_pid in random.sample(unselected, k=min(15, len(unselected))):
            if not fits_slot(in_pid, slot):
                continue
            trial_ids = [pid if pid != out_pid else in_pid for pid in current_ids]
            val = team_objective(trial_ids)["total"]
            if val > current_val + 1e-10:
                lineup[slot] = in_pid
                chosen.remove(out_pid); chosen.add(in_pid)
                unselected.remove(in_pid); unselected.append(out_pid)
                current_ids = list(lineup.values())
                current_val = val
                improved = True
                break
        if not improved:
            continue

    # output
    slot_to_player: Dict[str, str] = {}
    for slot, pid in lineup.items():
        name = nodes.loc[nodes["player_id"].astype(int) == int(pid), "player"].iloc[0]
        slot_to_player[slot] = name

    obj = team_objective(list(lineup.values()))

    return {
        "selected_kpis": kpis,
        "selected_mobility_metrics": mobility_metrics,
        "mobility_weights": mobility_weights,
        "weights": weights.__dict__,
        "score_cfg": score_cfg.__dict__,
        "has_positions": has_positions,
        "formation_slots": formation_slots,
        "objective": obj,
        "lineup": slot_to_player,
    }


# =============================
# High-level wrapper for CLI/UI
# =============================

def run_for_team(
    team_query: str,
    repo_root: Path,
    formation_slots: List[str],
    positions_col: str = "positions",
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    seed: int = 7,
    max_local_iters: int = 2500,
    kpis: Optional[List[str]] = None,
    mobility_metrics: Optional[List[str]] = None,
    mobility_weights: Optional[Dict[str, float]] = None,
    # Backward-compat: if older code passes --centrality, treat it as a single mobility metric
    centrality_col: Optional[str] = None,
) -> Dict:
    paths = find_team_csvs(repo_root=repo_root, team_query=team_query)

    nodes = pd.read_csv(paths["nodes_csv"])
    edges = pd.read_csv(paths["edges_csv"])

    if mobility_metrics is None and centrality_col:
        mobility_metrics = [centrality_col]

    result = optimize_lineup(
        nodes=nodes,
        edges=edges,
        formation_slots=formation_slots,
        positions_col=positions_col,
        kpis=kpis,
        mobility_metrics=mobility_metrics,
        mobility_weights=mobility_weights,
        weights=weights,
        score_cfg=score_cfg,
        seed=seed,
        max_local_iters=max_local_iters,
    )

    result["team_resolved_name"] = paths["team_resolved"]
    result["paths"] = {
        "nodes_csv": str(paths["nodes_csv"]),
        "edges_csv": str(paths["edges_csv"]),
    }
    return result
