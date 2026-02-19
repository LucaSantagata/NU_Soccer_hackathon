from __future__ import annotations

import re
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        # strip prefix and suffix to get core team name
        core = name
        if core.endswith(suffix):
            core = core[: -len(suffix)]
        core = re.sub(r"^\d+_", "", core)  # remove leading "1_" etc
        core_norm = _normalize_team_query(core)

        # exact core match
        if core_norm == q:
            return (100, len(core_norm))
        # contains match
        if q in core_norm:
            return (50, len(q))
        # token overlap
        q_tokens = set(q.split("_"))
        core_tokens = set(core_norm.split("_"))
        overlap = len(q_tokens.intersection(core_tokens))
        return (overlap, len(core_norm))

    best_node = max(node_files, key=lambda p: score_match(p, NODES_SUFFIX))
    best_edge = max(edge_files, key=lambda p: score_match(p, EDGES_SUFFIX))

    # Resolve “team name” from best_node core
    resolved = best_node.name.lower()
    resolved = resolved[: -len(NODES_SUFFIX)]
    resolved = re.sub(r"^\d+_", "", resolved)

    return {"nodes_csv": best_node, "edges_csv": best_edge, "team_resolved": resolved}


# =============================
# Objective model + optimization
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


def build_undirected_pass_matrix(
    edges: pd.DataFrame,
    player_ids: List[int],
    weight_col: str = "passes_per90_shared"
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


def estimate_minutes_proxy(edges: pd.DataFrame, player_ids: List[int]) -> np.ndarray:
    mins = {int(pid): 0.0 for pid in player_ids}
    for _, r in edges.iterrows():
        pid = int(r["from_id"])
        m = r.get("shared_minutes", np.nan)
        if pid in mins and not pd.isna(m):
            mins[pid] = max(mins[pid], float(m))
    return np.array([mins[int(pid)] for pid in player_ids], dtype=float)


def parse_positions_cell(x) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(p).strip().upper() for p in x if str(p).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s2 = re.sub(r"[\[\]'\"]", "", s)
        parts = re.split(r"[,\s;]+", s2)
        return [p.strip().upper() for p in parts if p.strip()]
    parts = re.split(r"[,\s;]+", s)
    return [p.strip().upper() for p in parts if p.strip()]


def formation_to_slots_433() -> List[str]:
    return ["GK",
            "LB", "CB", "CB", "RB",
            "DM", "CM", "AM",
            "LW", "ST", "RW"]


@dataclass
class ObjectiveWeights:
    w_player: float = 0.60
    w_centrality: float = 0.10
    w_cohesion: float = 0.30


@dataclass
class PlayerScoreConfig:
    std_penalty: float = 0.35
    min_minutes: float = 600.0
    reliability_floor: float = 0.20


def build_player_quality(
    nodes: pd.DataFrame,
    kpis_mean_cols: List[str],
    cfg: PlayerScoreConfig,
) -> np.ndarray:
    q = np.zeros(len(nodes), dtype=float)

    for mean_col in kpis_mean_cols:
        if mean_col not in nodes.columns:
            continue
        std_col = mean_col.replace("_mean", "_std")

        z_m = zscore(nodes[mean_col]).fillna(0.0).to_numpy(dtype=float)
        if is_negative_kpi(mean_col):
            z_m = -z_m
        q += z_m

        if std_col in nodes.columns:
            z_s = zscore(nodes[std_col]).fillna(0.0).to_numpy(dtype=float)
            q -= cfg.std_penalty * z_s

    q = (q - q.mean()) / (q.std() + 1e-9)
    return q


def build_centrality(nodes: pd.DataFrame, centrality_col: str) -> np.ndarray:
    if centrality_col not in nodes.columns:
        raise ValueError(f"centrality_col='{centrality_col}' not found in nodes columns.")
    c = nodes[centrality_col].fillna(0.0).to_numpy(dtype=float)
    c = (c - c.mean()) / (c.std() + 1e-9)
    return c


def cohesion_of_team(W_und: np.ndarray, idx: List[int]) -> float:
    if len(idx) < 2:
        return 0.0
    sub = W_und[np.ix_(idx, idx)]
    coh = (sub.sum() - np.trace(sub)) / 2.0
    pairs = len(idx) * (len(idx) - 1) / 2.0
    return float(coh / (pairs + 1e-12))


def optimize_lineup(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    formation_slots: List[str],
    positions_col: str = "positions",
    centrality_col: str = "net_pagerank",
    kpis: Optional[List[str]] = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    seed: int = 7,
    max_local_iters: int = 2000,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    # Basic required columns
    for col in ("player_id", "player"):
        if col not in nodes.columns:
            raise ValueError(f"nodes.csv must include '{col}' column.")

    player_ids = nodes["player_id"].astype(int).tolist()
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}

    # Network matrix + minutes proxy + reliability
    W_und = build_undirected_pass_matrix(edges, player_ids, weight_col="passes_per90_shared")
    minutes_proxy = estimate_minutes_proxy(edges, player_ids)

    reliability = np.sqrt(minutes_proxy / (minutes_proxy.max() + 1e-9))
    reliability = normalize01(reliability)
    reliability = score_cfg.reliability_floor + (1.0 - score_cfg.reliability_floor) * reliability

    eligible_mask = minutes_proxy >= float(score_cfg.min_minutes)
    eligible_ids = [pid for pid, ok in zip(player_ids, eligible_mask) if ok]
    if len(eligible_ids) < 11:
        eligible_ids = player_ids[:]  # auto-relax if too strict

    if kpis is None:
        kpis = select_kpis_balanced(nodes)

    q = build_player_quality(nodes, kpis, score_cfg) * reliability
    c = build_centrality(nodes, centrality_col) * reliability

    have_positions = positions_col in nodes.columns
    pos_lists = None
    if have_positions:
        pos_lists = nodes[positions_col].apply(parse_positions_cell).tolist()

    def fits_slot(pid: int, slot: str) -> bool:
        if not have_positions:
            return True
        i = id_to_idx[pid]
        return slot.upper() in set(pos_lists[i])

    def objective(selected_ids: List[int]) -> Tuple[float, float, float, float]:
        idx = [id_to_idx[pid] for pid in selected_ids]
        player_term = float(np.mean(q[idx])) if idx else 0.0
        cent_term = float(np.mean(c[idx])) if idx else 0.0
        coh_term = cohesion_of_team(W_und, idx)
        total = (weights.w_player * player_term +
                 weights.w_centrality * cent_term +
                 weights.w_cohesion * coh_term)
        return total, player_term, cent_term, coh_term

    # ---- Greedy build with slot constraints ----
    lineup: Dict[str, int] = {}
    chosen: set[int] = set()
    eligible_pool = eligible_ids[:]

    for slot in formation_slots:
        best_pid = None
        best_val = -1e18
        for pid in eligible_pool:
            if pid in chosen:
                continue
            if not fits_slot(pid, slot):
                continue
            trial = list(lineup.values()) + [pid]
            val, *_ = objective(trial)
            if val > best_val:
                best_val = val
                best_pid = pid

        if best_pid is None:
            # fallback: ignore positions for this slot if impossible
            for pid in eligible_pool:
                if pid in chosen:
                    continue
                trial = list(lineup.values()) + [pid]
                val, *_ = objective(trial)
                if val > best_val:
                    best_val = val
                    best_pid = pid

        lineup[slot] = best_pid
        chosen.add(best_pid)

    # ---- Local search by random swaps ----
    current_ids = list(lineup.values())
    current_val, *_ = objective(current_ids)
    unselected = [pid for pid in eligible_pool if pid not in chosen]

    improved = True
    iters = 0
    while improved and iters < max_local_iters:
        improved = False
        iters += 1

        slot = random.choice(formation_slots)
        out_pid = lineup[slot]
        if not unselected:
            break

        candidates = random.sample(unselected, k=min(10, len(unselected)))
        for in_pid in candidates:
            if have_positions and (not fits_slot(in_pid, slot)):
                continue

            trial_ids = [pid if pid != out_pid else in_pid for pid in current_ids]
            trial_val, *_ = objective(trial_ids)

            if trial_val > current_val + 1e-10:
                lineup[slot] = in_pid
                chosen.remove(out_pid)
                chosen.add(in_pid)

                unselected.remove(in_pid)
                unselected.append(out_pid)

                current_ids = list(lineup.values())
                current_val = trial_val
                improved = True
                break

    # Build output slot -> player name
    slot_to_player = {}
    for slot, pid in lineup.items():
        name = nodes.loc[nodes["player_id"].astype(int) == int(pid), "player"].iloc[0]
        slot_to_player[slot] = name

    total, player_term, cent_term, coh_term = objective(list(lineup.values()))

    return {
        "selected_kpis": kpis,
        "centrality_col": centrality_col,
        "weights": weights.__dict__,
        "score_cfg": score_cfg.__dict__,
        "has_positions": have_positions,
        "formation_slots": formation_slots,
        "objective": {
            "total": total,
            "player_term": player_term,
            "centrality_term": cent_term,
            "cohesion_term": coh_term,
        },
        "lineup": slot_to_player,
    }


# =============================
# High-level wrapper for main.py
# =============================
def run_for_team(
    team_query: str,
    repo_root: Path,
    formation_slots: List[str],
    centrality_col: str = "net_pagerank",
    positions_col: str = "positions",
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    seed: int = 7,
    max_local_iters: int = 2500,
    kpis: Optional[List[str]] = None,        # <-- add
) -> Dict:
    paths = find_team_csvs(repo_root=repo_root, team_query=team_query)

    nodes = pd.read_csv(paths["nodes_csv"])
    edges = pd.read_csv(paths["edges_csv"])

    result = optimize_lineup(
        nodes=nodes,
        edges=edges,
        formation_slots=formation_slots,
        positions_col=positions_col,
        centrality_col=centrality_col,
        kpis=kpis,                              # <-- use it
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

