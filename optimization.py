from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scoring import ObjectiveWeights, PlayerScoreConfig
from cohesion import CohesionConfig
from optimizer_engine import optimize_lineup as _optimize_lineup, ObjectiveConfig, SearchConfig

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
        name = path.name.lower()
        core = name
        if core.endswith(suffix):
            core = core[: -len(suffix)]
        core = re.sub(r"^\d+_", "", core)
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
# KPI selection
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

# =============================
# Default feature sets (used when user does NOT specify)
# =============================

_FIXED_KPIS_PRIORITY: List[str] = [
    # Attacking / chance creation
    "GOALS_mean",
    "SHOT_XG_mean",
    "POSTSHOT_XG_mean",
    "EXPECTED_GOAL_ASSISTS_mean",
    "ASSISTS_mean",

    # Ball progression / value
    "PXT_PASS_mean",
    "BYPASSED_OPPONENTS_mean",
    "BYPASSED_DEFENDERS_mean",

    # Passing volume/quality
    "SUCCESSFUL_PASSES_mean",
    "EXPECTED_PASSES_mean",

    # Defensive contribution / ball-winning
    "BALL_WIN_NUMBER_mean",
    "BALL_WIN_ADDED_TEAMMATES_mean",

    # Ball security (lower is better; sign handled)
    "BALL_LOSS_NUMBER_mean",
    "UNSUCCESSFUL_PASSES_mean",
]

_GK_KPI_KEYWORDS = [
    "SAVE", "SAVES", "GOALKEEP", "GK_", "POSTSHOT", "PSXG", "XG_PREVENT", "GOALS_CONCEDED",
    "CONCEDED", "CLEAN_SHEET", "CLEAN_SHEETS"
]

def default_kpis(nodes: pd.DataFrame, max_kpis: int = 12) -> List[str]:
    '''Default KPI policy:
    - use a fixed set of high-signal KPIs (interpretability + stability)
    - add GK-specific KPI means if they exist (helps GK realism)
    - cap to max_kpis
    '''
    available = set(nodes.columns)
    selected: List[str] = []

    for k in _FIXED_KPIS_PRIORITY:
        if k in available and k.endswith("_mean"):
            selected.append(k)
        if len(selected) >= max_kpis:
            return selected

    # Augment with GK-related mean KPIs if present
    for c in nodes.columns:
        if not c.endswith("_mean"):
            continue
        cu = c.upper()
        if any(kw in cu for kw in _GK_KPI_KEYWORDS):
            if c not in selected:
                selected.append(c)
        if len(selected) >= max_kpis:
            break

    return selected

def default_network_metrics(nodes: pd.DataFrame) -> List[str]:
    '''Default network policy:
    - use ALL network metric columns in nodes.csv.
    - convention: columns starting with 'net_' are network metrics.
    '''
    return [c for c in nodes.columns if c.lower().startswith("net_")]


def select_kpis_balanced(
    nodes: pd.DataFrame,
    categories: Dict[str, List[str]] = DEFAULT_KPI_CATEGORIES,
    per_category: int = 1,
    extra: int = 2,
    nan_thresh: float = 0.40,
    corr_thresh: float = 0.80,
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
# Formation templates
# =============================

FORMATION_TEMPLATES: Dict[str, List[str]] = {
    "4-3-3": ["GK",
              "LB", "CB1", "CB2", "RB",
              "DM", "CM", "AM",
              "LW", "ST", "RW"],

    "4-2-3-1": ["GK",
                "LB", "CB1", "CB2", "RB",
                "DM1", "DM2",
                "LW", "AM", "RW",
                "ST"],

    "4-4-2": ["GK",
              "LB", "CB1", "CB2", "RB",
              "LM", "CM1", "CM2", "RM",
              "ST1", "ST2"],

    "3-4-3": ["GK",
              "CB1", "CB2", "CB3",
              "LWB", "CM1", "CM2", "RWB",
              "LW", "ST", "RW"],

    "3-5-2": ["GK",
              "CB1", "CB2", "CB3",
              "LWB", "DM", "CM1", "CM2", "RWB",
              "ST1", "ST2"],

    "5-3-2": ["GK",
              "LWB", "CB1", "CB2", "CB3", "RWB",
              "CM1", "CM2", "CM3",
              "ST1", "ST2"],

    "4-1-4-1": ["GK",
                "LB", "CB1", "CB2", "RB",
                "DM",
                "LM", "CM1", "CM2", "RM",
                "ST"],
}

def formation_to_slots(formation: str) -> List[str]:
    f = formation.strip()
    if f not in FORMATION_TEMPLATES:
        raise ValueError(
            f"Unknown formation '{formation}'. Available: {sorted(FORMATION_TEMPLATES.keys())}"
        )
    return FORMATION_TEMPLATES[f]

# Keep this for backward compatibility (your existing calls still work)
def formation_to_slots_433() -> List[str]:
    return formation_to_slots("4-3-3")


# =============================
# Public API
# =============================

def optimize_lineup(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    formation_slots: List[str],
    positions_col: str = "positions",
    kpis: Optional[List[str]] = None,
    kpi_weights: Optional[Dict[str, float]] = None,
    mobility_metrics: Optional[List[str]] = None,
    mobility_weights: Optional[Dict[str, float]] = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    cohesion_cfg: CohesionConfig = CohesionConfig(),
    objective_cfg: ObjectiveConfig = ObjectiveConfig(),
    search_cfg: SearchConfig = SearchConfig(),
    seed: int = 7,
    position_group_z: bool = True,
) -> Dict:
    if kpis is None:
        kpis = default_kpis(nodes)
        if not kpis:
            kpis = select_kpis_balanced(nodes)

    if mobility_metrics is None:
        mobility_metrics = default_network_metrics(nodes)

    if len(formation_slots) != 11:
        raise ValueError(f"Formation must have 11 slots, got {len(formation_slots)}: {formation_slots}")

    return _optimize_lineup(
        nodes=nodes,
        edges=edges,
        formation_slots=formation_slots,
        positions_col=positions_col,
        kpis=kpis,
        kpi_weights=kpi_weights,
        mobility_metrics=mobility_metrics,
        mobility_weights=mobility_weights,
        weights=weights,
        score_cfg=score_cfg,
        cohesion_cfg=cohesion_cfg,
        obj_cfg=objective_cfg,
        search_cfg=search_cfg,
        seed=seed,
        position_group_z=position_group_z,
    )


def run_for_team(
    team_query: str,
    repo_root: Path,
    formation_slots: List[str],
    positions_col: str = "positions",
    weights: ObjectiveWeights = ObjectiveWeights(),
    score_cfg: PlayerScoreConfig = PlayerScoreConfig(),
    cohesion_cfg: CohesionConfig = CohesionConfig(),
    objective_cfg: ObjectiveConfig = ObjectiveConfig(),
    search_cfg: SearchConfig = SearchConfig(),
    seed: int = 7,
    kpis: Optional[List[str]] = None,
    kpi_weights: Optional[Dict[str, float]] = None,
    mobility_metrics: Optional[List[str]] = None,
    mobility_weights: Optional[Dict[str, float]] = None,
    centrality_col: Optional[str] = None,
    position_group_z: bool = True,
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
        kpi_weights=kpi_weights,
        mobility_metrics=mobility_metrics,
        mobility_weights=mobility_weights,
        weights=weights,
        score_cfg=score_cfg,
        cohesion_cfg=cohesion_cfg,
        objective_cfg=objective_cfg,
        search_cfg=search_cfg,
        seed=seed,
        position_group_z=position_group_z,
    )

    result["team_resolved_name"] = paths["team_resolved"]
    result["paths"] = {"nodes_csv": str(paths["nodes_csv"]), "edges_csv": str(paths["edges_csv"])}
    return result
