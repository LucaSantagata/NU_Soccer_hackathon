from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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

@dataclass(frozen=True)
class PlayerScoreConfig:
    std_penalty: float = 0.35
    min_minutes: float = 600.0
    reliability_floor: float = 0.25
    reliability_positive_only: bool = True

@dataclass(frozen=True)
class ObjectiveWeights:
    w_kpi: float = 0.55
    w_net: float = 0.25
    w_cohesion: float = 0.20

def apply_reliability(score: np.ndarray, reliability: np.ndarray, positive_only: bool = True) -> np.ndarray:
    if not positive_only:
        return score * reliability
    return np.where(score > 0, score * reliability, score)

def build_kpi_score_global(
    nodes: pd.DataFrame,
    kpis_mean_cols: List[str],
    cfg: PlayerScoreConfig,
    kpi_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    ws = kpi_weights or {k: 1.0 for k in kpis_mean_cols}
    score = np.zeros(len(nodes), dtype=float)

    for mean_col in kpis_mean_cols:
        if mean_col not in nodes.columns:
            continue
        std_col = mean_col.replace("_mean", "_std")

        zm = zscore(nodes[mean_col]).fillna(0.0).to_numpy(dtype=float)
        if is_negative_kpi(mean_col):
            zm = -zm
        score += float(ws.get(mean_col, 1.0)) * zm

        if std_col in nodes.columns:
            zs = zscore(nodes[std_col]).fillna(0.0).to_numpy(dtype=float)
            score -= float(ws.get(mean_col, 1.0)) * cfg.std_penalty * zs

    score = (score - score.mean()) / (score.std() + 1e-9)
    return score

def build_kpi_score_by_group(
    nodes: pd.DataFrame,
    kpis_mean_cols: List[str],
    group_labels: List[str],
    cfg: PlayerScoreConfig,
    kpi_weights: Optional[Dict[str, float]] = None,
    min_group_size: int = 4,
) -> np.ndarray:
    ws = kpi_weights or {k: 1.0 for k in kpis_mean_cols}
    n = len(nodes)
    score = np.zeros(n, dtype=float)

    groups = pd.Series(group_labels, index=nodes.index).fillna("UNK")
    unique_groups = groups.unique().tolist()

    global_z_mean = {k: zscore(nodes[k]).fillna(0.0).to_numpy(dtype=float) for k in kpis_mean_cols if k in nodes.columns}
    global_z_std = {}
    for k in kpis_mean_cols:
        s = k.replace("_mean", "_std")
        if s in nodes.columns:
            global_z_std[s] = zscore(nodes[s]).fillna(0.0).to_numpy(dtype=float)

    for mean_col in kpis_mean_cols:
        if mean_col not in nodes.columns:
            continue
        std_col = mean_col.replace("_mean", "_std")
        w = float(ws.get(mean_col, 1.0))

        z_m = np.zeros(n, dtype=float)
        z_s = np.zeros(n, dtype=float) if std_col in nodes.columns else None

        for g in unique_groups:
            mask = (groups == g).to_numpy()
            if mask.sum() < min_group_size:
                z_m[mask] = global_z_mean[mean_col][mask]
                if z_s is not None:
                    z_s[mask] = global_z_std.get(std_col, np.zeros(n))[mask]
                continue

            gm = zscore(nodes.loc[mask, mean_col]).fillna(0.0).to_numpy(dtype=float)
            z_m[mask] = gm
            if z_s is not None:
                gs = zscore(nodes.loc[mask, std_col]).fillna(0.0).to_numpy(dtype=float)
                z_s[mask] = gs

        if is_negative_kpi(mean_col):
            z_m = -z_m  

        score += w * z_m
        if z_s is not None:
            score -= w * cfg.std_penalty * z_s

    score = (score - score.mean()) / (score.std() + 1e-9)
    return score

def build_network_composite(
    nodes: pd.DataFrame,
    metrics: List[str],
    metric_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
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

def normalize_component_for_scalarization(x: float, method: str = "sigmoid") -> float:
    if method == "sigmoid":
        x = max(min(x, 40.0), -40.0)
        return float(1.0 / (1.0 + np.exp(-x)))
    if method == "clip01":
        return float(max(0.0, min(1.0, x)))
    raise ValueError(f"Unknown normalization method: {method}")