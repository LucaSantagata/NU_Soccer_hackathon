from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

CohesionMode = Literal["arithmetic_mean", "geometric_mean", "algebraic_connectivity"]

@dataclass(frozen=True)
class CohesionConfig:
    mode: CohesionMode = "arithmetic_mean"
    epsilon: float = 1e-9
    star_penalty: float = 0.0  # >0 penalizes centralized passing (optional)

def build_undirected_pass_matrix(edges_df, player_ids: List[int], weight_col: str = "passes_per90_shared") -> np.ndarray:
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    W = np.zeros((n, n), dtype=float)

    for _, r in edges_df.iterrows():
        i = id_to_idx.get(int(r["from_id"]), None)
        j = id_to_idx.get(int(r["to_id"]), None)
        if i is None or j is None:
            continue
        w = r.get(weight_col, np.nan)
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        W[i, j] += float(w)

    return (W + W.T) / 2.0

def _arithmetic_mean_strength(sub: np.ndarray, eps: float) -> float:
    n = sub.shape[0]
    if n < 2:
        return 0.0
    coh = (sub.sum() - np.trace(sub)) / 2.0
    pairs = n * (n - 1) / 2.0
    return float(coh / (pairs + eps))

def _geometric_mean_strength(sub: np.ndarray, eps: float) -> float:
    n = sub.shape[0]
    if n < 2:
        return 0.0
    vals = sub[np.triu_indices(n, k=1)]
    vals = np.maximum(vals, 0.0) + eps
    return float(np.exp(np.mean(np.log(vals))))

def _algebraic_connectivity(sub: np.ndarray) -> float:
    n = sub.shape[0]
    if n < 2:
        return 0.0
    A = np.maximum(sub, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)
    return float(eigvals[1]) if len(eigvals) >= 2 else 0.0

def _star_centralization_penalty(sub: np.ndarray, eps: float) -> float:
    n = sub.shape[0]
    if n < 3:
        return 0.0
    strength = sub.sum(axis=1) - np.diag(sub)
    mx = float(np.max(strength))
    if mx <= eps:
        return 0.0
    return float(np.sum(mx - strength) / ((n - 1) * mx + eps))

def cohesion_value(W_und: np.ndarray, idx: List[int], cfg: CohesionConfig) -> float:
    if len(idx) < 2:
        return 0.0
    sub = W_und[np.ix_(idx, idx)]

    if cfg.mode == "arithmetic_mean":
        base = _arithmetic_mean_strength(sub, cfg.epsilon)
    elif cfg.mode == "geometric_mean":
        base = _geometric_mean_strength(sub, cfg.epsilon)
    elif cfg.mode == "algebraic_connectivity":
        base = _algebraic_connectivity(sub)
    else:
        raise ValueError(f"Unknown cohesion mode: {cfg.mode}")

    if cfg.star_penalty > 0:
        base = base - cfg.star_penalty * _star_centralization_penalty(sub, cfg.epsilon)

    return float(base)