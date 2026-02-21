from __future__ import annotations

import random
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from position_utils import build_player_slot_eligibility
from scoring import (
    ObjectiveWeights,
    PlayerScoreConfig,
    apply_reliability,
    build_kpi_score_by_group,
    build_kpi_score_global,
    build_network_composite,
    normalize01,
    normalize_component_for_scalarization,
)
from cohesion import CohesionConfig, build_undirected_pass_matrix, cohesion_value


def extract_minutes(nodes: pd.DataFrame, edges: pd.DataFrame, player_ids: List[int]) -> np.ndarray:
    candidates = [c for c in nodes.columns if c.lower() in {
        "minutes", "minutes_played", "played_minutes", "time_played", "mins_played"
    }]
    if candidates:
        col = candidates[0]
        return nodes[col].fillna(0.0).to_numpy(dtype=float)

    mins = {int(pid): 0.0 for pid in player_ids}
    for _, r in edges.iterrows():
        m = r.get("shared_minutes", np.nan)
        if pd.isna(m):
            continue
        m = float(m)
        a = int(r["from_id"]); b = int(r["to_id"])
        if a in mins:
            mins[a] = max(mins[a], m)
        if b in mins:
            mins[b] = max(mins[b], m)
    return np.array([mins[int(pid)] for pid in player_ids], dtype=float)


def infer_position_group_from_eligibility(elig: Set[str]) -> str:
    if "GK" in elig:
        return "GK"
    if elig.intersection({"CB", "LB", "RB", "LWB", "RWB"}):
        return "DEF"
    if elig.intersection({"DM", "CM", "AM", "LM", "RM"}):
        return "MID"
    if elig.intersection({"LW", "RW", "ST"}):
        return "ATT"
    return "UNK"


@dataclass(frozen=True)
class SearchConfig:
    max_iters: int = 3000
    candidate_samples: int = 20
    tabu_tenure: int = 50


@dataclass(frozen=True)
class ObjectiveConfig:
    component_norm: str = "sigmoid"
    cohesion_norm: str = "edge_max"


def _normalize_cohesion(raw_coh: float, W_und: np.ndarray, method: str) -> float:
    if method == "none":
        return raw_coh
    if method == "edge_max":
        mx = float(np.max(W_und))
        if mx <= 1e-9:
            return 0.0
        return float(raw_coh / mx)
    raise ValueError(f"Unknown cohesion_norm method: {method}")


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
    obj_cfg: ObjectiveConfig = ObjectiveConfig(),
    search_cfg: SearchConfig = SearchConfig(),
    seed: int = 7,
    position_group_z: bool = True,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    for col in ("player_id", "player"):
        if col not in nodes.columns:
            raise ValueError(f"nodes.csv must include '{col}' column.")

    player_ids = nodes["player_id"].astype(int).tolist()
    id_to_idx = {pid: i for i, pid in enumerate(player_ids)}

    elig_slots = build_player_slot_eligibility(nodes, positions_col=positions_col)

    def fits_slot(pid: int, slot: str) -> bool:
        base_slot = re.sub(r"\d+$", "", slot.upper())
        return base_slot in elig_slots[id_to_idx[pid]]

    W_und = build_undirected_pass_matrix(edges, player_ids, weight_col="passes_per90_shared")

    minutes = extract_minutes(nodes, edges, player_ids)
    rel = np.sqrt(minutes / (minutes.max() + 1e-9))
    rel = normalize01(rel)
    rel = score_cfg.reliability_floor + (1.0 - score_cfg.reliability_floor) * rel

    eligible_mask = minutes >= float(score_cfg.min_minutes)
    eligible_ids = [pid for pid, ok in zip(player_ids, eligible_mask) if ok]
    if len(eligible_ids) < 11:
        eligible_ids = player_ids[:]

    if kpis is None:
        raise ValueError("kpis must be provided (use optimization_refactored.optimize_lineup which auto-selects).")

    if mobility_metrics is None:
        mobility_metrics = [c for c in nodes.columns if c.lower().startswith("net_")]

    if position_group_z:
        group_labels = [infer_position_group_from_eligibility(s) for s in elig_slots]
        kpi_vec = build_kpi_score_by_group(nodes, kpis, group_labels, score_cfg, kpi_weights=kpi_weights)
    else:
        kpi_vec = build_kpi_score_global(nodes, kpis, score_cfg, kpi_weights=kpi_weights)

    net_vec = build_network_composite(nodes, mobility_metrics, mobility_weights)

    kpi_vec = apply_reliability(kpi_vec, rel, positive_only=score_cfg.reliability_positive_only)
    net_vec = apply_reliability(net_vec, rel, positive_only=score_cfg.reliability_positive_only)

    def team_objective(selected_ids: List[int]) -> Dict[str, float]:
        idx = [id_to_idx[pid] for pid in selected_ids]

        kpi_term = float(np.mean(kpi_vec[idx])) if idx else 0.0
        net_term = float(np.mean(net_vec[idx])) if idx else 0.0
        raw_coh = cohesion_value(W_und, idx, cohesion_cfg)
        coh_term = _normalize_cohesion(raw_coh, W_und, obj_cfg.cohesion_norm)

        kpi_norm = normalize_component_for_scalarization(kpi_term, method=obj_cfg.component_norm)
        net_norm = normalize_component_for_scalarization(net_term, method=obj_cfg.component_norm)

        if obj_cfg.cohesion_norm == "none":
            coh_norm = normalize_component_for_scalarization(coh_term, method="sigmoid")
        else:
            coh_norm = float(np.clip(coh_term, 0.0, 1.0))

        total = (weights.w_kpi * kpi_norm +
                 weights.w_net * net_norm +
                 weights.w_cohesion * coh_norm)

        return {
            "total": float(total),
            "kpi_term": kpi_term,
            "net_term": net_term,
            "cohesion_raw": raw_coh,
            "cohesion_term": coh_term,
            "kpi_norm": kpi_norm,
            "net_norm": net_norm,
            "cohesion_norm": coh_norm,
        }

    lineup: Dict[str, int] = {}
    chosen: Set[int] = set()
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

    current_ids = list(lineup.values())
    current_val = team_objective(current_ids)["total"]
    unselected = [pid for pid in eligible_ids if pid not in chosen]

    tabu: Dict[Tuple[str, int, int], int] = {}

    def decay_tabu():
        dead = []
        for k in tabu:
            tabu[k] -= 1
            if tabu[k] <= 0:
                dead.append(k)
        for k in dead:
            tabu.pop(k, None)

    for _ in range(search_cfg.max_iters):
        if not unselected:
            break
        decay_tabu()

        slot = random.choice(formation_slots)
        out_pid = lineup[slot]
        candidates = random.sample(unselected, k=min(search_cfg.candidate_samples, len(unselected)))

        best_in = None
        best_val = current_val

        for in_pid in candidates:
            if not fits_slot(in_pid, slot):
                continue
            if (slot, out_pid, in_pid) in tabu:
                continue
            trial_ids = [pid if pid != out_pid else in_pid for pid in current_ids]
            val = team_objective(trial_ids)["total"]
            if val > best_val + 1e-10:
                best_val = val
                best_in = in_pid

        if best_in is None:
            continue

        in_pid = best_in
        lineup[slot] = in_pid
        chosen.remove(out_pid); chosen.add(in_pid)
        unselected.remove(in_pid); unselected.append(out_pid)

        tabu[(slot, in_pid, out_pid)] = search_cfg.tabu_tenure

        current_ids = list(lineup.values())
        current_val = best_val

    slot_to_player: Dict[str, str] = {}
    for slot, pid in lineup.items():
        name = nodes.loc[nodes["player_id"].astype(int) == int(pid), "player"].iloc[0]
        slot_to_player[slot] = name

    obj = team_objective(list(lineup.values()))

    return {
        "selected_kpis": kpis,
        "selected_mobility_metrics": mobility_metrics,
        "mobility_weights": mobility_weights,
        "kpi_weights": kpi_weights,
        "weights": weights.__dict__,
        "score_cfg": score_cfg.__dict__,
        "cohesion_cfg": cohesion_cfg.__dict__,
        "objective_cfg": obj_cfg.__dict__,
        "search_cfg": search_cfg.__dict__,
        "formation_slots": formation_slots,
        "objective": obj,
        "lineup": slot_to_player,
    }
