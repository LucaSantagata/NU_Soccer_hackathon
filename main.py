from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from optimization import (
    run_for_team,
    formation_to_slots_433,
    ObjectiveWeights,
    PlayerScoreConfig,
)


def _parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if not x:
        return None
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return parts or None


def main():
    parser = argparse.ArgumentParser(
        description="Optimize best XI for a given team using IMPECT open data (nodes + fully-connected pass network)."
    )
    parser.add_argument("--team", type=str, required=True, help='Team name query, e.g. "fc_bayern_muenchen" or "bayern".')
    parser.add_argument("--repo_root", type=str, default=".", help="Path to repo root (default: current working directory).")
    parser.add_argument("--formation", type=str, default="4-3-3", help='Formation string (currently supported: "4-3-3").')

    # KPI objective settings
    parser.add_argument("--min_minutes", type=float, default=600.0, help="Minimum minutes filter (default: 600).")
    parser.add_argument("--std_penalty", type=float, default=0.35, help="Penalty for KPI standard deviation (default: 0.35).")
    parser.add_argument("--kpis", type=str, default="", help="Optional comma-separated KPI mean columns to use. If empty, auto-select.")

    # Mobility objective settings
    parser.add_argument("--mobility_metrics", type=str, default="", help="Optional comma-separated mobility/net columns (e.g., net_pagerank,net_strength).")
    parser.add_argument("--centrality", type=str, default="", help="Backward-compatible: single net metric column to use (if mobility_metrics not set).")

    # Objective weights (backward compatible flags)
    parser.add_argument("--w_player", type=float, default=0.60, help="(Deprecated name) Weight for KPI term. Maps to w_kpi.")
    parser.add_argument("--w_centrality", type=float, default=0.10, help="(Deprecated name) Weight for mobility/net term. Maps to w_net.")
    parser.add_argument("--w_cohesion", type=float, default=0.30, help="Weight for within-XI cohesion term.")
    parser.add_argument("--positions_col", type=str, default="positions", help='Positions column name (default: "positions").')

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    if args.formation.strip() != "4-3-3":
        raise ValueError('Only formation="4-3-3" is wired in this CLI right now.')

    formation_slots = formation_to_slots_433()

    weights = ObjectiveWeights(
        w_kpi=float(args.w_player),
        w_net=float(args.w_centrality),
        w_cohesion=float(args.w_cohesion),
    )
    score_cfg = PlayerScoreConfig(
        std_penalty=float(args.std_penalty),
        min_minutes=float(args.min_minutes),
    )

    kpis = _parse_csv_list(args.kpis)
    mobility_metrics = _parse_csv_list(args.mobility_metrics)
    centrality_col = args.centrality.strip() or None

    result = run_for_team(
        team_query=args.team,
        repo_root=repo_root,
        formation_slots=formation_slots,
        positions_col=args.positions_col,
        weights=weights,
        score_cfg=score_cfg,
        seed=7,
        max_local_iters=2500,
        kpis=kpis,
        mobility_metrics=mobility_metrics,
        centrality_col=centrality_col,
    )

    print("\n============================")
    print(f"TEAM: {result['team_resolved_name']}")
    print("============================\n")

    print("Loaded files:")
    print(f"  nodes: {result['paths']['nodes_csv']}")
    print(f"  edges: {result['paths']['edges_csv']}")

    print("\nSelected KPIs:")
    for k in result["selected_kpis"]:
        print(" -", k)

    print("\nSelected mobility metrics:")
    for m in result.get("selected_mobility_metrics", []):
        print(" -", m)

    print("\nHas positions:", result["has_positions"])
    print("Objective components:", result["objective"])

    print("\nOptimal XI (slot -> player):")
    for slot, name in result["lineup"].items():
        print(f"  {slot:>2}  {name}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
