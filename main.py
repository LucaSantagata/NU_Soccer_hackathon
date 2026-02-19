from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from optimization import run_for_team, formation_to_slots_433, ObjectiveWeights, PlayerScoreConfig


def main():
    parser = argparse.ArgumentParser(
        description="Optimize best XI for a given team using IMPECT open data (nodes+fully-connected pass network)."
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help='Team name query, e.g. "fc_bayern_muenchen" or "bayern".',
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Path to repo root (default: current working directory).",
    )
    parser.add_argument(
        "--formation",
        type=str,
        default="4-3-3",
        help='Formation string (currently supported: "4-3-3").',
    )
    parser.add_argument(
        "--centrality",
        type=str,
        default="net_pagerank",
        help="Centrality column name in nodes.csv (e.g., net_pagerank, net_betweenness, net_strength...).",
    )
    parser.add_argument(
        "--min_minutes",
        type=float,
        default=600.0,
        help="Minimum minutes proxy to filter low-minute players (default: 600).",
    )
    parser.add_argument(
        "--std_penalty",
        type=float,
        default=0.35,
        help="Penalty weight for KPI standard deviation (default: 0.35).",
    )
    parser.add_argument(
        "--w_player",
        type=float,
        default=0.60,
        help="Objective weight for KPI-based player quality (default: 0.60).",
    )
    parser.add_argument(
        "--w_centrality",
        type=float,
        default=0.10,
        help="Objective weight for individual centrality (default: 0.10).",
    )
    parser.add_argument(
        "--w_cohesion",
        type=float,
        default=0.30,
        help="Objective weight for within-XI cohesion (default: 0.30).",
    )
    parser.add_argument(
        "--positions_col",
        type=str,
        default="positions",
        help='Column in nodes.csv containing eligible positions (default: "positions"). '
             'If missing, optimization will run without role constraints.',
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    if args.formation.strip() != "4-3-3":
        raise ValueError('Only formation="4-3-3" is wired in this main script right now.')

    formation_slots = formation_to_slots_433()

    weights = ObjectiveWeights(
        w_player=float(args.w_player),
        w_centrality=float(args.w_centrality),
        w_cohesion=float(args.w_cohesion),
    )
    score_cfg = PlayerScoreConfig(
        std_penalty=float(args.std_penalty),
        min_minutes=float(args.min_minutes),
    )

    result = run_for_team(
        team_query=args.team,
        repo_root=repo_root,
        formation_slots=formation_slots,
        centrality_col=args.centrality,
        positions_col=args.positions_col,
        weights=weights,
        score_cfg=score_cfg,
        seed=7,
        max_local_iters=2500,
    )

    # ---- Pretty print ----
    print("\n============================")
    print(f"TEAM: {result['team_resolved_name']}")
    print("============================\n")

    print("Loaded files:")
    print(f"  nodes: {result['paths']['nodes_csv']}")
    print(f"  edges: {result['paths']['edges_csv']}")

    print("\nSelected KPIs:")
    for k in result["selected_kpis"]:
        print(" -", k)

    print("\nCentrality:", result["centrality_col"])
    print("Has positions:", result["has_positions"])
    print("Objective components:", result["objective"])

    print("\nOptimal XI (slot -> player):")
    for slot, name in result["lineup"].items():
        print(f"  {slot:>2}  {name}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
