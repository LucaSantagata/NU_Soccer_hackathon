from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

from optimization import (
    find_team_csvs,
    run_for_team,
    ObjectiveWeights,
    PlayerScoreConfig,
)

# -----------------------------
# Formation templates
# -----------------------------
FORMATION_TEMPLATES: Dict[str, List[str]] = {
    "4-3-3": ["GK", "LB", "CB", "CB", "RB", "DM", "CM", "AM", "LW", "ST", "RW"],
    "4-2-3-1": ["GK", "LB", "CB", "CB", "RB", "DM", "DM", "LW", "AM", "RW", "ST"],
    "4-4-2": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"],
    "3-4-3": ["GK", "CB", "CB", "CB", "LM", "CM", "CM", "RM", "LW", "ST", "RW"],
    "3-5-2": ["GK", "CB", "CB", "CB", "LM", "DM", "CM", "AM", "RM", "ST", "ST"],
}

def parse_custom_slots(text: str) -> List[str]:
    """
    Accepts formats like:
      GK,LB,CB,CB,RB,DM,CM,AM,LW,ST,RW
    or newlines. Returns list of 11 slot labels.
    """
    parts = re.split(r"[,\n;]+", (text or "").strip())
    slots = [p.strip().upper() for p in parts if p.strip()]
    return slots

def guess_kpi_mean_cols(df: pd.DataFrame) -> List[str]:
    # Most IMPECT KPI means end with "_mean"
    cols = [c for c in df.columns if c.endswith("_mean")]
    # Drop obvious non-KPI means if any (keep conservative)
    drop = {"minutes_mean", "time_mean"}
    cols = [c for c in cols if c.lower() not in drop]
    return sorted(cols)

def guess_centrality_cols(df: pd.DataFrame) -> List[str]:
    # Anything that looks like a network metric
    keys = ("net_", "central", "pagerank", "between", "eigen", "closen", "strength", "degree")
    cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keys):
            cols.append(c)
    # Prefer net_* first
    cols = sorted(cols, key=lambda x: (0 if x.lower().startswith("net_") else 1, x))
    return cols

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IMPECT Lineup Optimizer", layout="wide")

st.title("IMPECT Open Data – Optimal XI Lineup Optimizer")

with st.sidebar:
    st.header("Data / Team")
    repo_root = st.text_input("Repo root", value=".", help="Folder that contains the two data directories.")
    team_query = st.text_input("Team name (fuzzy)", value="fc_bayern_muenchen")

    st.divider()
    st.header("Formation")
    formation_mode = st.radio("Formation mode", ["Template", "Custom slots"], index=0)

    if formation_mode == "Template":
        formation = st.selectbox("Formation template", list(FORMATION_TEMPLATES.keys()), index=0)
        formation_slots = FORMATION_TEMPLATES[formation]
        st.caption(f"Slots: {', '.join(formation_slots)}")
    else:
        default_slots = "GK,LB,CB,CB,RB,DM,CM,AM,LW,ST,RW"
        custom_text = st.text_area("Enter 11 slots (comma/newline separated)", value=default_slots, height=120)
        formation_slots = parse_custom_slots(custom_text)

    st.divider()
    st.header("Objective weights")
    w_player = st.slider("Player quality (KPIs) weight", 0.0, 1.0, 0.60, 0.05)
    w_cent = st.slider("Centrality weight", 0.0, 1.0, 0.10, 0.05)
    w_coh = st.slider("Cohesion (within-XI passes) weight", 0.0, 1.0, 0.30, 0.05)
    st.caption("Tip: weights don’t have to sum to 1, but it’s usually sensible.")

    st.divider()
    st.header("Player filtering / risk")
    min_minutes = st.number_input("Min minutes (proxy) filter", min_value=0.0, value=600.0, step=50.0)
    std_penalty = st.slider("Std penalty (consistency penalty)", 0.0, 1.0, 0.35, 0.05)

    st.divider()
    positions_col = st.text_input("Positions column name", value="positions")
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)
    max_local_iters = st.number_input("Local search iterations", min_value=100, value=2500, step=100)

# Validate formation slots
if len(formation_slots) != 11:
    st.error(f"Formation must define exactly 11 slots. You currently have {len(formation_slots)}.")
    st.stop()

# Load team files (preview + populate dropdowns)
repo_root_path = Path(repo_root).resolve()

try:
    paths = find_team_csvs(repo_root_path, team_query)
    nodes_path = paths["nodes_csv"]
    edges_path = paths["edges_csv"]
except Exception as e:
    st.error(f"Could not find team files: {e}")
    st.stop()

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Resolved team & files")
    st.write(f"**Team:** {paths['team_resolved']}")
    st.write(f"**Nodes:** `{nodes_path}`")
    st.write(f"**Edges:** `{edges_path}`")

# Load nodes for column discovery
nodes_df = pd.read_csv(nodes_path)

# Centrality choices
centrality_cols = guess_centrality_cols(nodes_df)
if not centrality_cols:
    centrality_cols = [c for c in nodes_df.columns if c.lower().startswith("net_")] or ["net_pagerank"]

with colB:
    st.subheader("Model inputs")
    centrality_col = st.selectbox(
        "Centrality / network metric column",
        centrality_cols,
        index=0 if "net_pagerank" not in centrality_cols else centrality_cols.index("net_pagerank"),
        help="This should exist in nodes.csv for this team.",
    )

# KPI choices
kpi_mean_cols = guess_kpi_mean_cols(nodes_df)

st.subheader("KPIs")
kpi_mode = st.radio("KPI selection", ["Auto-select (balanced)", "Manual select"], index=0, horizontal=True)

selected_kpis: Optional[List[str]] = None
if kpi_mode == "Manual select":
    default_manual = [c for c in kpi_mean_cols if any(k in c.upper() for k in ["GOALS", "SHOT_XG", "PXT_PASS", "SUCCESSFUL_PASSES"])]
    selected_kpis = st.multiselect(
        "Select KPI mean columns (the optimizer uses mean + std if available)",
        options=kpi_mean_cols,
        default=default_manual[:8] if default_manual else kpi_mean_cols[:8],
        help="Columns should end with _mean. The optimizer will also use the paired _std if present.",
    )
    if not selected_kpis:
        st.warning("Manual mode selected, but no KPIs chosen. Switch back to auto or pick a few KPIs.")
        st.stop()
else:
    st.caption("Auto mode picks a small, diverse KPI set for this team (attack/progression/passing/defending/security).")

# Run
st.divider()
run_btn = st.button("Optimize XI", type="primary")

if run_btn:
    weights = ObjectiveWeights(w_player=float(w_player), w_centrality=float(w_cent), w_cohesion=float(w_coh))
    score_cfg = PlayerScoreConfig(std_penalty=float(std_penalty), min_minutes=float(min_minutes))

    with st.spinner("Running optimization..."):
        result = run_for_team(
            team_query=team_query,
            repo_root=repo_root_path,
            formation_slots=formation_slots,
            centrality_col=centrality_col,
            positions_col=positions_col,
            weights=weights,
            score_cfg=score_cfg,
            seed=int(seed),
            max_local_iters=int(max_local_iters),
            kpis=selected_kpis,  # None => auto
        )

    st.success("Optimization completed.")

    # Outputs
    st.subheader("Selected features")
    st.write("**KPIs used:**")
    st.write(result["selected_kpis"])

    st.write(f"**Centrality used:** `{result['centrality_col']}`")
    st.write(f"**Has positions column:** `{result['has_positions']}`")

    st.subheader("Objective breakdown")
    st.json(result["objective"])

    st.subheader("Optimal XI")
    lineup = result["lineup"]
    lineup_df = pd.DataFrame({"Slot": list(lineup.keys()), "Player": list(lineup.values())})
    st.dataframe(lineup_df, use_container_width=True)

    # Optional: show nodes preview
    with st.expander("Preview team nodes data (first 20 rows)"):
        st.dataframe(nodes_df.head(20), use_container_width=True)
