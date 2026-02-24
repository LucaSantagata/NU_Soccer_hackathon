from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import re
from pathlib import Path
from typing import List

from optimization import (
    find_team_csvs,
    run_for_team,
    FORMATION_TEMPLATES,
    formation_to_slots,
    ObjectiveWeights,
    PlayerScoreConfig,
    SearchConfig,
)

# -----------------------------
# Helpers
# -----------------------------
def parse_custom_slots(text: str) -> List[str]:
    parts = re.split(r"[,;]+", (text or "").strip())
    return [p.strip().upper() for p in parts if p.strip()]

def guess_kpi_mean_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_mean")]
    drop = {"minutes_mean", "time_mean"}
    return sorted([c for c in cols if c.lower() not in drop])

def default_net_cols(df: pd.DataFrame) -> List[str]:
    # Default policy: ALL net_* columns (matches CLI defaults)
    net = [c for c in df.columns if c.lower().startswith("net_")]
    return sorted(net)


def _clean_team_slug(raw: str) -> str:
    """
    Convert filename core -> clean team slug.
    Examples:
      '1._fc_koeln'          -> 'fc_koeln'
      '1_fc_union_berlin'    -> 'fc_union_berlin'
      'bayer_04_leverkusen'  -> 'bayer_04_leverkusen'
    """
    s = raw.strip().lower()

    # remove leading index patterns like:
    #   "1_" , "1._", "01_", "2-_" (etc)
    s = re.sub(r"^\s*\d+\s*[\._-]?\s*", "", s)

    # normalize separators
    s = s.replace(" ", "_")
    s = re.sub(r"[\.\-]+", "_", s)    # dots/dashes -> underscore
    s = re.sub(r"_+", "_", s)         # collapse multiple underscores
    s = s.strip("_")

    return s

def list_teams(repo_root_path: Path) -> List[str]:
    """
    Reads team slugs from nodes directory filenames:
      Fully_connected_team_networks_with_kpis_and_netmetrics/*__nodes.csv
    Returns cleaned slugs, sorted.
    """
    nodes_dir = repo_root_path / "Fully_connected_team_networks_with_kpis_and_netmetrics"
    if not nodes_dir.exists():
        return []

    teams = []
    for p in nodes_dir.glob("*__nodes.csv"):
        core = p.name.replace("__nodes.csv", "")
        teams.append(_clean_team_slug(core))

    return sorted(set(teams))
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Big O(Goal) F.C.", layout="wide")
st.title("Big O(Goal) F.C. — Optimal XI Lineup Optimizer")

# Top controls (main page)
# Top controls (main page)
topA, topB, topC = st.columns([2, 1, 1])

with topB:
    repo_root = st.text_input("Repo root", value=".")

repo_root_path_preview = Path(repo_root).resolve()
team_options = list_teams(repo_root_path_preview)

with topA:
    if team_options:
        default_team = "fc_bayern_muenchen" if "fc_bayern_muenchen" in team_options else team_options[0]
        team_query = st.selectbox("Team", options=team_options, index=team_options.index(default_team))
    else:
        team_query = st.text_input("Team name (fuzzy match to filenames)", value="fc_bayern_muenchen")
        st.caption("Could not find team list in the nodes directory; using free-text team input.")

with topC:
    formation = st.selectbox(
        "Formation",
        options=list(FORMATION_TEMPLATES.keys()),
        index=(list(FORMATION_TEMPLATES.keys()).index("4-3-3") if "4-3-3" in FORMATION_TEMPLATES else 0),
    )

repo_root_path = repo_root_path_preview

# Resolve files
try:
    paths = find_team_csvs(repo_root_path, team_query)
    nodes_path = paths["nodes_csv"]
    edges_path = paths["edges_csv"]
except Exception as e:
    st.error(f"Could not find team files: {e}")
    st.stop()

nodes_df = pd.read_csv(nodes_path)

# Formation slots
formation_mode = st.radio("Formation mode", ["Template", "Custom slots"], index=0, horizontal=True)
if formation_mode == "Template":
    formation_slots = formation_to_slots(formation)
    st.caption(f"Slots: {', '.join(formation_slots)}")
else:
    default_slots = "GK,LB,CB1,CB2,RB,DM,CM,AM,LW,ST,RW"
    custom_text = st.text_area("Enter 11 slots (comma/newline separated)", value=default_slots, height=100)
    formation_slots = parse_custom_slots(custom_text)

if len(formation_slots) != 11:
    st.error(f"Formation must define exactly 11 slots. You currently have {len(formation_slots)}.")
    st.stop()

# # Display resolved files
# with st.expander("Resolved team & files", expanded=True):
#     st.write(f"**Team:** {paths['team_resolved']}")
#     st.write(f"**Nodes:** `{nodes_path}`")
#     st.write(f"**Edges:** `{edges_path}`")

st.divider()

# Sidebar: model knobs
with st.sidebar:
    st.header("Objective weights (scalarization)")
    w_kpi = st.slider("KPI term weight (player performance)", 0.0, 1.0, 0.60, 0.05)
    w_net = st.slider("Network term weight (mobility metrics)", 0.0, 1.0, 0.10, 0.05)
    w_coh = st.slider("Cohesion term weight (within-XI passes)", 0.0, 1.0, 0.30, 0.05)

    st.divider()
    st.header("Player filtering / consistency")
    min_minutes = st.number_input("Min minutes filter (players who have played at least this many minutes)", min_value=0.0, value=200.0, step=50.0)
    std_penalty = st.slider("Std penalty (inconsistency penalty) This term penalizes players with high variance in their KPIs across matches)", 0.0, 1.0, 0.35, 0.05)

    st.divider()
    st.header("Search parameters (Tabu Search)")
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)
    max_iters = st.number_input("Search iterations", min_value=100, value=2500, step=100)
    candidate_samples = st.number_input("Candidates per move", min_value=5, value=20, step=1)
    tabu_tenure = st.number_input("Tabu tenure", min_value=1, value=50, step=1)

# Feature selection panel
col1, col2 = st.columns([1, 1])

kpi_mean_cols = guess_kpi_mean_cols(nodes_df)
net_cols = default_net_cols(nodes_df)

with col1:
    st.subheader("KPIs")
    use_default_kpis = st.checkbox(
        "Use default KPIs (recommended; matches CLI defaults)",
        value=True,
        help="If checked, the optimizer uses a fixed curated KPI set (and adds GK KPIs if present).",
    )
    selected_kpis: Optional[List[str]] = None
    if not use_default_kpis:
        default_manual = [c for c in kpi_mean_cols if any(k in c.upper() for k in ["GOALS", "SHOT_XG", "PXT_PASS", "SUCCESSFUL_PASSES"])]
        selected_kpis = st.multiselect(
            "Select KPI mean columns",
            options=kpi_mean_cols,
            default=(default_manual[:10] if default_manual else kpi_mean_cols[:10]),
        )
        if not selected_kpis:
            st.warning("Manual KPI mode selected, but no KPIs chosen.")
            st.stop()
    else:
        st.caption("Default KPI policy: fixed curated KPIs (+ GK KPIs if present).")

with col2:
    st.subheader("Mobility / network metrics")
    use_default_net = st.checkbox(
        "Use ALL network metrics (recommended; matches CLI defaults)",
        value=True,
        help="If checked, the optimizer uses all net_* columns in nodes.csv.",
    )
    selected_net: Optional[List[str]] = None
    if not use_default_net:
        if not net_cols:
            st.warning("No net_* columns found in nodes.csv for this team.")
        selected_net = st.multiselect(
            "Select network metric columns (nodes.csv)",
            options=net_cols,
            default=(net_cols if net_cols else []),
            help="These are combined into a composite NET term using z-scores.",
        )
        if not selected_net:
            st.warning("Manual network mode selected, but no network metrics chosen.")
            st.stop()
    else:
        st.caption(f"Default network policy: ALL net_* columns ({len(net_cols)} found).")

st.divider()
run_btn = st.button("Optimize XI", type="primary")

if run_btn:
    weights = ObjectiveWeights(w_kpi=float(w_kpi), w_net=float(w_net), w_cohesion=float(w_coh))
    score_cfg = PlayerScoreConfig(std_penalty=float(std_penalty), min_minutes=float(min_minutes))
    search_cfg = SearchConfig(max_iters=int(max_iters), candidate_samples=int(candidate_samples), tabu_tenure=int(tabu_tenure))

    with st.spinner("Running optimization..."):
        result = run_for_team(
            team_query=team_query,
            repo_root=repo_root_path,
            formation_slots=formation_slots,
            weights=weights,
            score_cfg=score_cfg,
            seed=int(seed),
            search_cfg=search_cfg,
            kpis=(None if use_default_kpis else selected_kpis),
            mobility_metrics=(None if use_default_net else selected_net),
        )

    st.success("Optimization completed.")

    st.subheader("Selected features")
    st.write("**KPIs used:**", result["selected_kpis"])
    st.write("**Mobility metrics used:**", result.get("selected_mobility_metrics", []))

    st.subheader("Objective breakdown")
    st.json(result["objective"])

    st.subheader("Optimal XI")
    lineup = result["lineup"]
    lineup_df = pd.DataFrame({"Slot": list(lineup.keys()), "Player": list(lineup.values())})
    st.dataframe(lineup_df, use_container_width=True)

    with st.expander("Preview nodes data"):
        st.dataframe(nodes_df, use_container_width=True)
