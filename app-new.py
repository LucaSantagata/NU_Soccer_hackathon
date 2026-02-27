"""
Big O(Goal) F.C. — Optimal XI Lineup Optimizer
Streamlit UI — Industry-grade, Bundesliga-themed
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from optimization import (
    find_team_csvs,
    run_for_team,
    FORMATION_TEMPLATES,
    formation_to_slots,
    ObjectiveWeights,
    PlayerScoreConfig,
    default_kpis,
    default_network_metrics,
)
from optimizer_engine import SearchConfig, ObjectiveConfig
from cohesion import CohesionConfig
from scoring import PlayerScoreConfig

# ─────────────────────────────────────────────
#  Bundesliga team name display dictionary
# ─────────────────────────────────────────────
TEAM_DISPLAY_NAMES: Dict[str, str] = {
    # Bayern & Munich area
    "fc_bayern_muenchen":           "Bayern München",
    "fc_bayern_münchen":            "Bayern München",
    "bayern_muenchen":              "Bayern München",
    # Dortmund
    "borussia_dortmund":            "Borussia Dortmund",
    "bvb_09":                       "Borussia Dortmund",
    # Leverkusen
    "bayer_04_leverkusen":          "Bayer 04 Leverkusen",
    "bayer_leverkusen":             "Bayer 04 Leverkusen",
    # Leipzig
    "rb_leipzig":                   "RB Leipzig",
    "rasenballsport_leipzig":       "RB Leipzig",
    # Frankfurt
    "eintracht_frankfurt":          "Eintracht Frankfurt",
    # Wolfsburg
    "vfl_wolfsburg":                "VfL Wolfsburg",
    # Freiburg
    "sport_club_freiburg":          "SC Freiburg",
    "sc_freiburg":                  "SC Freiburg",
    # Gladbach
    "borussia_moenchengladbach":    "Borussia M'gladbach",
    "borussia_mönchengladbach":     "Borussia M'gladbach",
    "vfl_borussia_moenchengladbach": "Borussia M'gladbach",
    # Union Berlin
    "fc_union_berlin":              "1. FC Union Berlin",
    "union_berlin":                 "1. FC Union Berlin",
    "1_fc_union_berlin":            "1. FC Union Berlin",
    # Cologne
    "fc_koeln":                     "1. FC Köln",
    "fc_köln":                      "1. FC Köln",
    "1_fc_koeln":                   "1. FC Köln",
    # Hoffenheim
    "tsg_hoffenheim":               "TSG Hoffenheim",
    "tsg_1899_hoffenheim":          "TSG Hoffenheim",
    # Stuttgart
    "vfb_stuttgart":                "VfB Stuttgart",
    # Mainz
    "fsv_mainz_05":                 "1. FSV Mainz 05",
    "mainz_05":                     "1. FSV Mainz 05",
    "1_fsv_mainz_05":               "1. FSV Mainz 05",
    # Augsburg
    "fc_augsburg":                  "FC Augsburg",
    # Bochum
    "vfl_bochum":                   "VfL Bochum",
    "vfl_bochum_1848":              "VfL Bochum",
    # Hertha
    "hertha_bsc":                   "Hertha BSC",
    "hertha_berlin":                "Hertha BSC",
    # Schalke
    "fc_schalke_04":                "FC Schalke 04",
    "schalke_04":                   "FC Schalke 04",
    # Bremen
    "sv_werder_bremen":             "Werder Bremen",
    "werder_bremen":                "Werder Bremen",
    # Heidenheim
    "fc_heidenheim":                "1. FC Heidenheim",
    "1_fc_heidenheim":              "1. FC Heidenheim",
    # Darmstadt
    "sv_darmstadt_98":              "SV Darmstadt 98",
    "darmstadt_98":                 "SV Darmstadt 98",
    # Gladbach alt
    "mgladbach":                    "Borussia M'gladbach",
}

def slug_to_display(slug: str) -> str:
    """Convert a team slug to a human-readable display name."""
    if slug in TEAM_DISPLAY_NAMES:
        return TEAM_DISPLAY_NAMES[slug]
    # Fallback: title-case the slug, remove underscores
    pretty = slug.replace("_", " ").title()
    # Fix common abbreviations
    pretty = re.sub(r"\bFc\b", "FC", pretty)
    pretty = re.sub(r"\bFsv\b", "FSV", pretty)
    pretty = re.sub(r"\bVfl\b", "VfL", pretty)
    pretty = re.sub(r"\bVfb\b", "VfB", pretty)
    pretty = re.sub(r"\bRb\b", "RB", pretty)
    pretty = re.sub(r"\bTsg\b", "TSG", pretty)
    pretty = re.sub(r"\bBsc\b", "BSC", pretty)
    pretty = re.sub(r"\bSv\b", "SV", pretty)
    pretty = re.sub(r"\bSc\b", "SC", pretty)
    pretty = re.sub(r"\bBvb\b", "BVB", pretty)
    return pretty

# ─────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Big O(Goal) F.C. | Optimal XI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Global CSS — Bundesliga-inspired dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Design tokens ── */
:root {
    --bundesliga-red:   #D0021B;
    --bundesliga-dark:  #1a1a2e;
    --pitch-green:      #1A7A3A;
    --pitch-light:      #21943F;
    --gold:             #C8960C;
    --silver:           #6B7280;
    --surface-0:        #F0F2F6;
    --surface-1:        #FFFFFF;
    --surface-2:        #E8EBF0;
    --surface-3:        #D1D5DB;
    --border:           #CBD5E1;
    --text-primary:     #111827;
    --text-secondary:   #374151;
    --text-muted:       #9CA3AF;
    --accent-glow:      rgba(208, 2, 27, 0.2);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--surface-0);
    color: var(--text-primary);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .section-header {
    color: #F1F5F9 !important;
    border-bottom-color: #2d2d4e !important;
}
[data-testid="stSidebar"] .info-box {
    background: rgba(255,255,255,0.07);
    color: #CBD5E1;
    border-left-color: var(--gold);
}
[data-testid="stSidebar"] .info-box strong { color: #F1F5F9; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stToggle label {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.1) !important;
    border-color: #3d3d60 !important;
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .hdivider {
    border-top-color: #2d2d4e !important;
}

/* ── Headings ── */
h1 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.04em; }
h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.03em; }

/* ── Card / panel ── */
.card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--bundesliga-red);
}
.card-gold {
    border-left: 3px solid var(--gold);
}
.card-green {
    border-left: 3px solid var(--pitch-green);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #2d0a12 60%, #1a1a2e 100%);
    border: 1px solid #3d1a22;
    border-top: 4px solid var(--bundesliga-red);
    border-radius: 12px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "⚽";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.06;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.06em;
    line-height: 1;
    color: #FFFFFF;
    margin: 0;
}
.hero-title span { color: var(--bundesliga-red); }
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #CBD5E1;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: var(--bundesliga-red);
    color: white;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin-bottom: 0.75rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 0.08em;
    color: #111827;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header .icon { color: var(--bundesliga-red); }

/* ── Metric chip ── */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
    margin: 0.15rem;
}
.metric-chip:hover {
    background: var(--surface-3);
    color: var(--text-primary);
}

/* ── Objective pill ── */
.obj-pill {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.obj-pill-icon { font-size: 1.4rem; }
.obj-pill-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.05em;
    color: var(--text-primary);
}
.obj-pill-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.4;
}
.obj-pill-weight {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--bundesliga-red);
    min-width: 3rem;
    text-align: right;
}

/* ── Formation pitch ── */
.pitch-container {
    background: linear-gradient(180deg, var(--pitch-green) 0%, var(--pitch-light) 50%, var(--pitch-green) 100%);
    border-radius: 10px;
    padding: 1rem;
    position: relative;
    border: 2px solid rgba(255,255,255,0.15);
}
.pitch-lines {
    border: 2px solid rgba(255,255,255,0.35);
    border-radius: 6px;
    padding: 0.75rem;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.pitch-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin: 0.3rem 0;
}
.player-token {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: default;
}
.player-token .shirt {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--bundesliga-red);
    border: 2px solid white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.65rem;
    color: white;
    letter-spacing: 0.04em;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    transition: transform 0.2s;
}
.player-token .shirt:hover { transform: scale(1.12); }
.player-token .pname {
    font-size: 0.6rem;
    color: white;
    font-weight: 600;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    margin-top: 0.2rem;
    max-width: 64px;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.player-token.gk .shirt { background: #F5C518; color: #000; border-color: #000; }

/* ── Result table ── */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.result-table th {
    background: var(--surface-2);
    color: var(--text-secondary);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
}
.result-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--surface-2);
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
.result-table tr:hover td { background: var(--surface-2); }
.slot-badge {
    background: var(--bundesliga-red);
    color: white;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}
.slot-badge.gk { background: var(--gold); color: #000; }
.slot-badge.def { background: #1A7A3A; }
.slot-badge.mid { background: #1a4a8a; }
.slot-badge.att { background: var(--bundesliga-red); }

/* ── Objective breakdown ── */
.obj-breakdown {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.obj-card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.obj-card .label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
    font-weight: 600;
}
.obj-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #C8960C;
}
.obj-card .value.total { color: var(--bundesliga-red); font-size: 1.4rem; }

/* ── Divider ── */
.hdivider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
}

/* ── Tooltip / info box ── */
.info-box {
    background: #EEF2FF;
    border-left: 3px solid var(--gold);
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0.5rem 0;
    line-height: 1.5;
}
.info-box strong { color: var(--text-primary); }
.info-box code {
    background: #E0E7FF;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    color: #1e40af;
    font-size: 0.8rem;
}

/* ── Weight bar ── */
.weight-bar-wrap {
    background: var(--surface-3);
    border-radius: 4px;
    height: 6px;
    margin-top: 0.2rem;
}
.weight-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--bundesliga-red), var(--gold));
}

/* ── Streamlit widget overrides (light main area) ── */
.stSlider > div > div > div > div {
    background: var(--bundesliga-red) !important;
}
.stButton > button {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.1em !important;
    font-size: 1.1rem !important;
    height: 3rem !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: var(--bundesliga-red) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 24px var(--accent-glow) !important;
}
div[data-testid="stExpander"] {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.stRadio > div { gap: 0.5rem; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface-2);
    border-bottom: 2px solid var(--border);
    gap: 0.25rem;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.06em;
    font-size: 0.95rem;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--bundesliga-red) !important;
    border-bottom: 2px solid var(--bundesliga-red) !important;
    font-weight: 700;
}

/* ── Feature tag ── */
.feature-tag {
    display: inline-block;
    background: rgba(232,0,29,0.15);
    color: var(--bundesliga-red);
    border: 1px solid rgba(232,0,29,0.3);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.1rem 0.5rem;
    border-radius: 3px;
    margin-left: 0.4rem;
}
.feature-tag.green {
    background: rgba(26,122,58,0.2);
    color: #4CAF82;
    border-color: rgba(26,122,58,0.4);
}

/* ── Sweep table ── */
.sweep-row-best { background: rgba(245,197,24,0.08) !important; }

/* ── Sidebar nav ── */
.nav-item {
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.25rem;
    cursor: pointer;
    font-size: 0.9rem;
    color: var(--text-secondary);
    border-left: 2px solid transparent;
    transition: all 0.15s;
}
.nav-item:hover { color: var(--text-primary); background: var(--surface-2); }
.nav-item.active {
    color: var(--text-primary);
    border-left-color: var(--bundesliga-red);
    background: var(--surface-2);
}

/* ── Status badge ── */
.status-ok {
    display: inline-flex; align-items: center; gap: 0.3rem;
    color: #166534; font-size: 0.83rem; font-weight: 500;
    background: #dcfce7; padding: 0.25rem 0.6rem; border-radius: 4px;
}
.status-err {
    display: inline-flex; align-items: center; gap: 0.3rem;
    color: #991b1b; font-size: 0.83rem; font-weight: 500;
    background: #fee2e2; padding: 0.25rem 0.6rem; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def parse_custom_slots(text: str) -> List[str]:
    parts = re.split(r"[,;\n]+", (text or "").strip())
    return [p.strip().upper() for p in parts if p.strip()]


def _clean_team_slug(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"^\s*\d+\s*[\._-]?\s*", "", s)
    s = s.replace(" ", "_")
    s = re.sub(r"[\.\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def list_teams(repo_root_path: Path) -> List[str]:
    nodes_dir = repo_root_path / "Fully_connected_team_networks_with_kpis_and_netmetrics"
    if not nodes_dir.exists():
        return []
    teams = []
    for p in nodes_dir.glob("*__nodes.csv"):
        core = p.name.replace("__nodes.csv", "")
        teams.append(_clean_team_slug(core))
    return sorted(set(teams))


def guess_kpi_mean_cols(df: pd.DataFrame) -> List[str]:
    drop = {"minutes_mean", "time_mean"}
    return sorted([c for c in df.columns if c.endswith("_mean") and c.lower() not in drop])

def default_net_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.lower().startswith("net_")])


def slot_group(slot: str) -> str:
    base = re.sub(r"\d+$", "", slot.upper())
    if base == "GK":
        return "gk"
    if base in {"CB", "LB", "RB", "LWB", "RWB"}:
        return "def"
    if base in {"DM", "CM", "AM", "LM", "RM"}:
        return "mid"
    return "att"


def format_slot_badge(slot: str) -> str:
    g = slot_group(slot)
    return f'<span class="slot-badge {g}">{slot}</span>'


def render_pitch(lineup: Dict[str, str], formation_slots: List[str]) -> str:
    """Render an HTML pitch visualization grouped by position line."""
    gk_slots = [s for s in formation_slots if slot_group(s) == "gk"]
    def_slots = [s for s in formation_slots if slot_group(s) == "def"]
    mid_slots = [s for s in formation_slots if slot_group(s) == "mid"]
    att_slots = [s for s in formation_slots if slot_group(s) == "att"]

    def row_html(slots, label="") -> str:
        tokens = ""
        for slot in slots:
            name = lineup.get(slot, "?")
            short = name.split()[-1] if name != "?" else "?"
            short = short[:9]
            slot_disp = re.sub(r"\d+$", "", slot)
            is_gk = slot_group(slot) == "gk"
            extra = "gk" if is_gk else ""
            tokens += f"""
            <div class="player-token {extra}">
                <div class="shirt">{slot_disp}</div>
                <div class="pname">{short}</div>
            </div>"""
        return f'<div class="pitch-row">{tokens}</div>'

    html = '<div class="pitch-container"><div class="pitch-lines">'
    for row_slots in [att_slots, mid_slots, def_slots, gk_slots]:
        if row_slots:
            html += row_html(row_slots)
    html += '</div></div>'
    return html


def render_objective_breakdown(obj: Dict) -> str:
    total = obj.get("total", 0)
    kpi = obj.get("kpi_norm", obj.get("kpi_term", 0))
    net = obj.get("net_norm", obj.get("net_term", 0))
    coh = obj.get("cohesion_norm", obj.get("cohesion_term", 0))
    raw_coh = obj.get("cohesion_raw", 0)

    html = f"""
    <div class="obj-breakdown">
        <div class="obj-card">
            <div class="label">🧠 KPI Score</div>
            <div class="value">{kpi:.4f}</div>
        </div>
        <div class="obj-card">
            <div class="label">🕸️ Network Score</div>
            <div class="value">{net:.4f}</div>
        </div>
        <div class="obj-card">
            <div class="label">🔗 Cohesion Score</div>
            <div class="value">{coh:.4f}</div>
        </div>
        <div class="obj-card">
            <div class="label">📊 Raw Cohesion</div>
            <div class="value">{raw_coh:.2f}</div>
        </div>
        <div class="obj-card" style="grid-column: span 2;">
            <div class="label">⭐ Total Objective</div>
            <div class="value total">{total:.6f}</div>
        </div>
    </div>"""
    return html


# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "sweep_results" not in st.session_state:
    st.session_state.sweep_results = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Optimizer"

# ─────────────────────────────────────────────
#  Hero banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Bundesliga Analytics · Network Optimization · Tabu Search</div>
    <div class="hero-title">Big O<span>(Goal)</span> F.C.</div>
    <div class="hero-subtitle">
        Optimal XI Lineup Optimizer · Player KPI scoring + passing-network cohesion + mobility metrics
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar — full configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Repo & Team ──────────────────────────
    st.markdown('<div class="section-header"><span class="icon">📁</span>DATA SOURCE</div>', unsafe_allow_html=True)
    repo_root = st.text_input(
        "Repo root path",
        value=".",
        help="Path to the repository containing 'Fully_connected_team_networks_with_kpis_and_netmetrics' and 'Fully_connected_team_networks' directories."
    )
    repo_root_path = Path(repo_root).resolve()
    team_options = list_teams(repo_root_path)

    if team_options:
        default_team = "fc_bayern_muenchen" if "fc_bayern_muenchen" in team_options else team_options[0]
        # Build display-name list while keeping slug as value
        team_display_map = {slug: slug_to_display(slug) for slug in team_options}
        team_display_options = [team_display_map[s] for s in team_options]
        default_display = team_display_map[default_team]
        selected_display = st.selectbox(
            "Team",
            options=team_display_options,
            index=team_display_options.index(default_display),
            help="Select a Bundesliga team. Matched to nodes/edges CSVs by slug."
        )
        # Reverse-map back to slug for the backend
        display_to_slug = {v: k for k, v in team_display_map.items()}
        team_query = display_to_slug[selected_display]
    else:
        team_query = st.text_input(
            "Team name (fuzzy match)",
            value="fc_bayern_muenchen",
            help="Enter team slug. The optimizer uses fuzzy filename matching."
        )
        st.caption("⚠️ Team directory not found — using free-text input.")

    # ── Formation ───────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1rem;"><span class="icon">🏟️</span>FORMATION</div>', unsafe_allow_html=True)
    formation = st.selectbox(
        "Formation template",
        options=list(FORMATION_TEMPLATES.keys()),
        index=(list(FORMATION_TEMPLATES.keys()).index("4-3-3") if "4-3-3" in FORMATION_TEMPLATES else 0),
        help="Pre-defined formation slot templates. Use 'Custom' to specify your own."
    )
    formation_mode = st.radio("Slot mode", ["Template", "Custom slots"], index=0, horizontal=True)
    if formation_mode == "Custom slots":
        custom_text = st.text_area(
            "Custom slots (comma/newline separated)",
            value="GK,LB,CB1,CB2,RB,DM,CM,AM,LW,ST,RW",
            height=80
        )
        formation_slots = parse_custom_slots(custom_text)
    else:
        formation_slots = formation_to_slots(formation)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Note about weights location ───────────
    st.markdown("""
    <div class="info-box" style="background:rgba(255,255,255,0.07); color:#CBD5E1; border-left-color:#C8960C;">
        ⚖️ <strong style="color:#F1F5F9;">Objective weights (λ)</strong> are configured in the 
        <strong style="color:#F1F5F9;">⚽ Optimizer</strong> tab — they always sum to 1.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Player filtering ──────────────────────
    st.markdown('<div class="section-header"><span class="icon">🔍</span>PLAYER FILTERING</div>', unsafe_allow_html=True)
    min_minutes = st.number_input(
        "Min minutes played",
        min_value=0.0, value=200.0, step=50.0,
        help="Players with fewer total minutes than this threshold are excluded from selection. Ensures squad depth quality."
    )
    std_penalty = st.slider(
        "Inconsistency penalty (σ)",
        0.0, 1.0, 0.35, 0.05,
        help="Penalizes players with high within-season KPI variance. 0 = ignore variance, 1 = heavily penalize streaky players. Formula: score -= w · σ_penalty · z(std)."
    )

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Scoring internals ─────────────────────
    st.markdown('<div class="section-header"><span class="icon">📐</span>SCORING INTERNALS</div>', unsafe_allow_html=True)
    position_group_z = st.toggle(
        "Position-group z-scoring",
        value=True,
        help="If ON: z-scores KPIs within positional groups (GK/DEF/MID/ATT) — prevents GKs and strikers from being ranked on the same scale. Recommended."
    )
    reliability_floor = st.slider(
        "Reliability floor",
        0.0, 1.0, 0.25, 0.05,
        help="Minimum reliability multiplier. A player with 0 minutes gets this score multiplier. Prevents completely discarding low-minute players."
    )
    component_norm = st.selectbox(
        "Component normalization",
        options=["sigmoid", "clip01"],
        index=0,
        help="How each objective component is normalized before scalarization. 'sigmoid' maps ℝ → (0,1) smoothly; 'clip01' hard-clamps."
    )
    cohesion_norm = st.selectbox(
        "Cohesion normalization",
        options=["edge_max", "none"],
        index=0,
        help="'edge_max' divides raw cohesion by the maximum edge weight (making it relative). 'none' uses raw pass counts."
    )

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Search (Tabu) ─────────────────────────
    st.markdown('<div class="section-header"><span class="icon">🔬</span>TABU SEARCH PARAMETERS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        The optimizer uses <strong>Tabu Search</strong>: starts with a greedy XI, then iteratively swaps one player per slot, 
        maintaining a <em>tabu list</em> of recent moves to escape local optima.
    </div>
    """, unsafe_allow_html=True)
    seed = st.number_input("Random seed", min_value=0, value=7, step=1,
        help="Seed for reproducibility. Same seed + same data = same result every time.")
    max_iters = st.number_input("Max iterations", min_value=100, value=2500, step=100,
        help="Number of swap attempts. More = better quality, slower. 2500 is a solid default for Bundesliga squad sizes.")
    candidate_samples = st.number_input("Candidates per move", min_value=5, value=20, step=5,
        help="How many random replacement players are evaluated per slot per iteration. Higher = more thorough search per step.")
    tabu_tenure = st.number_input("Tabu tenure", min_value=1, value=50, step=5,
        help="How many iterations a swap is 'forbidden' after being made. Prevents cycling. Tune up if optimizer revisits same lineups.")

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:var(--text-muted); text-align:center; line-height:1.6;">
        Built on IMPECT Open Data · Bundesliga<br/>
        Tabu Search · Network Analytics · xG
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Validate formation
# ─────────────────────────────────────────────
if len(formation_slots) != 11:
    st.error(f"⚠️ Formation must define exactly 11 slots. You currently have **{len(formation_slots)}**: `{', '.join(formation_slots)}`")
    st.stop()

# ─────────────────────────────────────────────
#  Resolve team files (non-blocking)
# ─────────────────────────────────────────────
try:
    paths = find_team_csvs(repo_root_path, team_query)
    nodes_path = paths["nodes_csv"]
    edges_path = paths["edges_csv"]
    nodes_df = pd.read_csv(nodes_path)
    kpi_mean_cols = guess_kpi_mean_cols(nodes_df)
    net_cols = default_net_cols(nodes_df)
    files_ok = True
    file_error = None
except Exception as e:
    files_ok = False
    file_error = str(e)
    nodes_df = pd.DataFrame()
    kpi_mean_cols = []
    net_cols = []

# ─────────────────────────────────────────────
#  Main tab bar
# ─────────────────────────────────────────────
tab_optimizer, tab_features, tab_sweep, tab_about = st.tabs([
    "⚽  OPTIMIZER",
    "🧬  FEATURES",
    "🔁  LAMBDA SWEEP",
    "📖  ABOUT",
])


# ══════════════════════════════════════════════
#  TAB 1 — OPTIMIZER
# ══════════════════════════════════════════════
with tab_optimizer:

    # File status row
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if files_ok:
            st.markdown(f'<div class="status-ok">✅ &nbsp;Nodes CSV loaded — <code>{nodes_path.name}</code></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-err">❌ &nbsp;{file_error}</div>', unsafe_allow_html=True)
    with c2:
        if files_ok:
            st.markdown(f'<div class="status-ok">✅ &nbsp;Edges CSV — <code>{edges_path.name}</code></div>', unsafe_allow_html=True)
    with c3:
        if files_ok:
            st.markdown(f'<div class="status-ok">👥 &nbsp;{len(nodes_df)} players</div>', unsafe_allow_html=True)

    if not files_ok:
        st.error("Cannot proceed: team files not found. Check your **Repo root** path and **Team** name in the sidebar.")
        st.stop()

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Objective weights (in Optimizer tab, enforced sum=1) ──────────────────
    st.markdown('<div class="section-header"><span class="icon">⚖️</span>OBJECTIVE WEIGHTS — must sum to 1</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Weighted scalarization:</strong> &nbsp;
        <code style="background:#E8EBF0; padding:0.1rem 0.4rem; border-radius:3px; color:#111827;">
            f(XI) = λ<sub>kpi</sub> · σ(KPI) + λ<sub>net</sub> · σ(NET) + λ<sub>coh</sub> · COH
        </code><br/>
        Adjust any two sliders — the third auto-corrects to keep the sum at 1.0.
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        w_kpi_raw = st.slider(
            "🧠 λ_kpi  Player KPIs",
            0.0, 1.0,
            st.session_state.get("w_kpi", 0.60),
            0.05,
            key="w_kpi_slider",
            help="Individual player performance: goals, xG, progressive passes, ball wins, consistency. Higher = reward individual brilliance."
        )
    with wc2:
        w_net_raw = st.slider(
            "🕸️ λ_net  Network Mobility",
            0.0, 1.0,
            st.session_state.get("w_net", 0.10),
            0.05,
            key="w_net_slider",
            help="Graph-theoretic importance in the pass network: PageRank, betweenness, strength. Higher = reward connector/distributor players."
        )
    with wc3:
        w_coh_raw = st.slider(
            "🔗 λ_coh  Team Cohesion",
            0.0, 1.0,
            st.session_state.get("w_coh", 0.30),
            0.05,
            key="w_coh_slider",
            help="Within-XI passes-per-90 on shared edges. Measures how connected the 11 players are to each other. Higher = reward cohesive units."
        )

    # Normalize to sum=1
    _raw_sum = w_kpi_raw + w_net_raw + w_coh_raw
    if _raw_sum < 1e-9:
        w_kpi, w_net, w_coh = 1/3, 1/3, 1/3
    else:
        w_kpi = w_kpi_raw / _raw_sum
        w_net = w_net_raw / _raw_sum
        w_coh = w_coh_raw / _raw_sum
    st.session_state["w_kpi"] = w_kpi_raw
    st.session_state["w_net"] = w_net_raw
    st.session_state["w_coh"] = w_coh_raw

    # Visual bar showing normalized split
    bar_kpi = int(w_kpi * 300)
    bar_net = int(w_net * 300)
    bar_coh = 300 - bar_kpi - bar_net
    st.markdown(f"""
    <div style="margin: 0.5rem 0 0.25rem;">
        <div style="display:flex; height:10px; border-radius:6px; overflow:hidden; gap:2px;">
            <div style="width:{bar_kpi}px; background:#D0021B;" title="KPI {w_kpi:.0%}"></div>
            <div style="width:{bar_net}px; background:#1A7A3A;" title="NET {w_net:.0%}"></div>
            <div style="width:{bar_coh}px; background:#C8960C;" title="COH {w_coh:.0%}"></div>
        </div>
        <div style="display:flex; gap:1.5rem; margin-top:0.35rem; font-size:0.78rem; color:var(--text-secondary);">
            <span><span style="color:#D0021B;font-weight:700;">■</span> KPI {w_kpi:.0%}</span>
            <span><span style="color:#1A7A3A;font-weight:700;">■</span> Network {w_net:.0%}</span>
            <span><span style="color:#C8960C;font-weight:700;">■</span> Cohesion {w_coh:.0%}</span>
            <span style="margin-left:auto;color:var(--text-muted);">Sum = <strong style="color:#111827;">1.00</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # Formation preview
    col_form, col_info = st.columns([1, 1])
    with col_form:
        st.markdown('<div class="section-header"><span class="icon">🏟️</span>FORMATION PREVIEW</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card card-green" style="padding:0.75rem 1rem;">
            <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;letter-spacing:0.06em;">
                {formation if formation_mode == "Template" else "Custom"}</span>
            &nbsp;·&nbsp;
            <span style="font-size:0.82rem;color:var(--text-secondary);">
                {" · ".join(formation_slots)}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-header"><span class="icon">⚖️</span>NORMALIZED WEIGHTS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="obj-pill">
            <div class="obj-pill-icon">🧠</div>
            <div><div class="obj-pill-label">KPI</div>
                <div class="obj-pill-desc">Player performance (goals, xG, passes, ball wins…)</div></div>
            <div class="obj-pill-weight">{w_kpi:.2f}</div>
        </div>
        <div class="obj-pill">
            <div class="obj-pill-icon">🕸️</div>
            <div><div class="obj-pill-label">Network</div>
                <div class="obj-pill-desc">Graph-based mobility (PageRank, betweenness, strength…)</div></div>
            <div class="obj-pill-weight">{w_net:.2f}</div>
        </div>
        <div class="obj-pill">
            <div class="obj-pill-icon">🔗</div>
            <div><div class="obj-pill-label">Cohesion</div>
                <div class="obj-pill-desc">Intra-XI passing connections (passes-per-90 shared)</div></div>
            <div class="obj-pill-weight">{w_coh:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Run button ───────────────────────────
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("⚽  OPTIMIZE XI", type="primary", use_container_width=True)

    if run_btn:
        weights   = ObjectiveWeights(w_kpi=float(w_kpi), w_net=float(w_net), w_cohesion=float(w_coh))
        score_cfg = PlayerScoreConfig(
            std_penalty=float(std_penalty),
            min_minutes=float(min_minutes),
            reliability_floor=float(reliability_floor),
        )
        search_cfg = SearchConfig(
            max_iters=int(max_iters),
            candidate_samples=int(candidate_samples),
            tabu_tenure=int(tabu_tenure),
        )
        obj_cfg = ObjectiveConfig(
            component_norm=component_norm,
            cohesion_norm=cohesion_norm,
        )

        # Retrieve feature selections from session state (set in Features tab)
        use_default_kpis = st.session_state.get("use_default_kpis", True)
        use_default_net  = st.session_state.get("use_default_net", True)
        selected_kpis    = st.session_state.get("selected_kpis_manual", None)
        selected_net     = st.session_state.get("selected_net_manual", None)

        with st.spinner("Running Tabu Search optimization…"):
            try:
                result = run_for_team(
                    team_query=team_query,
                    repo_root=repo_root_path,
                    formation_slots=formation_slots,
                    weights=weights,
                    score_cfg=score_cfg,
                    objective_cfg=obj_cfg,
                    seed=int(seed),
                    search_cfg=search_cfg,
                    kpis=(None if use_default_kpis else selected_kpis),
                    mobility_metrics=(None if use_default_net else selected_net),
                    position_group_z=position_group_z,
                )
                st.session_state.result = result
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                st.session_state.result = None

    # ── Results ──────────────────────────────
    if st.session_state.result is not None:
        result = st.session_state.result
        lineup = result["lineup"]
        obj    = result["objective"]

        st.success("✅ Optimization complete!")
        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

        res_left, res_right = st.columns([1, 1])

        with res_left:
            st.markdown('<div class="section-header"><span class="icon">🏆</span>OPTIMAL XI</div>', unsafe_allow_html=True)

            # HTML pitch
            st.markdown(render_pitch(lineup, formation_slots), unsafe_allow_html=True)

        with res_right:
            st.markdown('<div class="section-header"><span class="icon">📋</span>LINEUP DETAILS</div>', unsafe_allow_html=True)

            table_rows = ""
            for slot, player in lineup.items():
                table_rows += f"""
                <tr>
                    <td>{format_slot_badge(slot)}</td>
                    <td style="font-weight:500;">{player}</td>
                </tr>"""
            st.markdown(f"""
            <table class="result-table">
                <thead><tr><th>Slot</th><th>Player</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>""", unsafe_allow_html=True)

        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">📊</span>OBJECTIVE BREAKDOWN</div>', unsafe_allow_html=True)
        st.markdown(render_objective_breakdown(obj), unsafe_allow_html=True)

        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

        with st.expander("🧬 Feature sets used"):
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**KPIs used:**")
                chips = "".join(f'<span class="metric-chip">{k}</span>' for k in result["selected_kpis"])
                st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
            with fc2:
                st.markdown("**Mobility metrics used:**")
                mchips = "".join(f'<span class="metric-chip">{m}</span>' for m in result.get("selected_mobility_metrics", []))
                # st.markdown(f'<div>{mchips if mchips else "<span style=\\'color:var(--text-muted)\\'>None</span>"}</div>', unsafe_allow_html=True)
                st.markdown(f"""<div>{mchips if mchips else '<span style="color:var(--text-muted)">None</span>'}</div>""", unsafe_allow_html=True)

        with st.expander("📦 Full config snapshot"):
            st.json({
                "weights":       result.get("weights"),
                "score_cfg":     result.get("score_cfg"),
                "cohesion_cfg":  result.get("cohesion_cfg"),
                "objective_cfg": result.get("objective_cfg"),
                "search_cfg":    result.get("search_cfg"),
            })

        with st.expander("🗂️ Raw nodes data"):
            st.dataframe(nodes_df, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — FEATURES
# ══════════════════════════════════════════════
with tab_features:
    if not files_ok:
        st.warning("Load a valid team first (check sidebar).")
        st.stop()

    st.markdown("""
    <div class="card card-accent">
        <b>Feature Selection</b> — choose which KPIs and network metrics feed into the objective function.
        Changes here affect the <em>next</em> optimization run.
    </div>
    """, unsafe_allow_html=True)

    feat_left, feat_right = st.columns([1, 1])

    with feat_left:
        st.markdown('<div class="section-header"><span class="icon">🧠</span>KPI MEAN FEATURES</div>', unsafe_allow_html=True)

        use_default_kpis = st.checkbox(
            "Use default KPI set (recommended)",
            value=st.session_state.get("use_default_kpis", True),
            help="Uses a curated list of high-signal KPIs matching the CLI --kpis default. Includes GK-specific KPIs if present."
        )
        st.session_state["use_default_kpis"] = use_default_kpis

        if use_default_kpis:
            default_used = default_kpis(nodes_df)
            st.markdown("**Default KPIs selected:**")
            chips = "".join(f'<span class="metric-chip">{"➕" if not any(k in c.upper() for k in ["LOSS","UNSUCCESSFUL"]) else "➖"} {c}</span>'
                           for c in default_used)
            st.markdown(f'<div style="margin-top:0.5rem;">{chips}</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box" style="margin-top:0.75rem;">
                ➕ = positive KPI (higher is better) &nbsp;|&nbsp; ➖ = negative KPI (auto-flipped via z-score negation)
            </div>
            """, unsafe_allow_html=True)
        else:
            default_manual = [c for c in kpi_mean_cols if any(k in c.upper() for k in
                              ["GOALS", "SHOT_XG", "PXT_PASS", "SUCCESSFUL_PASSES"])]
            manual_kpis = st.multiselect(
                "Select KPI mean columns",
                options=kpi_mean_cols,
                default=st.session_state.get("selected_kpis_manual", default_manual[:10] or kpi_mean_cols[:10]),
                help="Columns ending in _mean. Corresponding _std columns are auto-used for the inconsistency penalty."
            )
            st.session_state["selected_kpis_manual"] = manual_kpis
            if not manual_kpis:
                st.warning("No KPIs selected — optimization will fail.")

            neg_flagged = [c for c in manual_kpis if any(k in c.upper() for k in
                           ["LOSS","UNSUCCESSFUL","FOUL","YELLOW","RED","CONCEDED"])]
            if neg_flagged:
                st.markdown(f"""
                <div class="info-box">
                    🔄 <strong>Auto-negated</strong> (higher raw = worse):<br/>
                    {", ".join(f"<code>{c}</code>" for c in neg_flagged)}
                </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

        # KPI weights
        with st.expander("🎚️ Custom KPI weights (optional)"):
            st.markdown("""
            <div class="info-box">
                Override per-KPI contribution weight. Default = 1.0 for all. 
                Leave blank to use uniform weighting.
            </div>
            """, unsafe_allow_html=True)
            active_kpis = default_kpis(nodes_df) if use_default_kpis else st.session_state.get("selected_kpis_manual", [])
            kpi_w_overrides = {}
            for kpi in active_kpis[:8]:  # show up to 8 for UX
                w = st.number_input(f"w({kpi})", min_value=0.0, value=1.0, step=0.1, key=f"kpiw_{kpi}")
                kpi_w_overrides[kpi] = w
            if kpi_w_overrides:
                st.session_state["kpi_weights_override"] = kpi_w_overrides

    with feat_right:
        st.markdown('<div class="section-header"><span class="icon">🕸️</span>NETWORK / MOBILITY METRICS</div>', unsafe_allow_html=True)

        use_default_net = st.checkbox(
            "Use all net_* columns (recommended)",
            value=st.session_state.get("use_default_net", True),
            help="Uses every column starting with 'net_' in nodes.csv. Matches the CLI default behavior."
        )
        st.session_state["use_default_net"] = use_default_net

        if use_default_net:
            st.markdown(f"**All {len(net_cols)} network metrics:**")
            chips = "".join(f'<span class="metric-chip">{c}</span>' for c in net_cols)
            # st.markdown(f'<div style="margin-top:0.5rem;">{chips if chips else "<span style=\\'color:var(--text-muted)\\'>No net_* columns found</span>"}</div>', unsafe_allow_html=True)
            st.markdown(
                f"""<div style="margin-top:0.5rem;">
                    {chips if chips else '<span style="color:var(--text-muted)">No net_* columns found</span>'}
                </div>""", 
                unsafe_allow_html=True
            )
        
        else:
            manual_net = st.multiselect(
                "Select network metric columns",
                options=net_cols,
                default=st.session_state.get("selected_net_manual", net_cols),
                help="Graph metrics from the fully-connected pass network. Combined into a composite NET term via z-scores."
            )
            st.session_state["selected_net_manual"] = manual_net
            if not manual_net:
                st.warning("No network metrics selected — NET term will be zero.")

            if net_cols:
                st.markdown("**Metric preview (first 5 players):**")
                preview_cols = [c for c in manual_net if c in nodes_df.columns]
                if preview_cols and "player" in nodes_df.columns:
                    st.dataframe(
                        nodes_df[["player"] + preview_cols[:4]].head(5),
                        use_container_width=True
                    )

        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

        # Mobility weights
        with st.expander("🎚️ Custom mobility weights (optional)"):
            active_net = net_cols if use_default_net else st.session_state.get("selected_net_manual", [])
            mob_w_overrides = {}
            for m in active_net[:6]:
                mw = st.number_input(f"w({m})", min_value=0.0, value=1.0, step=0.1, key=f"netw_{m}")
                mob_w_overrides[m] = mw
            if mob_w_overrides:
                st.session_state["mob_weights_override"] = mob_w_overrides

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        ✅ Feature selections are <strong>saved automatically</strong> — click <strong>⚽ Optimize XI</strong> in the Optimizer tab to run with these settings.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — LAMBDA SWEEP
# ══════════════════════════════════════════════
with tab_sweep:
    st.markdown("""
    <div class="card card-gold">
        <b>Lambda (λ) Sweep</b> — Systematically explore how different objective weight combinations 
        change the optimal lineup. Useful for sensitivity analysis and understanding trade-offs 
        between individual KPI performance, network mobility, and team cohesion.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong><br/>
        1. Pick one weight to <em>sweep</em> across a range (the other two are kept fixed).<br/>
        2. Set the fixed values and step size.<br/>
        3. Click <strong>Run Sweep</strong>. Each point = one full optimization run.<br/>
        4. Compare resulting lineups and objective scores across the sweep axis.
    </div>
    """, unsafe_allow_html=True)

    if not files_ok:
        st.warning("Load a valid team first.")
        st.stop()

    sw1, sw2 = st.columns([1, 1])

    with sw1:
        sweep_param = st.selectbox(
            "Sweep axis (which λ to vary)",
            options=["λ_kpi", "λ_net", "λ_coh"],
            help="Select which weight to vary across the sweep range."
        )
        sweep_min   = st.number_input("Min value", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        sweep_max   = st.number_input("Max value", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        sweep_steps = st.number_input("Steps", min_value=2, max_value=20, value=5, step=1)

    with sw2:
        st.markdown("**Fixed weights for non-swept λ values:**")
        fixed_kpi = st.number_input("Fixed λ_kpi", 0.0, 1.0, w_kpi, 0.05, key="sw_fixed_kpi",
            help="Used when λ_kpi is not the sweep axis.")
        fixed_net = st.number_input("Fixed λ_net", 0.0, 1.0, w_net, 0.05, key="sw_fixed_net",
            help="Used when λ_net is not the sweep axis.")
        fixed_coh = st.number_input("Fixed λ_coh", 0.0, 1.0, w_coh, 0.05, key="sw_fixed_coh",
            help="Used when λ_coh is not the sweep axis.")
        sweep_seed = st.number_input("Seed", min_value=0, value=int(seed), step=1, key="sw_seed")
        sweep_iters = st.number_input("Iterations per run", min_value=100, value=min(int(max_iters), 1500), step=100, key="sw_iters")

    sweep_col, _ = st.columns([1, 3])
    with sweep_col:
        run_sweep_btn = st.button("🔁  RUN SWEEP", type="primary", use_container_width=True)

    if run_sweep_btn:
        sweep_values = np.linspace(float(sweep_min), float(sweep_max), int(sweep_steps))
        sweep_records = []

        score_cfg_sw = PlayerScoreConfig(
            std_penalty=float(std_penalty),
            min_minutes=float(min_minutes),
            reliability_floor=float(reliability_floor),
        )
        search_cfg_sw = SearchConfig(
            max_iters=int(sweep_iters),
            candidate_samples=int(candidate_samples),
            tabu_tenure=int(tabu_tenure),
        )
        obj_cfg_sw = ObjectiveConfig(component_norm=component_norm, cohesion_norm=cohesion_norm)

        prog = st.progress(0, text="Running sweep…")
        for i, val in enumerate(sweep_values):
            if sweep_param == "λ_kpi":
                ww = ObjectiveWeights(w_kpi=float(val), w_net=float(fixed_net), w_cohesion=float(fixed_coh))
            elif sweep_param == "λ_net":
                ww = ObjectiveWeights(w_kpi=float(fixed_kpi), w_net=float(val), w_cohesion=float(fixed_coh))
            else:
                ww = ObjectiveWeights(w_kpi=float(fixed_kpi), w_net=float(fixed_net), w_cohesion=float(val))

            try:
                res = run_for_team(
                    team_query=team_query,
                    repo_root=repo_root_path,
                    formation_slots=formation_slots,
                    weights=ww,
                    score_cfg=score_cfg_sw,
                    objective_cfg=obj_cfg_sw,
                    seed=int(sweep_seed),
                    search_cfg=search_cfg_sw,
                    kpis=None if st.session_state.get("use_default_kpis", True) else st.session_state.get("selected_kpis_manual"),
                    mobility_metrics=None if st.session_state.get("use_default_net", True) else st.session_state.get("selected_net_manual"),
                    position_group_z=position_group_z,
                )
                obj = res["objective"]
                lineup_str = " · ".join(f"{s}:{p.split()[-1]}" for s, p in res["lineup"].items())
                sweep_records.append({
                    sweep_param: round(float(val), 3),
                    "total": round(obj["total"], 5),
                    "kpi_norm": round(obj.get("kpi_norm", 0), 4),
                    "net_norm": round(obj.get("net_norm", 0), 4),
                    "cohesion_norm": round(obj.get("cohesion_norm", 0), 4),
                    "lineup": lineup_str,
                })
            except Exception as exc:
                sweep_records.append({sweep_param: round(float(val), 3), "total": None, "error": str(exc), "lineup": "ERROR"})

            prog.progress((i + 1) / len(sweep_values), text=f"Sweep step {i+1}/{len(sweep_values)}")

        prog.empty()
        st.session_state.sweep_results = (sweep_param, sweep_records)

    if st.session_state.sweep_results is not None:
        sw_axis, sw_records = st.session_state.sweep_results
        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><span class="icon">📈</span>SWEEP RESULTS — {sw_axis}</div>', unsafe_allow_html=True)

        sw_df = pd.DataFrame(sw_records)

        # Chart
        if "total" in sw_df.columns and sw_df["total"].notna().sum() > 1:
            chart_df = sw_df[[sw_axis, "total", "kpi_norm", "net_norm", "cohesion_norm"]].dropna()
            st.line_chart(chart_df.set_index(sw_axis)[["total", "kpi_norm", "net_norm", "cohesion_norm"]])

        # Best lineup highlight
        if sw_df["total"].notna().sum() > 0:
            best_idx = sw_df["total"].idxmax()
            best_row = sw_df.iloc[best_idx]
            st.markdown(f"""
            <div class="card card-gold">
                🏆 <strong>Best run</strong> — {sw_axis} = <code>{best_row[sw_axis]}</code> &nbsp;|&nbsp;
                Total = <code>{best_row['total']}</code><br/>
                <span style="font-size:0.82rem;color:var(--text-secondary);">{best_row.get('lineup','')}</span>
            </div>
            """, unsafe_allow_html=True)

        # Table with highlight
        st.dataframe(
            sw_df.drop(columns=["lineup"], errors="ignore"),
            use_container_width=True
        )

        with st.expander("📋 Full lineup strings per sweep step"):
            for rec in sw_records:
                label = f"{sw_axis}={rec[sw_axis]}"
                total_val = rec.get("total", "ERR")
                st.markdown(f"**{label}** (total={total_val}): `{rec.get('lineup','')}`")


# ══════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    about_l, about_r = st.columns([3, 2])

    with about_l:
        st.markdown("""
        <div class="section-header"><span class="icon">📖</span>ABOUT THIS TOOLKIT</div>

        <div class="card card-accent">
            <p style="margin:0; line-height:1.7;">
                <strong>Big O(Goal) F.C.</strong> is a data-driven Bundesliga lineup optimizer that combines
                three complementary signals — individual player statistics, graph-theoretic passing network
                analysis, and within-XI team cohesion — into a single scalarized objective function, 
                solved via <strong>Tabu Search</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;"><span class="icon">🔢</span>THE OBJECTIVE FUNCTION</div>

        <div class="card" style="margin-bottom:1rem;">
            <p style="font-size:0.9rem; line-height:1.7; color:var(--text-secondary); margin:0 0 0.75rem;">
                The optimizer selects 11 players (one per formation slot) to <strong style="color:var(--text-primary);">maximize a single score</strong> 
                built from three complementary signals, each normalized to [0,1] before being combined:
            </p>
            <div style="text-align:center; padding:0.75rem; background:var(--surface-2); border-radius:8px; margin-bottom:1rem;">
                <span style="font-size:1.05rem; font-weight:600; color:var(--text-primary); font-family:'DM Sans',sans-serif;">
                    f(XI) &nbsp;=&nbsp; 
                    <span style="color:#D0021B;">λ<sub>kpi</sub> · KPI_score</span>
                    &nbsp;+&nbsp;
                    <span style="color:#1A7A3A;">λ<sub>net</sub> · NET_score</span>
                    &nbsp;+&nbsp;
                    <span style="color:#C8960C;">λ<sub>coh</sub> · COH_score</span>
                    &nbsp;&nbsp;,&nbsp;&nbsp; λ sums to 1
                </span>
            </div>

            <div style="display:flex; flex-direction:column; gap:0.75rem;">
                <div style="border-left:4px solid #D0021B; padding:0.75rem 1rem; background:rgba(208,2,27,0.05); border-radius:0 8px 8px 0;">
                    <div style="font-weight:700; color:#D0021B; font-size:0.95rem; margin-bottom:0.3rem;">🧠 KPI Score — Individual Player Quality</div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.6;">
                        For each KPI (e.g. goals, xG, successful passes), player values are <strong style="color:var(--text-primary);">z-scored within their positional group</strong> (GK / DEF / MID / ATT) so a goalkeeper is only compared to other goalkeepers.
                        Negative KPIs like ball losses are automatically <strong style="color:var(--text-primary);">flipped in sign</strong>.
                        A <strong style="color:var(--text-primary);">consistency penalty</strong> subtracts a fraction of each KPI's standard deviation across matches, penalising streaky players.
                        The mean z-score across all KPIs gives each player a single performance number.
                    </div>
                </div>

                <div style="border-left:4px solid #1A7A3A; padding:0.75rem 1rem; background:rgba(26,122,58,0.05); border-radius:0 8px 8px 0;">
                    <div style="font-weight:700; color:#1A7A3A; font-size:0.95rem; margin-bottom:0.3rem;">🕸️ Network Score — Pass Network Centrality</div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.6;">
                        Computed from graph metrics in the <strong style="color:var(--text-primary);">fully-connected pass network</strong> (all players connected, weighted by passes-per-90 during shared minutes).
                        Metrics such as <strong style="color:var(--text-primary);">PageRank</strong> (who receives the ball from important players), 
                        <strong style="color:var(--text-primary);">betweenness centrality</strong> (who sits on most shortest paths), and 
                        <strong style="color:var(--text-primary);">weighted degree / strength</strong> (total passing volume) are z-scored and combined.
                        High network score → the player is a structural hub of the team's passing game.
                    </div>
                </div>

                <div style="border-left:4px solid #C8960C; padding:0.75rem 1rem; background:rgba(200,150,12,0.05); border-radius:0 8px 8px 0;">
                    <div style="font-weight:700; color:#C8960C; font-size:0.95rem; margin-bottom:0.3rem;">🔗 Cohesion Score — Within-XI Connection Density</div>
                    <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.6;">
                        Unlike the network score (which is per-player), cohesion is a <strong style="color:var(--text-primary);">team-level measure</strong>.
                        It sums the passes-per-90 on every directed edge that connects two players <em>both inside the selected XI</em>.
                        A high cohesion score means the 11 players have historically passed to each other often — they play as a connected unit, not isolated individuals.
                        Normalised by the maximum observed edge weight so the value sits in [0, 1].
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;"><span class="icon">🔬</span>SEARCH ALGORITHM — TABU SEARCH</div>
        <div class="card">
            <ol style="line-height:1.9; font-size:0.88rem; color:var(--text-secondary); margin:0; padding-left:1.2rem;">
                <li><strong style="color:var(--text-primary);">Greedy initialization</strong> — Fill each slot left-to-right with the best eligible player given the partial lineup.</li>
                <li><strong style="color:var(--text-primary);">Swap neighborhood</strong> — At each iteration, randomly pick a slot and sample <em>k</em> candidate replacements.</li>
                <li><strong style="color:var(--text-primary);">Tabu list</strong> — Recent (slot, in, out) swaps are forbidden for <em>tenure</em> iterations to escape local optima.</li>
                <li><strong style="color:var(--text-primary);">Acceptance</strong> — Accept any non-tabu swap that strictly improves the objective.</li>
                <li><strong style="color:var(--text-primary);">Termination</strong> — After <em>max_iters</em> iterations or when no improving swap is found.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with about_r:
        st.markdown("""
        <div class="section-header"><span class="icon">🗂️</span>DATA FORMAT</div>
        <div class="card card-green">
            <p style="font-size:0.85rem; line-height:1.7; margin:0; color:var(--text-secondary);">
                <strong style="color:var(--text-primary);">nodes.csv</strong> — One row per player.<br/>
                Required: <code>player_id</code>, <code>player</code>, <code>positions</code><br/>
                Optional: <code>*_mean</code>, <code>*_std</code> KPI columns; <code>net_*</code> network columns.<br/><br/>
                <strong style="color:var(--text-primary);">edges.csv</strong> — Fully-connected pass network.<br/>
                Required: <code>from_id</code>, <code>to_id</code><br/>
                Optional: <code>passes_per90_shared</code> (for cohesion), <code>shared_minutes</code>
            </p>
        </div>

        <div class="section-header" style="margin-top:1rem;"><span class="icon">⌨️</span>CLI EQUIVALENT</div>
        <div class="card">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; 
                        background:var(--surface-2); padding:0.75rem 1rem; border-radius:6px;
                        border:1px solid var(--border); white-space:pre-wrap; color:var(--text-secondary);">
python main.py \\
  --team fc_bayern_muenchen \\
  --repo_root . \\
  --formation 4-3-3 \\
  --w_player 0.60 \\
  --w_centrality 0.10 \\
  --w_cohesion 0.30 \\
  --min_minutes 200 \\
  --std_penalty 0.35 \\
  --kpis GOALS_mean,SHOT_XG_mean \\
  --mobility_metrics net_pagerank
            </div>
        </div>

        <div class="section-header" style="margin-top:1rem;"><span class="icon">💡</span>QUICK TIPS</div>
        <div class="card">
            <ul style="font-size:0.85rem; line-height:1.8; color:var(--text-secondary); margin:0; padding-left:1.2rem;">
                <li>Start with <strong style="color:var(--text-primary);">default KPI + network</strong> settings — they match the CLI defaults.</li>
                <li>Use <strong style="color:var(--text-primary);">Lambda Sweep</strong> to see how sensitive the lineup is to weight changes.</li>
                <li>Increase <strong style="color:var(--text-primary);">Tabu tenure</strong> if the optimizer seems to cycle. Lower it for faster exploration.</li>
                <li><strong style="color:var(--text-primary);">Position-group z-scoring</strong> (ON by default) prevents GKs dominating attacking KPIs.</li>
                <li>A high <strong style="color:var(--text-primary);">cohesion weight</strong> tends to lock in core-squad combinations who play together often.</li>
                <li>Use <strong style="color:var(--text-primary);">min minutes ≥ 200</strong> to filter out players without meaningful data.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; font-size:0.78rem; color:var(--text-muted); line-height:2;">
        Big O(Goal) F.C. · Built on IMPECT Open Data · Bundesliga ·
        Tabu Search Optimization · Network Analytics · Expected Goals<br/>
        <span style="color:var(--bundesliga-red);">⚽</span>
    </div>
    """, unsafe_allow_html=True)
