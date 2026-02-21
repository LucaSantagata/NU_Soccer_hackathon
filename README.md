# IMPECT Lineup Optimizer (Passing Network + KPIs)

This project selects an **optimal starting XI** for a given team using:

1) **Player KPIs** (mean and std across matches)  
2) **Mobility / network metrics** computed on the season passing network (e.g., centralities)  
3) **Within-XI cohesion** derived from the pass-rate network among the selected XI  
4) **Formation constraints** using each player’s season position list (`positions` column)

The optimizer is interactive (Streamlit UI) and scriptable (CLI).

---

## Data layout (expected)

Place these two folders at the repo root:

```
Fully_connected_team_networks/
    <team>__fully_connected.csv
Fully_connected_team_networks_with_kpis_and_netmetrics/
    <team>__nodes.csv
```

Team name queries are matched fuzzily against filenames (prefixes like `1_...` are supported).

### Edge file (`__fully_connected.csv`) required columns
- `from_id` (int)
- `to_id` (int)
- `shared_minutes` (float)
- `passes_per90_shared` (float)  ← used as edge weight for cohesion

### Node file (`__nodes.csv`) required columns
- `player_id` (int)
- `player` (string)
- `positions` (list-like string)  
  Example cell:
  ```
  [ATTACKING_MIDFIELD (Centre), RIGHT_WINGER (Right)]
  ```
- KPI columns: typically `*_mean` and `*_std`
- network metric columns: typically `net_*` (pagerank, betweenness, strength, etc.)

---

## Default behavior (important)

If the user does **not** specify features:

- **KPIs:** uses a curated, fixed KPI set (stable + interpretable).  
  If goalkeeper-specific KPIs exist (e.g., saves, xG prevented, conceded, clean sheets), they’re automatically added.
- **Network metrics:** uses **ALL** network metrics (all `net_*` columns in `nodes.csv`).

If the user specifies KPIs and/or network metrics, the optimizer uses exactly those.

---

## Run from the command line

### Basic run (defaults)
```bash
python main.py --team bayern --formation 4-3-3
```

### Choose formation
```bash
python main.py --team bayern --formation 4-2-3-1
```

### Manual KPI list (comma-separated KPI mean columns)
```bash
python main.py --team bayern --kpis GOALS_mean,SHOT_XG_mean,PXT_PASS_mean
```

### Manual network metric list
```bash
python main.py --team bayern --mobility_metrics net_pagerank,net_strength,net_betweenness
```

### Weights / filtering
```bash
python main.py --team bayern --w_player 0.6 --w_centrality 0.2 --w_cohesion 0.2 --min_minutes 800 --std_penalty 0.4
```

---

## Streamlit UI

Install Streamlit:
```bash
pip install streamlit
```

Run:
```bash
streamlit run app.py
```

In the UI you can:
- type a team name
- choose a formation template (default 4-3-3) or custom slots
- keep defaults (recommended) or manually select KPIs / network metrics
- tune weights, minutes filter, and search iterations
- press **Optimize XI**

---

## Formation templates and duplicate roles

Formations use **unique slot identifiers** for duplicates (e.g., `CB1`, `CB2`, `ST1`, `ST2`) so both players are retained in outputs.

---

## Troubleshooting

### “Output has 10 players”
This happens if a formation uses duplicate slot names like `CB` repeated (dict overwrite). Use unique slot IDs (`CB1`, `CB2`) or the provided templates.

### “No eligible player found for slot”
Your `positions` strings may contain labels not covered by the mapping. Extend the mapping in `position_utils.py`.

---

## Citation
If you use this framework in research, please cite the accompanying manuscript in this repository.
