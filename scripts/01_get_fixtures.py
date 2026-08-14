# Step 1: download the real match results (home/away teams and scores) for the
# four 2015/16 leagues. Needed later by the match simulation (Step 13).
import os
import pandas as pd
from statsbombpy import sb

os.makedirs("data", exist_ok=True)
comps = {"La Liga": 11, "Premier League": 2, "Serie A": 12, "Ligue 1": 7}
rows = []
for name, cid in comps.items():
    m = sb.matches(competition_id=cid, season_id=27)
    m = m[["match_id", "home_team", "away_team", "home_score", "away_score"]].copy()
    m["competition"] = name
    rows.append(m)
fx = pd.concat(rows, ignore_index=True)
fx.to_csv("data/fixtures.csv", index=False)
print("saved data/fixtures.csv:", len(fx), "matches")
