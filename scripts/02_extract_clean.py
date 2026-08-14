# Step 2: pull all shots from the four full 2015/16 leagues, clean them,
# and save one tidy table to data/shots_2015_16_top4.csv
#
# Run this once. It downloads ~1500 matches, so it takes a few minutes.
# After it saves the CSV you never need to download again: just read the CSV.

from statsbombpy import sb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# the four leagues that have a full 2015/16 season in the open data
LEAGUES = {"La Liga": 11, "Premier League": 2, "Serie A": 12, "Ligue 1": 7}
SEASON = 27  # 2015/2016

# columns we want to keep off each shot (some may be missing in older rows)
KEEP = ["match_id", "minute", "period", "player", "team", "play_pattern",
        "shot_body_part", "shot_technique", "shot_type", "shot_outcome",
        "shot_first_time", "under_pressure", "shot_key_pass_id",
        "shot_statsbomb_xg", "location"]


def shots_for_match(match_id, competition):
    # download one match, keep only the shot events
    ev = sb.events(match_id=match_id)
    s = ev[ev["type"] == "Shot"].copy()
    for c in KEEP:
        if c not in s.columns:
            s[c] = pd.NA          # make sure every column exists
    s = s[KEEP]
    s["competition"] = competition
    return s


def collect_league(name, cid):
    matches = sb.matches(competition_id=cid, season_id=SEASON)
    ids = matches["match_id"].tolist()
    out = []
    # download several matches at once to save time
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(shots_for_match, mid, name): mid for mid in ids}
        for f in as_completed(futures):
            out.append(f.result())
    league = pd.concat(out, ignore_index=True)
    print(f"{name}: {len(ids)} matches, {len(league)} shots")
    return league


# pull all four leagues
all_shots = pd.concat([collect_league(n, c) for n, c in LEAGUES.items()],
                      ignore_index=True)

# ---- cleaning ----
# split the [x, y] location into two plain number columns
all_shots["x"] = all_shots["location"].apply(lambda v: v[0] if isinstance(v, list) else pd.NA)
all_shots["y"] = all_shots["location"].apply(lambda v: v[1] if isinstance(v, list) else pd.NA)
all_shots = all_shots.drop(columns=["location"])

# make a simple 0/1 goal column
all_shots["goal"] = (all_shots["shot_outcome"] == "Goal").astype(int)

# under_pressure is True or missing -> turn into True/False
all_shots["under_pressure"] = all_shots["under_pressure"].fillna(False).astype(bool)
all_shots["shot_first_time"] = all_shots["shot_first_time"].fillna(False).astype(bool)

# drop penalties: they are always from the spot and would distort a distance/angle model
all_shots = all_shots[all_shots["shot_type"] != "Penalty"].copy()

# drop any shot with no location, since we need x and y for the geometry
all_shots = all_shots.dropna(subset=["x", "y"]).reset_index(drop=True)

all_shots.to_csv("data/shots_2015_16_top4.csv", index=False)
print("\nSAVED data/shots_2015_16_top4.csv")
print("total shots:", len(all_shots), "| goals:", all_shots["goal"].sum(),
      "| conversion: %.1f%%" % (100 * all_shots["goal"].mean()))
