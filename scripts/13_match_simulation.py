# Step 13: Monte Carlo match simulation from shot xG.
# Each shot is a weighted coin flip (Bernoulli with prob = its xG). A team's
# goals in a match are the sum of these flips, which follows a Poisson-binomial
# distribution. Simulating many times gives win/draw/loss probabilities and the
# expected points a team deserved from its chances.

import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 10000   # simulations per match

sh = pd.read_csv("data/shots_with_xg.csv")
fx = pd.read_csv("data/fixtures.csv")

# match_id -> {team -> array of shot xG}
shotmap = {}
for (mid, team), grp in sh.groupby(["match_id", "team"]):
    shotmap.setdefault(mid, {})[team] = grp["xg"].to_numpy()

def sim_goals(xgs):
    if len(xgs) == 0:
        return np.zeros(N, dtype=int)
    draws = rng.random((N, len(xgs))) < xgs      # coin flip per shot per sim
    return draws.sum(axis=1)

# accumulate expected points and actual points per (competition, team)
acc = {}
def add(comp, team, xpts, apts):
    k = (comp, team)
    if k not in acc: acc[k] = {"xP":0.0, "actP":0, "games":0}
    acc[k]["xP"] += xpts; acc[k]["actP"] += apts; acc[k]["games"] += 1

for r in fx.itertuples(index=False):
    teams = shotmap.get(r.match_id, {})
    hg = sim_goals(teams.get(r.home_team, np.array([])))
    ag = sim_goals(teams.get(r.away_team, np.array([])))
    # simulated points from shot quality
    xP_home = np.where(hg > ag, 3, np.where(hg == ag, 1, 0)).mean()
    xP_away = np.where(ag > hg, 3, np.where(hg == ag, 1, 0)).mean()
    # actual points from the real scoreline
    if r.home_score > r.away_score:   aP_home, aP_away = 3, 0
    elif r.home_score < r.away_score: aP_home, aP_away = 0, 3
    else:                             aP_home, aP_away = 1, 1
    add(r.competition, r.home_team, xP_home, aP_home)
    add(r.competition, r.away_team, xP_away, aP_away)

tab = (pd.DataFrame([{"competition":c, "team":t, **v} for (c,t),v in acc.items()])
         .assign(xP=lambda d: d.xP.round(1),
                 diff=lambda d: (d.actP - d.xP).round(1))
         .sort_values(["competition","actP"], ascending=[True, False]))
tab.to_csv("outputs/xg_league_tables.csv", index=False)

# show the La Liga table as an example
ll = tab[tab.competition=="La Liga"].sort_values("actP", ascending=False)
print("La Liga: actual points vs expected points (from xG)")
print(ll[["team","games","actP","xP","diff"]].head(20).to_string(index=False))

# biggest over/under performers across all four leagues
print("\nMost OVER-performing (more real points than xG deserved):")
print(tab.sort_values("diff", ascending=False).head(6)[["competition","team","actP","xP","diff"]].to_string(index=False))
print("\nMost UNDER-performing:")
print(tab.sort_values("diff").head(6)[["competition","team","actP","xP","diff"]].to_string(index=False))
