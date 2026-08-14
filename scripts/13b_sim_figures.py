import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
sh = pd.read_csv("data/shots_with_xg.csv")
fx = pd.read_csv("data/fixtures.csv")
tab = pd.read_csv("outputs/xg_league_tables.csv")

shotmap = {}
for (mid, team), grp in sh.groupby(["match_id","team"]):
    shotmap.setdefault(mid, {})[team] = grp["xg"].to_numpy()

# ---- Figure A: actual points vs expected points (all leagues) ----
colors = {"La Liga":"#1F3864","Premier League":"#2E7D5B","Serie A":"#C6612F","Ligue 1":"#6A4C93"}
fig, ax = plt.subplots(figsize=(8,8))
for comp, c in colors.items():
    d = tab[tab.competition==comp]
    ax.scatter(d.xP, d.actP, color=c, label=comp, s=35, alpha=.8)
lim=[15,95]; ax.plot(lim, lim, "--", color="grey")
# annotate a few notable teams
notable = ["Leicester City","Atlético Madrid","Juventus","Aston Villa","Toulouse","Hellas Verona"]
for _, r in tab[tab.team.isin(notable)].iterrows():
    ax.annotate(r.team, (r.xP, r.actP), fontsize=8,
                xytext=(4,4), textcoords="offset points")
ax.set_xlabel("expected points from xG"); ax.set_ylabel("actual points")
ax.set_title("Actual vs expected points, 2015/16\n(above line = overperformed their chances)")
ax.legend(); ax.grid(alpha=.3); ax.set_xlim(lim); ax.set_ylim(lim)
plt.tight_layout(); plt.savefig("outputs/xpoints_vs_actual.png", dpi=140, bbox_inches="tight")

# ---- Figure B: one match, exact Poisson-binomial vs Monte Carlo ----
def pb_pmf(ps):                      # exact Poisson-binomial distribution
    dist = np.array([1.0])
    for p in ps: dist = np.convolve(dist, [1-p, p])
    return dist

# pick the match with the most total shots for a rich example
best_mid = max(shotmap, key=lambda m: sum(len(v) for v in shotmap[m].values()))
row = fx[fx.match_id==best_mid].iloc[0]
h, a = row.home_team, row.away_team
hxg = shotmap[best_mid].get(h, np.array([])); axg = shotmap[best_mid].get(a, np.array([]))
ph, pa = pb_pmf(hxg), pb_pmf(axg)

# Monte Carlo for the same match
N=20000
hg = (rng.random((N,len(hxg))) < hxg).sum(1)
ag = (rng.random((N,len(axg))) < axg).sum(1)
pwin = (hg>ag).mean(); pdraw=(hg==ag).mean(); plose=(hg<ag).mean()

fig, ax = plt.subplots(1,2, figsize=(13,5))
k = np.arange(0, max(len(ph),len(pa)))
ax[0].bar(np.arange(len(ph))-0.2, ph, width=0.4, color="#1F3864", label=f"{h} (exact)")
ax[0].bar(np.arange(len(pa))+0.2, pa, width=0.4, color="#C6612F", label=f"{a} (exact)")
mh = np.bincount(hg, minlength=len(ph))/N
ax[0].scatter(np.arange(len(ph)), mh, color="black", zorder=3, s=20, label="Monte Carlo")
ax[0].set_xlabel("goals"); ax[0].set_ylabel("probability")
ax[0].set_title("Goal distributions (Poisson-binomial)"); ax[0].legend(fontsize=8)
ax[0].set_xlim(-0.6, 6.6)

ax[1].bar([f"{h}\nwin","draw",f"{a}\nwin"], [pwin,pdraw,plose],
          color=["#1F3864","grey","#C6612F"])
ax[1].set_ylabel("probability")
ax[1].set_title(f"Match outcome probabilities\n(actual result {row.home_score}-{row.away_score})")
for i,v in enumerate([pwin,pdraw,plose]): ax[1].text(i, v+0.01, f"{v:.0%}", ha="center")
plt.tight_layout(); plt.savefig("outputs/example_match_sim.png", dpi=140, bbox_inches="tight")
print(f"example match: {h} vs {a} ({row.home_score}-{row.away_score})")
print("win/draw/lose: %.0f%% / %.0f%% / %.0f%%" % (100*pwin,100*pdraw,100*plose))
print("saved xpoints_vs_actual.png and example_match_sim.png")
