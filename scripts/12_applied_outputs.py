# Step 12: applied outputs from the final model (XGBoost).
#  1. out-of-fold xG for every shot (so no shot is scored by a model that saw it)
#  2. a shot map coloured by xG, and an xG surface over the pitch
#  3. a player finishing table: actual goals minus expected goals

import json, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/model_features.csv")
FEATURES = json.load(open("data/feature_list.json"))
X, y = df[FEATURES], df["goal"]

xgb = XGBClassifier(n_estimators=115, max_depth=4, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                    eval_metric="logloss", random_state=42, n_jobs=-1)

# out-of-fold predictions: each shot scored by a fold that excluded it
cv = StratifiedKFold(5, shuffle=True, random_state=42)
df["xg"] = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
df.to_csv("data/shots_with_xg.csv", index=False)
print("total shots %d | total goals %d | total xG %.1f" % (len(df), df.goal.sum(), df.xg.sum()))

# ---------------- pitch drawing helper (attacking half, goal on right) ----------------
def pitch(ax):
    ax.plot([60,120,120,60,60],[0,0,80,80,0], color="black", lw=1.2)
    ax.plot([102,102],[18,62], color="black", lw=1)          # penalty box
    ax.plot([102,120],[18,18], color="black", lw=1)
    ax.plot([102,120],[62,62], color="black", lw=1)
    ax.plot([114,114],[30,50], color="black", lw=1)          # six-yard box
    ax.plot([114,120],[30,30], color="black", lw=1)
    ax.plot([114,120],[50,50], color="black", lw=1)
    ax.plot([120,120],[36,44], color="red", lw=4)            # goal
    ax.scatter([108],[40], s=8, color="black")               # penalty spot
    ax.set_xlim(58,122); ax.set_ylim(-2,82); ax.set_aspect("equal"); ax.axis("off")

# ---------------- Figure 1: shot map coloured by xG ----------------
fig, ax = plt.subplots(figsize=(9,6))
s = df.sample(min(6000, len(df)), random_state=1)      # subsample so points are visible
sc = ax.scatter(s["x"], s["y"], c=s["xg"], cmap="viridis", s=8, alpha=.6)
plt.colorbar(sc, label="xG"); pitch(ax); ax.set_title("Shot map coloured by xG (sample of shots)")
plt.tight_layout(); plt.savefig("outputs/shot_map_xg.png", dpi=140, bbox_inches="tight")

# ---------------- Figure 2: xG surface for a standard open-play shot ----------------
gx = np.linspace(84,120,200); gy = np.linspace(0,80,200)
GX, GY = np.meshgrid(gx, gy)
dist = np.sqrt((120-GX)**2 + (40-GY)**2)
p = np.sqrt((120-GX)**2 + (36-GY)**2); q = np.sqrt((120-GX)**2 + (44-GY)**2)
ang = np.degrees(np.arccos(np.clip((p**2+q**2-64)/(2*p*q), -1, 1)))
grid = pd.DataFrame(0, index=np.arange(GX.size), columns=FEATURES)   # all features 0 = typical shot
grid["distance"] = dist.ravel(); grid["angle_deg"] = ang.ravel()
xgb.fit(X, y, verbose=False)
surf = xgb.predict_proba(grid)[:,1].reshape(GX.shape)
fig, ax = plt.subplots(figsize=(8,5.5))
im = ax.imshow(surf, extent=[84,120,0,80], origin="lower", cmap="viridis", aspect="equal")
ax.plot([120,120],[36,44], color="red", lw=4); plt.colorbar(im, label="xG")
ax.set_title("XGBoost xG surface (standard open-play shot)"); ax.axis("off")
plt.tight_layout(); plt.savefig("outputs/xg_surface_xgb.png", dpi=140, bbox_inches="tight")

# ---------------- Player finishing table ----------------
g = (df.groupby("player")
       .agg(shots=("goal","size"), goals=("goal","sum"),
            xg=("xg","sum"), sb_xg=("shot_statsbomb_xg","sum"))
       .reset_index())
g = g[g["shots"] >= 40].copy()
g["diff"] = g["goals"] - g["xg"]          # goals above expectation
g = g.round({"xg":1,"sb_xg":1,"diff":1})
g.sort_values("diff", ascending=False).to_csv("outputs/finishing_table.csv", index=False)

print("\nTOP over-performers (scored more than expected):")
print(g.sort_values("diff", ascending=False).head(10)[["player","shots","goals","xg","diff"]].to_string(index=False))
print("\nTOP under-performers (scored fewer than expected):")
print(g.sort_values("diff").head(10)[["player","shots","goals","xg","diff"]].to_string(index=False))

# diverging bar figure: 10 best and 10 worst
top = g.sort_values("diff", ascending=False).head(10)
bot = g.sort_values("diff").head(10)
sel = pd.concat([bot, top]).sort_values("diff")
fig, ax = plt.subplots(figsize=(9,8))
colors = ["#C0392B" if d < 0 else "#2E7D5B" for d in sel["diff"]]
ax.barh(sel["player"], sel["diff"], color=colors)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("goals minus xG  (right = clinical, left = wasteful)")
ax.set_title("Finishing: goals above/below expectation (min 40 shots)")
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("outputs/finishing_table.png", dpi=140, bbox_inches="tight")
print("\nsaved shot_map_xg.png, xg_surface_xgb.png, finishing_table.csv/.png")
