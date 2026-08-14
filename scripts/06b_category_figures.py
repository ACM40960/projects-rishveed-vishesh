import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/shots_with_features.csv")
NAVY, GREEN = "#1F3864", "#2E7D5B"

def conv(col):
    g = df.groupby(col)["goal"].agg(n="size", c="mean")
    g = g[g["n"] >= 100].sort_values("c")
    return g.index.astype(str), (100*g["c"]).values

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

lab, val = conv("shot_body_part")
ax[0,0].barh(lab, val, color=NAVY); ax[0,0].set_title("Conversion by body part (%)")

lab, val = conv("shot_technique")
ax[0,1].barh(lab, val, color=GREEN); ax[0,1].set_title("Conversion by technique (%)")

lab, val = conv("play_pattern")
ax[1,0].barh(lab, val, color=NAVY); ax[1,0].set_title("Conversion by play pattern (%)")

# context: first-time and under-pressure
ctx_labels, ctx_vals = [], []
for name, col in [("First-time", "shot_first_time"), ("Under pressure", "under_pressure")]:
    for flag in [False, True]:
        sub = df[df[col] == flag]
        ctx_labels.append(f"{name}: {'yes' if flag else 'no'}")
        ctx_vals.append(100*sub["goal"].mean())
ax[1,1].barh(ctx_labels, ctx_vals, color=GREEN); ax[1,1].set_title("Conversion by context (%)")

for a in ax.flat: a.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("outputs/category_conversion.png", dpi=140, bbox_inches="tight")
print("saved outputs/category_conversion.png")
