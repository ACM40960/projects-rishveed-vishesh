# Step 3: turn each shot's (x, y) into two geometric features:
#   distance  -> how far the shot is from the centre of the goal
#   angle     -> how wide the goal looks from where the shot was taken
# Then save the enriched data and draw the supporting figures.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/shots_2015_16_top4.csv")

# StatsBomb pitch is 120 long x 80 wide. The goal being attacked is at x = 120,
# with posts at y = 36 and y = 44 (8 units wide), so the centre is (120, 40).
GX = 120.0
POST_LOW, POST_HIGH = 36.0, 44.0
GOAL_Y = 40.0
W = POST_HIGH - POST_LOW            # goal width = 8

x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

# distance to the centre of the goal
df["distance"] = np.sqrt((GX - x) ** 2 + (GOAL_Y - y) ** 2)

# angle subtended by the two posts, via the law of cosines
p = np.sqrt((GX - x) ** 2 + (POST_LOW - y) ** 2)     # shot to near/low post
q = np.sqrt((GX - x) ** 2 + (POST_HIGH - y) ** 2)    # shot to far/high post
cos_theta = (p ** 2 + q ** 2 - W ** 2) / (2 * p * q)
cos_theta = np.clip(cos_theta, -1.0, 1.0)            # guard against tiny float overspill
df["angle_rad"] = np.arccos(cos_theta)
df["angle_deg"] = np.degrees(df["angle_rad"])

df.to_csv("data/shots_with_features.csv", index=False)
print("saved data/shots_with_features.csv")
print("distance: min %.1f  mean %.1f  max %.1f" % (df.distance.min(), df.distance.mean(), df.distance.max()))
print("angle_deg: min %.1f  mean %.1f  max %.1f" % (df.angle_deg.min(), df.angle_deg.mean(), df.angle_deg.max()))

# ---------- Figure 1: iso-distance and iso-angle curves on the pitch ----------
gx = np.linspace(60, 120, 300)
gy = np.linspace(0, 80, 300)
GXg, GYg = np.meshgrid(gx, gy)
Dg = np.sqrt((GX - GXg) ** 2 + (GOAL_Y - GYg) ** 2)
pg = np.sqrt((GX - GXg) ** 2 + (POST_LOW - GYg) ** 2)
qg = np.sqrt((GX - GXg) ** 2 + (POST_HIGH - GYg) ** 2)
Ag = np.degrees(np.arccos(np.clip((pg**2 + qg**2 - W**2) / (2*pg*qg), -1, 1)))

def draw_pitch(ax):
    ax.plot([60,120,120,60,60],[0,0,80,80,0], color="black", lw=1)
    ax.plot([POST_LOW*0+120,120],[POST_LOW,POST_HIGH], color="red", lw=4)  # goal
    ax.set_xlim(60,122); ax.set_ylim(-2,82); ax.set_aspect("equal"); ax.axis("off")

fig, axes = plt.subplots(1, 2, figsize=(13,5))
c0 = axes[0].contour(GXg, GYg, Dg, levels=[6,12,18,24,30,36], colors="#1F3864")
axes[0].clabel(c0, inline=True, fontsize=8, fmt="%.0f")
draw_pitch(axes[0]); axes[0].set_title("Iso-distance curves (units to goal centre)")

c1 = axes[1].contour(GXg, GYg, Ag, levels=[5,10,20,30,45,60,90], colors="#2E7D5B")
axes[1].clabel(c1, inline=True, fontsize=8, fmt="%.0f°")
draw_pitch(axes[1]); axes[1].set_title("Iso-angle curves (goal width seen, degrees)")
plt.tight_layout(); plt.savefig("outputs/geometry_iso_curves.png", dpi=140, bbox_inches="tight")
print("saved outputs/geometry_iso_curves.png")

# ---------- Figure 2: do the features actually predict goals? ----------
fig, axes = plt.subplots(1, 2, figsize=(13,4.5))
db = pd.cut(df["distance"], bins=np.arange(0,45,3))
conv_d = df.groupby(db, observed=True)["goal"].mean()
axes[0].plot([iv.mid for iv in conv_d.index], conv_d.values*100, "o-", color="#1F3864")
axes[0].set_xlabel("distance to goal (units)"); axes[0].set_ylabel("goals scored (%)")
axes[0].set_title("Conversion falls with distance"); axes[0].grid(alpha=.3)

ab = pd.cut(df["angle_deg"], bins=np.arange(0,95,7))
conv_a = df.groupby(ab, observed=True)["goal"].mean()
axes[1].plot([iv.mid for iv in conv_a.index], conv_a.values*100, "o-", color="#2E7D5B")
axes[1].set_xlabel("angle to goal (degrees)"); axes[1].set_ylabel("goals scored (%)")
axes[1].set_title("Conversion rises with a wider angle"); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/feature_validation.png", dpi=140, bbox_inches="tight")
print("saved outputs/feature_validation.png")
