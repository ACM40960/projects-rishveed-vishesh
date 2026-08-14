import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/shots_with_features.csv")
c = pd.read_csv("outputs/baseline_coefficients.csv", index_col=0)["coef (log-odds)"]
b0, bd, ba = c["const"], c["distance"], c["angle_deg"]
sig = lambda z: 1/(1+np.exp(-z))

# ---- Figure 1: fitted sigmoid curves ----
med_a = df["angle_deg"].median(); med_d = df["distance"].median()
dd = np.linspace(0, 40, 200)
aa = np.linspace(0, 90, 200)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(dd, sig(b0 + bd*dd + ba*med_a), color="#1F3864")
ax[0].set_xlabel("distance to goal"); ax[0].set_ylabel("predicted xG")
ax[0].set_title(f"xG vs distance (angle fixed at {med_a:.0f}°)"); ax[0].grid(alpha=.3)
ax[1].plot(aa, sig(b0 + bd*med_d + ba*aa), color="#2E7D5B")
ax[1].set_xlabel("angle to goal (degrees)"); ax[1].set_ylabel("predicted xG")
ax[1].set_title(f"xG vs angle (distance fixed at {med_d:.0f})"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/baseline_curves.png", dpi=140, bbox_inches="tight")

# ---- Figure 2: xG surface over the attacking third ----
gx = np.linspace(84, 120, 240); gy = np.linspace(0, 80, 240)
GX, GY = np.meshgrid(gx, gy)
dist = np.sqrt((120-GX)**2 + (40-GY)**2)
p = np.sqrt((120-GX)**2 + (36-GY)**2); q = np.sqrt((120-GX)**2 + (44-GY)**2)
ang = np.degrees(np.arccos(np.clip((p**2+q**2-64)/(2*p*q), -1, 1)))
xg = sig(b0 + bd*dist + ba*ang)

fig, ax = plt.subplots(figsize=(7, 5.2))
im = ax.imshow(xg, extent=[84,120,0,80], origin="lower", cmap="viridis", aspect="equal")
ax.plot([120,120],[36,44], color="red", lw=4)
ax.plot([84,120,120,84,84],[0,0,80,80,0], color="white", lw=1)
plt.colorbar(im, label="predicted xG"); ax.set_title("Baseline xG surface (distance + angle)")
ax.axis("off")
plt.tight_layout(); plt.savefig("outputs/baseline_xg_surface.png", dpi=140, bbox_inches="tight")
print("saved baseline_curves.png and baseline_xg_surface.png")
