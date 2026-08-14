import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

NAVY, GREEN, GREY = "#1F3864", "#2E7D5B", "grey"

# ---- odds-ratio plot (drop the tiny-sample Corner and the intercept) ----
c = pd.read_csv("outputs/enhanced_coefficients.csv", index_col=0)
c = c.drop(index=["const", "shot_type_Corner"])
c = c.sort_values("odds_ratio")
colors = [GREEN if v > 1 else NAVY for v in c["odds_ratio"]]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(c.index, c["odds_ratio"], color=colors)
ax.axvline(1, color="black", lw=1)          # OR = 1 means no effect
ax.set_xscale("log")
ax.set_xlabel("odds ratio (log scale)  —  right of 1 helps, left of 1 hurts")
ax.set_title("Enhanced model: effect of each feature on scoring odds")
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("outputs/enhanced_odds_ratios.png", dpi=140, bbox_inches="tight")

# ---- calibration: baseline vs enhanced ----
b = pd.read_csv("data/baseline_test_predictions.csv")
e = pd.read_csv("data/enhanced_test_predictions.csv")
fig, ax = plt.subplots(figsize=(6.5, 6))
for d, col, lab in [(b, NAVY, "baseline"), (e, GREEN, "enhanced")]:
    fp, mp = calibration_curve(d["y_true"], d["xg_pred"], n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=col, label=lab)
ax.plot([0,1],[0,1], "--", color=GREY, label="perfect")
ax.set_xlabel("predicted xG"); ax.set_ylabel("actual goal rate")
ax.set_title("Calibration: baseline vs enhanced"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/enhanced_calibration.png", dpi=140, bbox_inches="tight")
print("saved enhanced_odds_ratios.png and enhanced_calibration.png")
