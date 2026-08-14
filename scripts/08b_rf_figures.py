import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

NAVY, GREEN, ORANGE, GREY = "#1F3864", "#2E7D5B", "#C6612F", "grey"

# feature importance
imp = pd.read_csv("outputs/rf_importances.csv", index_col=0).iloc[:,0].sort_values().tail(12)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set_xlabel("importance (share of predictive power)")
ax.set_title("Random Forest feature importance")
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("outputs/rf_importances.png", dpi=140, bbox_inches="tight")

# calibration: enhanced logit vs random forest
e = pd.read_csv("data/enhanced_test_predictions.csv")
r = pd.read_csv("data/rf_test_predictions.csv")
fig, ax = plt.subplots(figsize=(6.5, 6))
for d, col, lab in [(e, GREEN, "enhanced logit"), (r, ORANGE, "random forest")]:
    fp, mp = calibration_curve(d["y_true"], d["xg_pred"], n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=col, label=lab)
ax.plot([0,1],[0,1], "--", color=GREY, label="perfect")
ax.set_xlabel("predicted xG"); ax.set_ylabel("actual goal rate")
ax.set_title("Calibration: logistic vs random forest"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/rf_calibration.png", dpi=140, bbox_inches="tight")
print("saved rf_importances.png and rf_calibration.png")
