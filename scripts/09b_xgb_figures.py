import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

NAVY, GREEN, ORANGE, PURPLE, GREY = "#1F3864", "#2E7D5B", "#C6612F", "#6A4C93", "grey"

imp = pd.read_csv("outputs/xgb_importances.csv", index_col=0).iloc[:,0].sort_values().tail(12)
fig, ax = plt.subplots(figsize=(8.5,6))
ax.barh(imp.index, imp.values, color=PURPLE)
ax.set_xlabel("importance (gain)"); ax.set_title("XGBoost feature importance")
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("outputs/xgb_importances.png", dpi=140, bbox_inches="tight")

files = [("data/enhanced_test_predictions.csv", GREEN, "enhanced logit"),
         ("data/rf_test_predictions.csv", ORANGE, "random forest"),
         ("data/xgb_test_predictions.csv", PURPLE, "xgboost")]
fig, ax = plt.subplots(figsize=(6.5,6))
for f, col, lab in files:
    d = pd.read_csv(f)
    fp, mp = calibration_curve(d["y_true"], d["xg_pred"], n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=col, label=lab)
ax.plot([0,1],[0,1],"--",color=GREY,label="perfect")
ax.set_xlabel("predicted xG"); ax.set_ylabel("actual goal rate")
ax.set_title("Calibration: all advanced models"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/all_calibration.png", dpi=140, bbox_inches="tight")
print("saved xgb_importances.png and all_calibration.png")
