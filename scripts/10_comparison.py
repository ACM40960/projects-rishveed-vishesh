# Step 10: consolidate every model into one comparison of AUC, Brier, LogLoss,
# and calibration (ECE = expected calibration error, lower = better calibrated).

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

def ece(y, p, bins=10):
    # weighted average gap between predicted prob and actual rate across bins
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.digitize(p, edges) - 1
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

# recover StatsBomb xG on the same test rows
df = pd.read_csv("data/model_features.csv")
X_tr, X_te, y_tr, y_te = train_test_split(
    df.index, df["goal"], test_size=0.20, stratify=df["goal"], random_state=42)
sb = df.loc[X_te, ["shot_statsbomb_xg", "goal"]].dropna()

preds = {
    "Baseline logit (dist+angle)": "data/baseline_test_predictions.csv",
    "Enhanced logit (context)":    "data/enhanced_test_predictions.csv",
    "Random Forest":               "data/rf_test_predictions.csv",
    "XGBoost":                     "data/xgb_test_predictions.csv",
}
rows = []
store = {}
for name, f in preds.items():
    d = pd.read_csv(f); yt = d["y_true"].values; pp = d["xg_pred"].values
    store[name] = (yt, pp)
    rows.append([name, roc_auc_score(yt,pp), brier_score_loss(yt,pp), log_loss(yt,pp), ece(yt,pp)])
# StatsBomb benchmark
yt, pp = sb["goal"].values, sb["shot_statsbomb_xg"].values
store["StatsBomb xG (benchmark)"] = (yt, pp)
rows.append(["StatsBomb xG (benchmark)", roc_auc_score(yt,pp), brier_score_loss(yt,pp), log_loss(yt,pp), ece(yt,pp)])

table = pd.DataFrame(rows, columns=["model","AUC","Brier","LogLoss","ECE"]).round(4)
table.to_csv("outputs/model_comparison.csv", index=False)
print(table.to_string(index=False))

# ---- summary figure: AUC bars, Brier bars, calibration overlay ----
order = list(preds.keys()) + ["StatsBomb xG (benchmark)"]
colors = ["#1F3864","#2E7D5B","#C6612F","#6A4C93","#999999"]
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

t = table.set_index("model").loc[order]
ax[0].barh(order, t["AUC"], color=colors); ax[0].set_xlim(0.7, 0.83)
ax[0].set_title("AUC (higher is better)"); ax[0].invert_yaxis(); ax[0].grid(alpha=.3, axis="x")
ax[1].barh(order, t["Brier"], color=colors); ax[1].set_xlim(0.070, 0.081)
ax[1].set_title("Brier (lower is better)"); ax[1].invert_yaxis(); ax[1].set_yticklabels([]); ax[1].grid(alpha=.3, axis="x")

for name, col in zip(order, colors):
    yt, pp = store[name]
    fp, mp = calibration_curve(yt, pp, n_bins=10, strategy="quantile")
    ax[2].plot(mp, fp, "o-", color=col, label=name, ms=4)
ax[2].plot([0,1],[0,1],"--",color="grey")
ax[2].set_xlim(0,0.4); ax[2].set_ylim(0,0.4)
ax[2].set_xlabel("predicted xG"); ax[2].set_ylabel("actual goal rate")
ax[2].set_title("Calibration"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=140, bbox_inches="tight")
print("\nsaved outputs/model_comparison.csv and outputs/model_comparison.png")
