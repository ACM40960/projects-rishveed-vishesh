# Step 5: evaluate the baseline model on the held-out test shots.
# Three questions: can it rank (AUC), are its probabilities accurate (Brier),
# and are they calibrated (reliability plot)? We also compare to StatsBomb's xG.

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# our baseline test predictions saved in Step 4
pred = pd.read_csv("data/baseline_test_predictions.csv")
y = pred["y_true"].values
phat = pred["xg_pred"].values

# recover the same test rows to grab StatsBomb's own xG as a benchmark
df = pd.read_csv("data/shots_with_features.csv")
X_tr, X_te, y_tr, y_te = train_test_split(
    df[["distance","angle_deg"]], df["goal"], test_size=0.20,
    stratify=df["goal"], random_state=42)
assert (y_te.values == y).all()          # confirms the rows line up
sb_xg = df.loc[X_te.index, "shot_statsbomb_xg"].values

def scores(name, yt, pp):
    m = {"model": name,
         "AUC": roc_auc_score(yt, pp),
         "Brier": brier_score_loss(yt, pp),
         "LogLoss": log_loss(yt, pp)}
    return m

rows = [scores("Baseline logit (dist+angle)", y, phat)]
ok = ~np.isnan(sb_xg)
rows.append(scores("StatsBomb xG (benchmark)", y[ok], sb_xg[ok]))
metrics = pd.DataFrame(rows)

# no-skill reference: predict the base rate for every shot
base = y.mean()
print("test shots:", len(y), "| goal rate: %.3f" % base)
print("no-skill Brier (predict base rate):", round(base*(1-base), 4))
print(metrics.round(4).to_string(index=False))
metrics.to_csv("outputs/model_metrics.csv", index=False)

# ---- Figure: ROC curve + calibration (reliability) plot ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

fpr, tpr, _ = roc_curve(y, phat)
ax[0].plot(fpr, tpr, color="#1F3864", label=f"AUC = {metrics.AUC[0]:.3f}")
ax[0].plot([0,1],[0,1], "--", color="grey")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
ax[0].set_title("ROC curve"); ax[0].legend(); ax[0].grid(alpha=.3)

frac_pos, mean_pred = calibration_curve(y, phat, n_bins=10, strategy="quantile")
ax[1].plot(mean_pred, frac_pos, "o-", color="#1F3864", label="baseline model")
ax[1].plot([0,1],[0,1], "--", color="grey", label="perfect calibration")
ax[1].set_xlabel("predicted xG"); ax[1].set_ylabel("actual goal rate")
ax[1].set_title("Calibration (reliability) plot"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/baseline_evaluation.png", dpi=140, bbox_inches="tight")

# histogram of predicted xG (shows most shots are low value)
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.hist(phat, bins=40, color="#2E7D5B", edgecolor="white")
ax2.set_xlabel("predicted xG"); ax2.set_ylabel("number of shots")
ax2.set_title("Distribution of predicted xG"); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/baseline_xg_hist.png", dpi=140, bbox_inches="tight")
print("saved figures and outputs/model_metrics.csv")
