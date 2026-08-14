# Step 11: try to improve XGBoost's probability calibration.
# Two standard methods:
#   Platt scaling  -> fit a logistic curve that re-maps the predictions
#   Isotonic       -> fit a flexible non-decreasing step function
# Both are fit on a separate calibration slice, then judged on the test set.

import json, numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

df = pd.read_csv("data/model_features.csv")
FEATURES = json.load(open("data/feature_list.json"))
X, y = df[FEATURES], df["goal"]

# same test set as always
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
# split training data into a model-fitting part and a calibration part
X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=7)

# fit XGBoost (same settings chosen in Step 9) on the fitting part only
base = XGBClassifier(n_estimators=115, max_depth=4, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                     eval_metric="logloss", random_state=42, n_jobs=-1)
base.fit(X_fit, y_fit, verbose=False)

clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
p_cal  = base.predict_proba(X_cal)[:, 1]
p_test = base.predict_proba(X_te)[:, 1]

# Platt scaling: logistic regression on the log-odds of the predictions
platt = LogisticRegression().fit(logit(clip(p_cal)).reshape(-1, 1), y_cal)
p_platt = platt.predict_proba(logit(clip(p_test)).reshape(-1, 1))[:, 1]

# Isotonic regression: monotonic map from predicted prob to observed rate
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
p_iso = iso.predict(p_test)

def ece(yt, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -1e-9, 1+1e-9
    idx = np.digitize(p, edges) - 1
    return sum((idx==b).mean()*abs(p[idx==b].mean()-yt[idx==b].mean()) for b in range(bins) if (idx==b).sum())

rows = []
for name, p in [("XGBoost (uncalibrated)", p_test),
                ("XGBoost + Platt", p_platt),
                ("XGBoost + Isotonic", p_iso)]:
    rows.append([name, roc_auc_score(y_te, p), brier_score_loss(y_te, p), ece(y_te.values, p)])
tab = pd.DataFrame(rows, columns=["model","AUC","Brier","ECE"]).round(4)
print(tab.to_string(index=False))
tab.to_csv("outputs/calibration_correction.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 6))
for name, p, col in [("uncalibrated", p_test, "#6A4C93"),
                     ("+ Platt", p_platt, "#2E7D5B"),
                     ("+ Isotonic", p_iso, "#C6612F")]:
    fp, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=col, label=name, ms=4)
ax.plot([0,1],[0,1],"--",color="grey")
ax.set_xlim(0,0.4); ax.set_ylim(0,0.4)
ax.set_xlabel("predicted xG"); ax.set_ylabel("actual goal rate")
ax.set_title("Calibration correction (XGBoost)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/calibration_correction.png", dpi=140, bbox_inches="tight")
print("saved outputs/calibration_correction.csv and .png")
