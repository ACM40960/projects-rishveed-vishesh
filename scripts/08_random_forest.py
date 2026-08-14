# Step 8: Random Forest xG model. Same features, same split, same evaluation.
# A forest can capture combinations of features that logistic regression cannot,
# but its probabilities are usually less well calibrated - we check both.

import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

df = pd.read_csv("data/model_features.csv")
FEATURES = json.load(open("data/feature_list.json"))
X, y = df[FEATURES], df["goal"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# light tuning: hold out a validation slice of the training data and try a few
# settings, picking the one with the best (lowest) Brier score on validation.
X_t2, X_val, y_t2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=1)

configs = [
    dict(n_estimators=300, max_depth=8,  min_samples_leaf=50),
    dict(n_estimators=300, max_depth=12, min_samples_leaf=50),
    dict(n_estimators=300, max_depth=None, min_samples_leaf=50),
    dict(n_estimators=300, max_depth=12, min_samples_leaf=20),
]
best, best_brier = None, 1e9
for cfg in configs:
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **cfg).fit(X_t2, y_t2)
    pv = rf.predict_proba(X_val)[:, 1]
    b = brier_score_loss(y_val, pv)
    print("cfg", cfg, "-> val Brier %.4f" % b)
    if b < best_brier:
        best, best_brier = cfg, b
print("chosen:", best)

# refit best config on the full training set
rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best).fit(X_tr, y_tr)
phat = rf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, phat)
brier = brier_score_loss(y_te, phat)
ll = log_loss(y_te, phat)
print("\nRandom Forest -> AUC %.4f | Brier %.4f | LogLoss %.4f" % (auc, brier, ll))

pd.DataFrame({"y_true": y_te.values, "xg_pred": phat}).to_csv(
    "data/rf_test_predictions.csv", index=False)

# feature importance
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.to_csv("outputs/rf_importances.csv")
print("\ntop features:\n", imp.head(8).round(4).to_string())

# append to comparison table
m = pd.read_csv("outputs/model_metrics.csv")
m = m[m["model"] != "Random Forest"]
m = pd.concat([m, pd.DataFrame([{"model":"Random Forest","AUC":auc,"Brier":brier,"LogLoss":ll}])],
              ignore_index=True)
m.to_csv("outputs/model_metrics.csv", index=False)
print("\ncomparison so far:\n", m.round(4).to_string(index=False))
