# Step 9: XGBoost (gradient-boosted trees). Trees are built one after another,
# each correcting the errors of the last. Same features, split, and evaluation.

import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

df = pd.read_csv("data/model_features.csv")
FEATURES = json.load(open("data/feature_list.json"))
X, y = df[FEATURES], df["goal"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
# validation slice for early stopping / light tuning
X_t2, X_val, y_t2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=1)

grid = [dict(max_depth=d, learning_rate=lr)
        for d in [3, 4, 6] for lr in [0.05, 0.1]]

best, best_ll, best_it = None, 1e9, None
for g in grid:
    m = XGBClassifier(n_estimators=600, subsample=0.9, colsample_bytree=0.9,
                      min_child_weight=5, eval_metric="logloss",
                      early_stopping_rounds=30, random_state=42, n_jobs=-1, **g)
    m.fit(X_t2, y_t2, eval_set=[(X_val, y_val)], verbose=False)
    pv = m.predict_proba(X_val)[:, 1]
    ll = log_loss(y_val, pv)
    print(g, "-> best_iter", m.best_iteration, "| val LogLoss %.4f" % ll)
    if ll < best_ll:
        best, best_ll, best_it = g, ll, m.best_iteration
print("chosen:", best, "| trees:", best_it + 1)

# refit on the full training set with the chosen settings
final = XGBClassifier(n_estimators=best_it + 1, subsample=0.9, colsample_bytree=0.9,
                      min_child_weight=5, eval_metric="logloss",
                      random_state=42, n_jobs=-1, **best)
final.fit(X_tr, y_tr, verbose=False)

phat = final.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, phat)
brier = brier_score_loss(y_te, phat)
ll = log_loss(y_te, phat)
print("\nXGBoost -> AUC %.4f | Brier %.4f | LogLoss %.4f" % (auc, brier, ll))

pd.DataFrame({"y_true": y_te.values, "xg_pred": phat}).to_csv(
    "data/xgb_test_predictions.csv", index=False)
pd.Series(final.feature_importances_, index=FEATURES).sort_values(
    ascending=False).to_csv("outputs/xgb_importances.csv")

m = pd.read_csv("outputs/model_metrics.csv")
m = m[m["model"] != "XGBoost"]
m = pd.concat([m, pd.DataFrame([{"model":"XGBoost","AUC":auc,"Brier":brier,"LogLoss":ll}])],
              ignore_index=True)
m.to_csv("outputs/model_metrics.csv", index=False)
print("\ncomparison so far:\n", m.round(4).to_string(index=False))
