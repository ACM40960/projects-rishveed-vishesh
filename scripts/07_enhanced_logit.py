# Step 7: enhanced logistic regression using distance, angle, and all the
# context features from Step 6. Same recipe as the baseline so the comparison
# is fair: same train/test split, same evaluation.

import json, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

df = pd.read_csv("data/model_features.csv")
FEATURES = json.load(open("data/feature_list.json"))

X = df[FEATURES]
y = df["goal"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

X_tr_c = sm.add_constant(X_tr)
X_te_c = sm.add_constant(X_te)
model = sm.Logit(y_tr, X_tr_c).fit(disp=False, maxiter=200)

# coefficients as odds ratios
coef = model.params
tab = pd.DataFrame({
    "coef": coef,
    "odds_ratio": np.exp(coef),
    "p_value": model.pvalues,
}).round(4)
print(tab.to_string())
tab.to_csv("outputs/enhanced_coefficients.csv")

# evaluate on the held-out test set
phat = model.predict(X_te_c).values
auc = roc_auc_score(y_te, phat)
brier = brier_score_loss(y_te, phat)
ll = log_loss(y_te, phat)
print("\nEnhanced logit -> AUC %.4f | Brier %.4f | LogLoss %.4f" % (auc, brier, ll))

# save predictions and append to the running comparison table
pd.DataFrame({"y_true": y_te.values, "xg_pred": phat}).to_csv(
    "data/enhanced_test_predictions.csv", index=False)

metrics = pd.read_csv("outputs/model_metrics.csv")
metrics = metrics[metrics["model"] != "Enhanced logit (context)"]
metrics = pd.concat([metrics, pd.DataFrame([{
    "model": "Enhanced logit (context)", "AUC": auc, "Brier": brier, "LogLoss": ll}])],
    ignore_index=True)
metrics.to_csv("outputs/model_metrics.csv", index=False)
print("\nrunning comparison table:")
print(metrics.round(4).to_string(index=False))
