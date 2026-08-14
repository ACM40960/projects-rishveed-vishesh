# Step 4: baseline logistic regression using only distance and angle.
# We fit the model, read off what the coefficients mean, and save the
# test-set predictions so Step 5 can evaluate the model fairly.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/shots_with_features.csv")

# predictors (distance, angle) and target (goal = 1/0)
X = df[["distance", "angle_deg"]]
y = df["goal"]

# hold out 20% of shots to test on later; stratify so both parts have the
# same goal rate. random_state fixes the split so results are reproducible.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# statsmodels needs an explicit intercept column
X_tr_c = sm.add_constant(X_tr)
X_te_c = sm.add_constant(X_te)

model = sm.Logit(y_tr, X_tr_c).fit(disp=False, maxiter=100)
print(model.summary())

# coefficients as odds ratios, with 95% confidence intervals
coef = model.params
ci = model.conf_int()
table = pd.DataFrame({
    "coef (log-odds)": coef,
    "odds_ratio": np.exp(coef),
    "OR_low": np.exp(ci[0]),
    "OR_high": np.exp(ci[1]),
    "p_value": model.pvalues,
})
print("\n", table.round(4).to_string())
table.to_csv("outputs/baseline_coefficients.csv")

# save test-set predictions for Step 5
pred = pd.DataFrame({"y_true": y_te.values, "xg_pred": model.predict(X_te_c).values})
pred.to_csv("data/baseline_test_predictions.csv", index=False)
print("\nsaved outputs/baseline_coefficients.csv and data/baseline_test_predictions.csv")

# quick look: what xG does the model give for a typical central chance?
example = sm.add_constant(pd.DataFrame({"distance":[11.0], "angle_deg":[38.0]}), has_constant="add")
print("example central shot (11 units, 38 deg) -> xG = %.3f" % model.predict(example)[0])
