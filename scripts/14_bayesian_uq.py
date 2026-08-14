# Step 14: Bayesian version of the xG model for uncertainty quantification.
# Instead of a single number per coefficient (and per shot), we get a whole
# posterior distribution, which gives credible intervals.
#
# With ~37k shots the posterior is very close to Gaussian (Bernstein-von Mises),
# so here we use the large-sample posterior N(beta_hat, Sigma) for a fast, exact
# demonstration. The provided Stan model reproduces this with full MCMC.

import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.special import expit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/model_features.csv")
X = sm.add_constant(df[["distance", "angle_deg"]])
y = df["goal"]

fit = sm.Logit(y, X).fit(disp=False)
beta_hat = fit.params.values
Sigma = fit.cov_params().values

# draw posterior samples of the coefficients
rng = np.random.default_rng(42)
S = 5000
draws = rng.multivariate_normal(beta_hat, Sigma, size=S)   # (S, 3): const, dist, angle
names = ["intercept", "distance", "angle"]

summary = pd.DataFrame({
    "post_mean": draws.mean(0),
    "cri_2.5%": np.percentile(draws, 2.5, axis=0),
    "cri_97.5%": np.percentile(draws, 97.5, axis=0),
}, index=names).round(4)
print("Posterior coefficient summary (95% credible intervals):")
print(summary.to_string())
summary.to_csv("outputs/bayes_coefficients.csv")

# ---- Figure: coefficient posteriors + xG-vs-distance with credible band ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, (nm, col) in enumerate(zip(["distance", "angle"], ["#1F3864", "#2E7D5B"])):
    ax[0].hist(draws[:, i+1], bins=60, density=True, alpha=.7, color=col, label=nm)
ax[0].axvline(0, color="black", lw=1, ls="--")
ax[0].set_xlabel("coefficient value (log-odds)"); ax[0].set_ylabel("posterior density")
ax[0].set_title("Posterior of the coefficients"); ax[0].legend()

med_angle = df["angle_deg"].median()
dd = np.linspace(0, 40, 200)
# predicted xG for every posterior draw, at the median angle
Xg = np.column_stack([np.ones_like(dd), dd, np.full_like(dd, med_angle)])
P = expit(draws @ Xg.T)                      # (S, 200)
lo, mid, hi = np.percentile(P, [2.5, 50, 97.5], axis=0)
ax[1].plot(dd, mid, color="#1F3864", label="posterior median xG")
ax[1].fill_between(dd, lo, hi, color="#1F3864", alpha=.25, label="95% credible band")
ax[1].set_xlabel("distance to goal"); ax[1].set_ylabel("xG")
ax[1].set_title(f"xG vs distance with uncertainty (angle = {med_angle:.0f}°)")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/bayes_uncertainty.png", dpi=140, bbox_inches="tight")

# ---- example shots: xG with a credible interval ----
print("\nExample shots (xG with 95% credible interval):")
for d, a, label in [(8, 40, "close, central"), (18, 20, "edge of box"), (30, 10, "long range, tight")]:
    xrow = np.array([1.0, d, a])
    ps = expit(draws @ xrow)
    print(f"  {label:20s} d={d:2d}, angle={a:2d}  ->  xG {ps.mean():.3f}  [{np.percentile(ps,2.5):.3f}, {np.percentile(ps,97.5):.3f}]")
print("\nsaved outputs/bayes_coefficients.csv and outputs/bayes_uncertainty.png")
