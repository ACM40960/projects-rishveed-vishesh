# Step 15: sensitivity analysis of the xG model.
#  (a) LOCAL: analytic partial derivatives of the logistic model - how much xG
#      changes for a small change in distance or angle, and where on the pitch
#      the model is most sensitive.
#  (b) GLOBAL: Sobol indices - over the whole range of shots, how much of the
#      variation in xG is driven by distance vs angle.

import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.special import expit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

df = pd.read_csv("data/model_features.csv")
X = sm.add_constant(df[["distance", "angle_deg"]])
fit = sm.Logit(df["goal"], X).fit(disp=False)
b0, bd, ba = fit.params["const"], fit.params["distance"], fit.params["angle_deg"]
print("coefficients: intercept %.4f | distance %.4f | angle %.4f" % (b0, bd, ba))

def xg(d, a): return expit(b0 + bd*d + ba*a)

# ---- (a) LOCAL: derivatives dp/dd = bd*p(1-p), dp/da = ba*p(1-p) ----
med_a = df["angle_deg"].median()
dd = np.linspace(2, 40, 200)
p = xg(dd, med_a)
dpdd = bd * p*(1-p)      # change in xG per extra unit of distance
dpda = ba * p*(1-p)      # change in xG per extra degree of angle

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(dd, dpdd, color="#1F3864", label="per unit distance")
ax[0].plot(dd, dpda, color="#2E7D5B", label="per degree angle")
ax[0].axhline(0, color="black", lw=.8)
ax[0].set_xlabel("distance to goal"); ax[0].set_ylabel("change in xG")
ax[0].set_title(f"Local sensitivity of xG (angle = {med_a:.0f}°)"); ax[0].legend(); ax[0].grid(alpha=.3)

# pitch map of overall sensitivity magnitude = p(1-p)*sqrt(bd^2+ba^2)
gx = np.linspace(84,120,240); gy = np.linspace(0,80,240)
GX, GY = np.meshgrid(gx, gy)
dist = np.sqrt((120-GX)**2 + (40-GY)**2)
pp = np.sqrt((120-GX)**2 + (36-GY)**2); qq = np.sqrt((120-GX)**2 + (44-GY)**2)
ang = np.degrees(np.arccos(np.clip((pp**2+qq**2-64)/(2*pp*qq), -1, 1)))
pg = xg(dist, ang)
sens = pg*(1-pg)*np.sqrt(bd**2 + ba**2)
im = ax[1].imshow(sens, extent=[84,120,0,80], origin="lower", cmap="magma", aspect="equal")
ax[1].plot([120,120],[36,44], color="cyan", lw=4); plt.colorbar(im, ax=ax[1], label="sensitivity magnitude")
ax[1].set_title("Where xG is most sensitive to position"); ax[1].axis("off")
plt.tight_layout(); plt.savefig("outputs/sensitivity_local.png", dpi=140, bbox_inches="tight")

# ---- (b) GLOBAL: Sobol indices over realistic shot ranges ----
problem = {"num_vars": 2, "names": ["distance", "angle"], "bounds": [[1, 35], [2, 90]]}
param = sobol_sample.sample(problem, 4096)
Yout = xg(param[:, 0], param[:, 1])
Si = sobol_analyze.analyze(problem, Yout, print_to_console=False)
print("\nSobol indices (share of xG variance explained):")
print("  first-order S1:  distance %.3f | angle %.3f" % (Si["S1"][0], Si["S1"][1]))
print("  total-effect ST: distance %.3f | angle %.3f" % (Si["ST"][0], Si["ST"][1]))

fig, ax = plt.subplots(figsize=(7,5))
x = np.arange(2); w=0.35
ax.bar(x-w/2, Si["S1"], w, label="first-order (S1)", color="#1F3864")
ax.bar(x+w/2, Si["ST"], w, label="total-effect (ST)", color="#2E7D5B")
ax.set_xticks(x); ax.set_xticklabels(["distance","angle"])
ax.set_ylabel("share of xG variance"); ax.set_title("Global sensitivity (Sobol indices)")
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("outputs/sensitivity_sobol.png", dpi=140, bbox_inches="tight")

pd.DataFrame({"feature":["distance","angle"], "S1":Si["S1"], "ST":Si["ST"]}).round(3).to_csv(
    "outputs/sobol_indices.csv", index=False)
print("\nsaved sensitivity_local.png, sensitivity_sobol.png, sobol_indices.csv")
