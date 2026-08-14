# Runs the whole project in the correct order. Just run:  python run_all.py
# The first two scripts download data from StatsBomb, so allow a few minutes.
import os, sys, subprocess

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

scripts = [
    "scripts/01_get_fixtures.py",
    "scripts/02_extract_clean.py",
    "scripts/03_geometry.py",
    "scripts/04_baseline_logit.py",
    "scripts/04b_baseline_figures.py",
    "scripts/05_baseline_eval.py",
    "scripts/06_encode_features.py",
    "scripts/06b_category_figures.py",
    "scripts/07_enhanced_logit.py",
    "scripts/07b_enhanced_figures.py",
    "scripts/08_random_forest.py",
    "scripts/08b_rf_figures.py",
    "scripts/09_xgboost.py",
    "scripts/09b_xgb_figures.py",
    "scripts/10_comparison.py",
    "scripts/11_calibration.py",
    "scripts/12_applied_outputs.py",
    "scripts/13_match_simulation.py",
    "scripts/13b_sim_figures.py",
    "scripts/14_bayesian_uq.py",
    "scripts/15_sensitivity.py",
]
for s in scripts:
    print(f"\n===== running {s} =====", flush=True)
    if subprocess.run([sys.executable, s]).returncode != 0:
        print(f"!! {s} failed - stopping."); sys.exit(1)
print("\nAll steps finished. See the data/ and outputs/ folders.")
