# Step 6: turn the context columns into numeric features the model can use.
# Words -> 0/1 columns (one-hot encoding). For each category we drop one level
# as a "reference", so every coefficient later reads as "compared to a typical shot".

import json, numpy as np, pandas as pd

df = pd.read_csv("data/shots_with_features.csv")
cat_cols = ["shot_body_part", "shot_technique", "shot_type", "play_pattern"]

# --- quick EDA: how common is each category and how often does it score? ---
for c in cat_cols:
    g = (df.groupby(c)["goal"].agg(n="size", conv="mean")
           .sort_values("n", ascending=False))
    g["conv"] = (100*g["conv"]).round(1)
    print(f"\n{c}:\n{g.to_string()}")

# --- group very rare levels (<1% of shots) into "Other" to keep things stable ---
def group_rare(s, thresh=0.01):
    freq = s.value_counts(normalize=True)
    rare = freq[freq < thresh].index
    return s.where(~s.isin(rare), "Other")

for c in ["shot_technique", "play_pattern"]:
    df[c] = group_rare(df[c])

# booleans -> 0/1
df["under_pressure"] = df["under_pressure"].astype(int)
df["shot_first_time"] = df["shot_first_time"].astype(int)

# reference category = the most common level of each
refs = {c: df[c].value_counts().idxmax() for c in cat_cols}
print("\nreference categories (dropped):", refs)

# one-hot encode, dropping the reference column
dummies = []
for c in cat_cols:
    d = pd.get_dummies(df[c], prefix=c).astype(int)
    d = d.drop(columns=[f"{c}_{refs[c]}"])
    dummies.append(d)

num = df[["distance", "angle_deg", "under_pressure", "shot_first_time"]]
feat = pd.concat([num] + dummies, axis=1)
FEATURES = list(feat.columns)

# keep id columns for later steps (finishing table, simulation), then features, then goal
ids = df[["match_id", "competition", "team", "player", "x", "y", "shot_statsbomb_xg"]]
out = pd.concat([ids, feat, df["goal"]], axis=1)
out.to_csv("data/model_features.csv", index=False)
json.dump(FEATURES, open("data/feature_list.json", "w"))

print(f"\nsaved data/model_features.csv with {len(FEATURES)} features")
print("features:", FEATURES)
