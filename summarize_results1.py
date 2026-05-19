import os, io, json
import pandas as pd
import math


RUNS = "artifacts/runs"
rows = []
for fn in os.listdir(RUNS):
    if fn.endswith('.json'):
        with io.open(os.path.join(RUNS, fn), 'r', encoding='utf-8') as f:
            r = json.load(f)
        rows.append(r)


df = pd.DataFrame(rows)

df = df.sort_values(by=["vocab_size"]).reset_index(drop=True)
print(df)


os.makedirs("artifacts", exist_ok=True)
df.to_csv("artifacts/results.csv", index=False)
print("Saved artifacts/results.csv")