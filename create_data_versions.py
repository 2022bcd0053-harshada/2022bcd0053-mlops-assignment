import pandas as pd

df = pd.read_csv("data/data.csv")

# Version 1 (partial)
df.sample(frac=0.5).to_csv("data/data_v1.csv", index=False)

# Version 2 (full copy)
df.to_csv("data/data_v2.csv", index=False)