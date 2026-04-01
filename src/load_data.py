import seaborn as sns
import pandas as pd
import os

print("Running script...")

os.makedirs("data", exist_ok=True)
print("Created data folder")

df = sns.load_dataset("penguins")
print("Loaded dataset")

df = df.dropna()
print("Dropped NA")

df['species'] = df['species'].astype('category').cat.codes

df = df.sample(frac=0.8)

df.to_csv("data/data.csv", index=False)
print("Saved CSV")