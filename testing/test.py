import pandas as pd

df = pd.read_csv("predictions.csv")

print(df)
df2 = pd.DataFrame()

df2["id"] = df["id"]
df2["prediction"] = df["prediction"]
df2.to_csv('predictions.csv', index=False)