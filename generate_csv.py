import pandas as pd

df = pd.read_csv("output_times.txt", sep=":", header=None)

df.columns = ['function', 'time_in_ms']

df["function"] = df["function"].map(lambda x: x.lstrip('Function ').rstrip(" time"))
df["time_in_ms"] = df["time_in_ms"].map(lambda x: x.rstrip(" ms"))

df[['time_in_ms']] = df[['time_in_ms']].astype(float)

df = df.sort_values(by='time_in_ms', ascending=False)

df.to_csv("output_times.csv", index=False)

print(df)
