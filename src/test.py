from src.load import load_metadata, to_number
from src.eda import show

df = load_metadata()
df["budget"] = to_number(df["budget"])      # text -> numbers
df["revenue"] = to_number(df["revenue"])

# 1. Condition: one True/False per row
mask = (df["budget"] > 0) & (df["budget"] < 1000)

# 2. Keep the matching rows, and only the columns you want to look at
small = df.loc[mask, ["id", "title", "release_date", "budget", "revenue"]]

# 3. Sort so the strangest cases come first
small = small.sort_values("revenue", ascending=False)

# 4. Pretty print
print(f"{len(small)} movies with 0 < budget < 1000")
show(small.head(50))