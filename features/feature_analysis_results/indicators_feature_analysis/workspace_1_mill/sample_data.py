#%%
# File paths
import pandas as pd

# File paths
input_file = "./EURUSD_1min_sampled_indicators.csv"
output_file = "./EURUSD_1min_sampled_with_indicators.csv"
num_lines = 1_000_000  # Number of lines to sample

# Read file with header row included
df = pd.read_csv(input_file, engine="python", header=0)

# Take only the last N rows
df = df.tail(num_lines)

# Display DataFrame
df.head()

#%%
# Save as CSV
df.to_csv(output_file, index=False)
print(f"CSV file saved as {output_file}")
# %%
