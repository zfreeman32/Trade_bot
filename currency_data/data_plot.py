import pandas as pd
import matplotlib.pyplot as plt

# Load your data
csv_file = r'C:\Users\zeb.freeman\Documents\Trade_bot\currency_data\eurusd_hour.csv'
data = pd.read_csv(csv_file)

# Combine 'Date' and 'Time' columns into a 'moment' column
data['moment'] = data['Date'] + ' ' + data['Time']

# Convert the 'moment' column to a datetime format
data['moment'] = pd.to_datetime(data['moment'])

# Convert the data to a Pandas DataFrame
data = pd.DataFrame(data).reset_index(drop=True)

# Slice the DataFrame to include only the first 1000 rows
data_subset = data.loc[3975:5000]

# Plot 'moment' on the x-axis and 'Close' on the y-axis for the first 1000 data points
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(data_subset['moment'], data_subset['Close'], label='Close Price', color='blue')
plt.xlabel('Moment')
plt.ylabel('Close Price')
plt.title('EUR/USD Close Price Over Time (First 1000 Data Points)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()