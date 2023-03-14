import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("EMG_All.csv")

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Plot first signal
plt.plot(X)
plt.xlabel("Time")
plt.ylabel("Signal Amplitude")
plt.show()