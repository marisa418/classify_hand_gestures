import pandas as pd
import numpy as np
from numpy import array
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
# Load data and preprocess as needed

data = pd.read_csv("emg.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

# sos = signal.iirfilter(91, [5, 100], rs=150, btype='band',
#                        analog=False, ftype='cheby2', fs=9600,
#                        output='sos')
# X = signal.sosfilt(sos, X, axis=0)
X_rms = np.sqrt(np.mean(np.square(X), axis=1))
X1 = pd.concat([X, pd.DataFrame(X_rms, columns=["RMS"])], axis=1)
# Calculate the absolute differences between adjacent samples in each channel

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=42)

# Define base estimator (decision tree)
base_estimator = DecisionTreeClassifier(max_depth=6,max_features = None)

# Define bagging classifier
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Train bagging classifier
bagging.fit(X_train, y_train)

# Evaluate bagging classifier
score = bagging.score(X_test, y_test)
print('Accuracy:', score)