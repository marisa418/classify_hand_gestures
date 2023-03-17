import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix
# Load data
data = pd.read_csv("emg12.csv")

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Scale features using MinMaxScaler
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# scaler = RobustScaler()
# X_scaled = scaler.fit_transform(X)

# qt = QuantileTransformer(output_distribution='uniform')
# X_scaled = qt.fit_transform(X)

# pt = PowerTransformer(method='yeo-johnson')
# X_scaled = pt.fit_transform(X)

# X_scaled = preprocessing.normalize(X, norm='l2')


