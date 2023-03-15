import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy import signal
from imblearn.over_sampling import SMOTE
from numpy import array
import warnings
warnings.filterwarnings("ignore")
# Load data
dataset = pd.read_csv("EmgCopy.csv")
data=array(dataset)
X = data[:,:-1]
y = data[:,-1:]
# X1 = X1.reshape(-1)
# sos = signal.iirfilter(88, [50, 400], rs=150, btype='band',
#                        analog=False, ftype='cheby2', fs=9600,
#                        output='sos')
# X = signal.sosfilt(sos,X1)
# X = X.reshape(-1,1)

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Subspace KNN
knn = KNeighborsClassifier(n_neighbors=100, algorithm="brute")
knn.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = knn.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
