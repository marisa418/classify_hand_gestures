import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pywt
# Load data
dataset = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Data/Data123/data123.csv')
data=array(dataset)
X = data[:,:-1]
y = data[:,-1:]
scaler = StandardScaler()
X = scaler.fit_transform(X)
dwt_coeffs = pywt.wavedec(X, 'db4', level=9, axis=1)
dwt_coeffs = np.concatenate(dwt_coeffs, axis=1)
X= np.hstack((dwt_coeffs, X))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
