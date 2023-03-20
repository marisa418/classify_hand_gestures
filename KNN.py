import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from numpy import array
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

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
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=88)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
knn_best = KNeighborsClassifier(n_neighbors=10, 
                                weights='uniform', 
                                algorithm='brute')
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

