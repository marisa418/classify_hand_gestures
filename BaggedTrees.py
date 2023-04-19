import joblib
import pandas as pd
import numpy as np
from numpy import array
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pywt
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=6,max_features = None)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=88)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)
joblib.dump(bagging, 'bagging_model.joblib')