import pandas as pd
import numpy as np
from numpy import array
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
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

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5), activation='tanh', solver='adam', alpha=0.0001,learning_rate='constant' )
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
