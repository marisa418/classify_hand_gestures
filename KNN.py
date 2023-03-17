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
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
# Load data
dataset = pd.read_csv("Emg012.csv")
data=array(dataset)
X = data[:,:-1]
y = data[:,-1:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object to find the best parameter values
# grid_search = GridSearchCV(knn, param_grid, cv=5)

# # Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)

# # Print the best parameter values
# print("Best parameters: ", grid_search.best_params_)

# Use the best parameter values to create a KNN classifier object and fit it to the training data
knn_best = KNeighborsClassifier(n_neighbors=100, 
                                weights='uniform', 
                                algorithm='brute')
knn_best.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = knn_best.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
