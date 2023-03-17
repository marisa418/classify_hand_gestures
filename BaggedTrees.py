import pandas as pd
import numpy as np
from numpy import array
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

# Define base estimator (decision tree)
base_estimator = DecisionTreeClassifier(max_depth=6,max_features = None)

# Define bagging classifier
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Train bagging classifier
bagging.fit(X_train, y_train)

# Evaluate bagging classifier
score = bagging.score(X_test, y_test)
print('Accuracy:', score)