import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("EMG400.csv")

# Split data into training and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Subspace KNN
knn = KNeighborsClassifier(n_neighbors=100, algorithm="brute")
knn.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = knn.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
