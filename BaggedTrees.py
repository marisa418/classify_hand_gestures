import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data.csv')

X = data.iloc[:, :-1] # Select all columns except the last one
y = data.iloc[:, -1]  # Select the last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base estimator (decision tree)
base_estimator = DecisionTreeClassifier(max_depth=3)

# Define bagging classifier
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Train bagging classifier
bagging.fit(X_train, y_train)

# Evaluate bagging classifier
score = bagging.score(X_test, y_test)
print('Accuracy:', score)
