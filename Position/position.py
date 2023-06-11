import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from numpy import array
import pywt
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("posi number4")

data = pd.read_csv('hello/Position/p1.csv')

X = data.iloc[:, :2]
y = data.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

knn = KNeighborsClassifier()
knn_best = KNeighborsClassifier(n_neighbors=10, 
                            weights='uniform', 
                            algorithm='brute')
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
print("Acc KNN ",accuracy1)


base_estimator = DecisionTreeClassifier(max_depth=4,max_features = None)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=88)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred)
print("Acc Bag ",accuracy2)


mlp = MLPClassifier(hidden_layer_sizes=(5), activation='tanh', solver='adam', alpha=0.0001,learning_rate='constant' )
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred)
print("Acc MLP ",accuracy3)


svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy4 = accuracy_score(y_test, y_pred)
print("Acc SVM ",accuracy4)