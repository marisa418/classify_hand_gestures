import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def acc(X):
    n_samples, n_features = X.shape
    aac = np.zeros(n_samples)
    for i in range(n_samples):
        aac[i] = np.mean(np.abs(np.diff(X[i, :-1])))
    return np.hstack((aac.reshape(-1, 1), X))
# Load data
data =pd.read_csv('emgL.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2:]

sos = signal.iirfilter(90, [60,4500], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
X = signal.sosfilt(sos,X)
X1 = acc(X)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

base_estimator = DecisionTreeClassifier(max_depth=4,max_features = None)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=88)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print("Accuracy: ", accuracy)