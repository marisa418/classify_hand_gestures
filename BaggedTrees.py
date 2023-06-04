import pandas as pd
import numpy as np
import pywt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def dwt(X):
    dwt_coeffs = pywt.wavedec(X, 'db4', level=9, axis=1)
    dwt_coeffs = np.concatenate(dwt_coeffs, axis=1)
    return np.hstack((dwt_coeffs, X))
# Load data
data =pd.read_csv('Emg_Signal.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2:]

sos = signal.iirfilter(90, [30,4500], rs=150, btype='band',
                        analog=False, ftype='cheby2', fs=9600,
                        output='sos')
X = signal.sosfilt(sos,X)
X1 = dwt(X)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

base_estimator = DecisionTreeClassifier(max_depth=6,max_features = 0.8)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=15, random_state=88)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

