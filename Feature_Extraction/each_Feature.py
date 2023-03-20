import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from numpy import array
import pywt
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import warnings
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
def rms(X):
    return np.sqrt(np.mean(np.square(X), axis=1))

def ssi(X):
    return np.sum(np.square(X), axis=1)

def mav(X):
    return np.mean(np.abs(X), axis=1)

def iemg(X):
    return np.sum(np.abs(X), axis=1)

def wl(X):
    return np.sum(np.abs(np.diff(X)), axis=1)

def acc(X):
    n_samples, n_features = X.shape
    aac = np.zeros(n_samples)
    for i in range(n_samples):
        aac[i] = np.mean(np.abs(np.diff(X[i, :-1])))
    return np.hstack((aac.reshape(-1, 1), X))

def dwt(X):
    dwt_coeffs = pywt.wavedec(X, 'db4', level=9, axis=1)
    dwt_coeffs = np.concatenate(dwt_coeffs, axis=1)
    return np.hstack((dwt_coeffs, X))
data = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Data/Data123/data123.csv')
data=array(data)
X = data[:,:-1]
y = data[:,-1:]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(-1,1)
knn = KNeighborsClassifier()
# RMS
print('RMS:')
X1 = rms(X).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)


# SSI
print('SSI:')
X1 = ssi(X).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)

# Mav
print('MAV:')
X1 = mav(X).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)

# IEMG
print('IEMG:')
X1 = iemg(X).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)


#WL
print('WL:')
X1 = wl(X).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)
#ACC
print('ACC:')
X1 = acc(X)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)

# DWT
print('DWT:')
X1 = dwt(X)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma="auto")
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('Accuracy:', score)