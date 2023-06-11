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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
# Load dataset
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

data = pd.read_csv('all.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2:]
from sklearn.preprocessing import StandardScaler
sos = signal.iirfilter(90, [30,4500], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
X = signal.sosfilt(sos,X)




print("all7")
X_f = np.hstack((rms(X).reshape(-1, 1),ssi(X).reshape(-1, 1),
                        mav (X).reshape(-1, 1),
                       wl(X).reshape(-1, 1), iemg(X).reshape(-1, 1),acc(X), dwt(X)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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

print("rms+ssi+mav")
X_f = np.hstack((rms(X).reshape(-1, 1),ssi(X).reshape(-1, 1),
                        mav (X).reshape(-1, 1)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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

print("wl+iemg+acc")
X_f = np.hstack((wl(X).reshape(-1, 1), iemg(X).reshape(-1, 1),acc(X)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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

print("ssi+wl+dwt")
X_f = np.hstack((ssi(X).reshape(-1, 1),wl(X).reshape(-1, 1),dwt(X)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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

print("rms+iemg+dwt")
X_f = np.hstack((rms(X).reshape(-1, 1),iemg(X).reshape(-1, 1), dwt(X)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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

print("rms+ssi+mav+wl+iemg")
X_f = np.hstack((rms(X).reshape(-1, 1),ssi(X).reshape(-1, 1),
                        mav (X).reshape(-1, 1),
                       wl(X).reshape(-1, 1), iemg(X).reshape(-1, 1)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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


print("rms+ssi+wl+acc+dwt")
X_f = np.hstack((rms(X).reshape(-1, 1),ssi(X).reshape(-1, 1),
                       wl(X).reshape(-1, 1),acc(X), dwt(X)
                        ))

X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.3, random_state=88)
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