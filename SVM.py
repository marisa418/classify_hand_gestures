import warnings
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pywt
from scipy import signal
warnings.filterwarnings("ignore")
# Load data
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
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1,gamma= 'auto')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
