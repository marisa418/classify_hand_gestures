import pandas as pd
import numpy as np
from numpy import array
import warnings
import pywt
import time
import pickle
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq , fftshift
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
dataset = pd.read_csv('EMG_All.csv')
data=array(dataset)
X=data[:,:-1]*(0.2)
y = data[:,-1:]
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
Xlist = X.tolist()
ssi = [(abs(Xlist[i][0]))**2 for i in range(len(Xlist))] 
WL = [abs(Xlist[i+1][0]- Xlist[i][0])for i in range(len(Xlist)-1)]
WLmax = WL[(len(WL)-1)]
WL.append(WLmax)
WL = array(WL)
ssi = array(ssi)
WL = WL.reshape(-1,1)
ssi = ssi.reshape(-1,1)
y = y.reshape(-1,1)
data1 = np.concatenate((WL,ssi), axis=1)
x_train, x_test, y_train, y_test = train_test_split(data1,y,test_size=0.2,random_state=101)

knn = KNeighborsClassifier(n_neighbors=100, algorithm="brute")
knn.fit(x_train, y_train)

# Make predictions and evaluate accuracy
y_pred = knn.predict(x_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
