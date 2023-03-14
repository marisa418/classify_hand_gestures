import pandas as pd
import numpy as np
from numpy import array
import warnings
import pywt
import time
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter,filtfilt
from scipy import signal
warnings.filterwarnings("ignore")
dataset = pd.read_csv('emg12.csv')
data=array(dataset)
X = data[:,:-1]
y = data[:,-1:]
X = np.abs(X)/np.max(X)
X=X.reshape(-1)
sos = signal.butter(6,5,'hp',fs=9600,output='sos')
X = signal.sosfilt(sos, X)
sos1 = signal.butter(6,3000,'low',fs=9600,output='sos')
X = signal.sosfilt(sos1,X)
X = X.reshape(-1,1)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
y = y.reshape(-1,1)
Xlist = X.tolist()
ssi = [(abs(Xlist[i][0]))**2 for i in range(len(Xlist))] 
WL = [Xlist[i+1][0]- Xlist[i][0]for i in range(len(Xlist)-1)]
WLmax = WL[(len(WL)-1)]
WL.append(WLmax)
WL = array(WL)
ssi = array(ssi)
WL = WL.reshape(-1,1)
ssi = ssi.reshape(-1,1)
acc = [(1/len(Xlist))*(abs(Xlist[i+1][0]- Xlist[i][0]))for i in range(len(Xlist)-1)]
accmax = acc[(len(acc)-1)]
acc.append(accmax)
acc = array(acc)
acc = acc.reshape(-1,1)
iemg = [abs(Xlist[i][0]) for i in range(len(Xlist))]
iemg = array(iemg)
iemg = iemg.reshape(-1,1)
rms = [np.sqrt((1/len(Xlist))*((Xlist[i][0])**2))for i in range(len(Xlist))]
rms = array(rms)
rms = rms.reshape(-1,1)
mav = [(1/len(Xlist))*abs(Xlist[i][0]) for i in range(len(Xlist))]
mav = array(mav)
mav = mav.reshape(-1,1)
coeff_all = pywt.wavedec(X,'db4',level=8)
cA8,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1 = coeff_all
omp0 = pywt.upcoef('a',cA8.reshape(-1),'db4',level=8)[:len(X)]
omp1 = pywt.upcoef('d',cD1.reshape(-1),'db4',level=1)[:len(X)]
omp2 = pywt.upcoef('d',cD2.reshape(-1),'db4',level=2)[:len(X)]
omp3 = pywt.upcoef('d',cD3.reshape(-1),'db4',level=3)[:len(X)]
omp4 = pywt.upcoef('d',cD4.reshape(-1),'db4',level=4)[:len(X)]
omp5 = pywt.upcoef('d',cD5.reshape(-1),'db4',level=5)[:len(X)]
omp6 = pywt.upcoef('d',cD6.reshape(-1),'db4',level=6)[:len(X)]
omp7 = pywt.upcoef('d',cD7.reshape(-1),'db4',level=7)[:len(X)]
omp8 = pywt.upcoef('d',cD8.reshape(-1),'db4',level=8)[:len(X)]
recon = pywt.waverec(coeff_all, 'db4')
somcoff = omp0 + omp1 + omp2 + omp3 + omp4 + omp5 + omp6 + omp7 + omp8
somcoff = somcoff.reshape(-1,1)
data1 = np.concatenate((X,WL,ssi,acc,mav,rms,iemg,somcoff),axis=1)
x_train, x_test, y_train, y_test = train_test_split(data1,y,test_size=0.2,random_state=101)

#---------------------------------------------------Algorithm Model---------------------------------------------####
m_rf = Pipeline(steps=[('model', RandomForestClassifier())])
parameters = {
                'model__n_estimators':[i for i in range (10,101,10)],
                'model__criterion':['gini','entropy'],
                'model__max_features':['auto','sqrt','log2']
             }

rf_grid = GridSearchCV(m_rf,cv=10,n_jobs=-1,param_grid=parameters,scoring = 'accuracy')
rf_grid.fit(x_train,y_train)
y_pred = rf_grid.predict(x_test)
print(rf_grid.best_params_)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(precision_recall_fscore_support(y_test,y_pred,average='weighted'))

#Support Vector Matchine
m_svm = Pipeline(steps=[('model2', svm.SVC())])
svm_grid = {
               #'model2__C': [0.1,1,10], 
               'model2__gamma': ['scale','auto'],
               #'model2__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
svm_grid_pipeline = GridSearchCV(m_svm,cv=10,n_jobs=-1,param_grid=svm_grid,scoring = 'accuracy')
svm_grid_pipeline.fit(x_train,y_train)
svm_pred = svm_grid_pipeline.predict(x_test)
print(svm_grid_pipeline.best_params_)
print(classification_report(y_test,svm_pred))
print(confusion_matrix(y_test,svm_pred))
print(accuracy_score(y_test,svm_pred))
print(precision_recall_fscore_support(y_test,svm_pred,average='weighted'))

 
m_gnb = Pipeline(steps=[('gnbmodel', GaussianNB())])
gnb_grid = {
               'gnbmodel__var_smoothing': np.logspace(0,-9, num=100)
           }

gnb_grid_pipeline = GridSearchCV(m_gnb,cv=10,n_jobs=-1,param_grid=gnb_grid,scoring = 'accuracy')
gnb_grid_pipeline.fit(x_train,y_train)
gnb_pred = gnb_grid_pipeline.predict(x_test)
print(gnb_grid_pipeline.best_params_)
print(classification_report(y_test,gnb_pred))
print(confusion_matrix(y_test,gnb_pred))
print(accuracy_score(y_test,gnb_pred))
print(precision_recall_fscore_support(y_test,gnb_pred,average='weighted'))


m_knn = Pipeline(steps=[('knnmodel', KNeighborsClassifier())])
knn_grid = {
                'knnmodel__weights': ['uniform', 'distance'],
                'knnmodel__algorithm' :['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knnmodel__n_neighbors' :[i for i in range (1,20,1)],               
           }

knn_grid = GridSearchCV(m_knn,cv=10,n_jobs=-1,param_grid=knn_grid,scoring = 'accuracy')
knn_grid.fit(x_train,y_train)
knn_pred = knn_grid.predict(x_test)
print(knn_grid.best_params_)
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))
print(accuracy_score(y_test,knn_pred))
print(precision_recall_fscore_support(y_test,knn_pred,average='weighted'))


#MLP
m_mlp = Pipeline(steps=[('mlpmodel', MLPClassifier())])
mlp_grid = {
            'mlpmodel__hidden_layer_sizes': [(10,30,10),(20,)],
            'mlpmodel__activation': ['tanh', 'relu'],
            'mlpmodel__solver': ['sgd', 'adam'],
            'mlpmodel__alpha': [0.0001, 0.05],
            'mlpmodel__learning_rate': ['constant','adaptive'],
             }
                                     
mlp_grid_pipeline = GridSearchCV(m_mlp,cv=10,n_jobs=-1,param_grid=mlp_grid,scoring = 'accuracy')
mlp_grid_pipeline.fit(x_train,y_train)
mlp_pred = mlp_grid_pipeline.predict(x_test)
print(mlp_grid_pipeline.best_params_)
print(classification_report(y_test,mlp_pred))
print(confusion_matrix(y_test,mlp_pred))
print(accuracy_score(y_test,mlp_pred))
print(precision_recall_fscore_support(y_test,mlp_pred,average='weighted'))


import pickle

f = open('Emg_Knn.pkl', 'wb')
pickle.dump(knn_grid, f)
f.close()

c = open('Emg_Romdowforest.pkl', 'wb')
pickle.dump(rf_grid, c)
c.close()

