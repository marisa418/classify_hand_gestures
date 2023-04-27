import pandas as pd
import numpy as np
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('emgL.csv')

X = data.iloc[:, :2]
y = data.iloc[:, 2]



ranges_freq = [
    (10,100),(10,200),(10,300),(10,400),(10,500),(10,600),(10,700),(10,800),(10,900),(10,1000),(10,1500),(10,2000),(10,2500),(10,3000),(10,3500),(10,4000),(10,4500),
    (20,100),(20,200),(20,300),(20,400),(20,500),(20,600),(20,700),(20,800),(20,900),(20,1000),(20,1500),(20,2000),(20,2500),(20,3000),(20,3500),(20,4000),(20,4500),
    (30,100),(30,200),(30,300),(30,400),(30,500),(30,600),(30,700),(30,800),(30,900),(30,1000),(30,1500),(30,2000),(30,2500),(30,3000),(30,3500),(30,4000),(30,4500),
    (40,100),(40,200),(40,300),(40,400),(40,500),(40,600),(40,700),(40,800),(40,900),(40,1000),(40,1500),(40,2000),(40,2500),(40,3000),(40,3500),(40,4000),(40,4500),
    (50,100),(50,200),(50,300),(50,400),(50,500),(50,600),(50,700),(50,800),(50,900),(50,1000),(50,1500),(50,2000),(50,2500),(50,3000),(50,3500),(50,4000),(50,4500),
    (60,100),(60,200),(60,300),(60,400),(60,500),(60,600),(60,700),(60,800),(60,900),(60,1000),(60,1500),(60,2000),(60,2500),(60,3000),(60,3500),(60,4000),(60,4500),
    (70,100),(70,200),(70,300),(70,400),(70,500),(70,600),(70,700),(70,800),(70,900),(70,1000),(70,1500),(70,2000),(70,2500),(70,3000),(70,3500),(70,4000),(70,4500),
    (80,100),(80,200),(80,300),(80,400),(80,500),(80,600),(80,700),(80,800),(80,900),(80,1000),(80,1500),(80,2000),(80,2500),(80,3000),(80,3500),(80,4000),(80,4500),]
data =pd.read_csv('emgL.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2:]
f1 = []
f2 = []
f3 = []
f4 = []
for i in ranges_freq:

    print(i)
    sos = signal.iirfilter(90, i, rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
    X1 = signal.sosfilt(sos,X)


    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=88)


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

    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=1,gamma="auto")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy4 = accuracy_score(y_test, y_pred)
    print("Acc SVM ",accuracy4)
    
    print("------------------------------------------------------------------")
    f1.append(accuracy1)
    f2.append(accuracy2)
    f3.append(accuracy3)
    f4.append(accuracy4)

print("f1",f1)
print("f2",f2)
print("f3",f3)
print("f4",f4)