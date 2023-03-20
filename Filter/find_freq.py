from numpy import array
import pandas as pd
from scipy import signal
from sklearn.metrics import mean_squared_error
from scipy import signal
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

dataset = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Data/Data123/data123.csv')
data=array(dataset)
X = data[:,:-1]
y = data[:,-1]

Rm = []
ACC=[]


ranges_freq = [(55,100),(55,300),(55,500),(55,700),(55,900),(5,1000),(5,1500),(5,2000),(5,2500),(5,3000),(5,3500),(5,4000),(5,4500),(5,3000)]

X = X.reshape(-1)
for i in ranges_freq:
    sos = signal.iirfilter(90, i, rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
    X_filtered = signal.sosfilt(sos,X)
    X_filtered = X_filtered.reshape(-1,1)

# Calculate the RMSE between the original data and the filtered data
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=88)
    knn = KNeighborsClassifier()
    knn_best = KNeighborsClassifier(n_neighbors=10, 
                                weights='uniform', 
                                algorithm='brute')
    knn_best.fit(X_train, y_train)
    y_pred = knn_best.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(i)
    print("Accuracy:", score)