import pandas as pd
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:/Users/maris/OneDrive/Desktop/All โปรเจคจบ/PjHaRo/hello/Position/fist2.csv')
data.dropna(inplace=True)

X = data.iloc[:, :2]
y = data.iloc[:, 2]

ranges_freq = [(55,100),(55,300),(55,500),(55,700),(55,900),(5,1000),(5,1500),(5,2000),(5,2500),(5,3000),(5,3500),(5,4000),(5,4500),(5,3000)]

X = X.values.reshape(-1)
#for i in ranges_freq:
sos = signal.iirfilter(90, [50,300], rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
X_filtered = signal.sosfilt(sos,X)
    
print(X_filtered)
X_filtered = X_filtered.reshape(-1,1)
print(X_filtered)
    # X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=88)
    # knn = KNeighborsClassifier()
    # knn_best = KNeighborsClassifier(n_neighbors=10, 
    #                             weights='uniform', 
    #                             algorithm='brute')
    # knn_best.fit(X_train, y_train)
    # y_pred = knn_best.predict(X_test)
    # score = accuracy_score(y_test, y_pred)
    # print(i)
    # print("Accuracy:", score)

