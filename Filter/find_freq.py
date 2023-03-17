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
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Emg012.csv')
data=array(dataset)
X = data[:,:-1]
y = data[:,-1]


ranges_freq = [(50,100),(50,200),(50,300),(50,400),(50,500),(50,600),(50,700),(50,800),(50,900),(50,1000),(50,1500),(50,2000),(50,2500),(50,3000),(50,3500),(50,4000),(50,4500)]

X = X.reshape(-1)
for i in ranges_freq:
    sos = signal.iirfilter(91, i, rs=150, btype='band',
                       analog=False, ftype='cheby2', fs=9600,
                       output='sos')
    X_filtered = signal.sosfilt(sos,X)
    X_filtered = X_filtered.reshape(-1,1)

# Calculate the RMSE between the original data and the filtered data
    # rmse = mean_squared_error(X, X_filtered, squared=False)
    print("Freq: ",i)
    # print(f"RMSE: {rmse:.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)


    base_estimator = DecisionTreeClassifier(max_depth=6,max_features = None)

# Define bagging classifier
    bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Train bagging classifier
    bagging.fit(X_train, y_train)

# Evaluate bagging classifier
    score = bagging.score(X_test, y_test)
    print('Accuracy:', score)
    