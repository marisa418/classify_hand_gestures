import numpy as np
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

# Load the dataset
dataset = pd.read_csv('Data.csv')
data = array(dataset)

# Split the dataset into input features (X) and labels (y)
X = data[:,:-1].astype(np.float32)
y = data[:,-1:].astype(np.int32).ravel()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a SVM classifier on the filtered signal
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Evaluate the accuracy of the classifier on the test set
accuracy = svm.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)
