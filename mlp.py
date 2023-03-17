import pandas as pd
from numpy import array
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
# Load data
dataset = pd.read_csv("Emg012.csv")
data=array(dataset)
X = data[:,:-1]
y = data[:,-1:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='adam', alpha=0.0001,learning_rate='constant', )

# Train MLP classifier
mlp.fit(X_train, y_train)

# Evaluate MLP classifier
score = mlp.score(X_test, y_test)
print('Accuracy:', score)
