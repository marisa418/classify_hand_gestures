import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data and preprocess as needed
data = pd.read_csv("Data.csv")
X = data.iloc[:, :-1].values  # convert to numpy array
y = data.iloc[:, -1].values

# Reshape data to fit LSTM model input shape
X = X.reshape(X.shape[0], 1, X.shape[1])  # shape (num_samples, time_steps=1, num_features)

# Normalize the data
X = (X - np.mean(X)) / np.std(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(1, X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Evaluate LSTM model on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy:', accuracy)
