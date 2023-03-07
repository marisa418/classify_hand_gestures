import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from numpy import array
from sklearn.neighbors import NearestNeighbors
import itertools
from itertools import product

# Define the MLkNN classifier
class MyMLkNN(MLkNN):
    
    def __init__(self, k=10, s=1.0):
        super().__init__(k=k, s=s)
        self.knn_ = None
        self.cond_labels_ = None
        self.cond_feature_prob_ = None
        self.predicted_proba_ = None
        self.predicted_ = None
        self.score_ = None
        self.y_test_ = None
        self._label_count = None
        self._feature_count = None
        self.X_train = None
        self.y_train = None
        self.X_train_subs = None

    def fit(self, X, y, x_indices):
        self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        self._label_count = y.shape[1]
        self._feature_count = len(x_indices)
        self.X_train = X
        self.y_train = y
        self.X_train_subs = self.X_train[:, x_indices] 
        self._label_cache = []

        for i in range(self._label_count):
            label_indices = np.where(y[:, i] == 1)[0]
            self._label_cache.append(label_indices)
        self.cond_labels_ = self.generate_conditional_labels(y)
        self.cond_feature_prob_ = self.generate_conditional_feature_probabilities(X, y, self.cond_labels_)
        return self

    def generate_conditional_labels(self, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        label_combinations = list(itertools.product([0, 1], repeat=self._label_count))
        conditional_labels = {}
        for combination in label_combinations:
            combination = np.array(combination).reshape(1, -1)
            conditional_labels[tuple(combination[0])] = []
        for i in range(self._label_count):
            for combination in label_combinations:
                label_indices = np.where(y[:, i] == combination[i])[0]
                conditional_labels[tuple(combination)].append(label_indices)
        return conditional_labels
    
    def generate_conditional_feature_probabilities(self, X, y, cond_labels):
        """
        Calculate the conditional probabilities of each feature given the labels.
        """
        cond_feature_prob = []
        for i in range(self._label_count):
            label_indices = np.where(y[:, i] == 1)[0]
            cond_feature_prob.append([])
            for j in range(self._feature_count):
                feature_values = X[label_indices, j]
                unique_values = np.unique(feature_values)
                value_prob = []
                for value in unique_values:
                    count = np.count_nonzero(feature_values == value)
                    prob = count / len(feature_values)
                    value_prob.append(prob)
                cond_feature_prob[i].append((unique_values, value_prob))
        return cond_feature_prob
    def generate_predicted_probabilities(self, X):
        proba = []
        for x in X:
            x_indices = np.nonzero(x)[0]
            x_subs = X[:, x_indices] 
            distances = np.sqrt(np.sum(((self.X_train_subs - x_subs) ** 2), axis=1))
            knn_indices = np.argsort(distances)[:self.k]
            knn_labels = self.y_train[knn_indices]
            knn_dists = distances[knn_indices]
            knn_labels_all = np.zeros((self.k, self._label_count))
            for i in range(self.k):
                label_indices = np.where(knn_labels[i] == 1)[0]
                knn_labels_all[i, label_indices] = 1
            weights = np.exp(-knn_dists / self.sigma)
            probs = np.sum(knn_labels_all * weights[:, None], axis=0) / np.sum(weights)
            proba.append(probs)
        return np.array(proba)

# Load the dataset
dataset = pd.read_csv('Data.csv')
data = array(dataset)
X = data[:,:-1].astype(np.float32)
y = data[:, -1:].astype(np.int32).reshape(-1, 1)
print("X",data[:,:-1].shape)
print("y",data[:, -1:].shape)
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a MyMLkNN classifier

x_indices = [0]

subspace_knn = MyMLkNN(k=5)
subspace_knn.fit(X_train, y_train, x_indices=x_indices)

subspace_knn.predicted_proba_ = subspace_knn.generate_predicted_probabilities(X_test)
subspace_knn.predicted_ = subspace_knn.predict(X_test)
subspace_knn.score_ = subspace_knn.score(y_test, subspace_knn.predicted_)

# Evaluate the classifier
subspace_knn_score = subspace_knn.score_

print("MyMLkNN accuracy:", subspace_knn_score)
