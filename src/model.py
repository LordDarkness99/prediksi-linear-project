# src/model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

def predict(X, w):
    return X.dot(w)

def compute_cost(X, y, w):
    m = len(y)
    return (1/(2*m)) * np.sum((predict(X, w) - y)**2)

def gradient_descent(X, y, w_init, alpha=0.01, iterations=1000, verbose=False):
    m = len(y)
    w = w_init.copy().astype(float)
    J_history = []
    for i in range(iterations):
        preds = predict(X, w)
        error = preds - y
        grad = (1/m) * (X.T @ error)
        w -= alpha * grad
        J_history.append(compute_cost(X, y, w))
        if verbose and i % 100 == 0:
            print(f"Iter {i} cost={J_history[-1]:.4e}")
    return w, J_history

def train_sklearn(X_unbiased, y):
    # X_unbiased: without bias column, shape (m, n_features)
    lr = LinearRegression()
    lr.fit(X_unbiased, y)
    return lr

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
