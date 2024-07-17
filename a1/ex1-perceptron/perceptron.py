from itertools import combinations
import numpy as np


def perceptron(X, y, w, b, max_pass):
    mistakes = []
    n, _ = X.shape
    for _ in range(max_pass):
        mistake_count = 0
        for i in range(n):
            if y[i] * (np.dot(X[i], w) + b) <= 0:
                w = w + y[i] * X[i]
                b = b + y[i]
                mistake_count += 1
        mistakes.append(mistake_count)
        # print(f"Pass {t+1}: Mistakes = {mistake_count}")
    return w, b, mistakes


def perceptron_shuffled(X, y, w, b, max_pass):
    mistakes = []
    n, _ = X.shape
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    for _ in range(max_pass):
        mistake_count = 0
        for i in range(n):
            if y_shuffled[i] * (np.dot(X_shuffled[i], w) + b) <= 0:
                w = w + y_shuffled[i] * X_shuffled[i]
                b = b + y_shuffled[i]
                mistake_count += 1
        mistakes.append(mistake_count)
        # print(f"Pass {t+1}: Mistakes = {mistake_count}")
    return w, b, mistakes


def perceptron_one_vs_all(X, y, max_pass):
    classes = np.unique(y)
    classifiers = {}
    for cls in classes:
        y_binary = np.where(y == cls, 1, -1)
        w = np.zeros(X.shape[1])
        b = 0
        w, b, _ = perceptron(X, y_binary, w, b, max_pass)
        classifiers[cls] = (w, b)

    return classifiers


def predict_one_vs_all(X, classifiers):
    classes = list(classifiers.keys())
    scores = np.zeros((X.shape[0], len(classes)))
    for _, cls in enumerate(classifiers):
        w, b = classifiers[cls]
        scores[:, int(cls)] = np.dot(X, w) + b

    return np.argmax(scores, axis=1)


def perceptron_one_vs_one(X, y, max_pass):
    classes = np.unique(y)
    classifiers = {}
    for cls1, cls2 in combinations(classes, 2):
        mask = np.logical_or(y == cls1, y == cls2)
        X_pair = X[mask]
        y_pair = y[mask]
        y_pair_binary = np.where(y_pair == cls1, 1, -1)
        w = np.zeros(X.shape[1])
        b = 0
        w, b, _ = perceptron(X_pair, y_pair_binary, w, b, max_pass)
        classifiers[(cls1, cls2)] = (w, b)

    return classifiers


def predict_one_vs_one(X, classifiers):
    classes = list({cls for pair in classifiers.keys() for cls in pair})
    votes = np.zeros((X.shape[0], len(classes)))
    for (cls1, cls2), (w, b) in classifiers.items():
        scores = np.dot(X, w) + b
        predictions = np.where(scores > 0, cls1, cls2)
        for i, pred in enumerate(predictions):
            votes[i, int(pred)] += 1

    return np.argmax(votes, axis=1)


def perceptron_multiclass(X, y, max_passes):
    classes = np.unique(y)
    n, d = X.shape
    X = np.hstack((X, np.ones((n, 1))))
    # indices = np.random.permutation(n)
    # X = X[indices]
    # y = y[indices]
    weights = np.zeros((len(classes), d + 1))
    for _ in range(max_passes):
        for i in range(X.shape[0]):
            x_i = X[i]
            y_i = y[i]
            scores = np.dot(weights, x_i)
            y_hat = np.argmax(scores)
            if y_hat != y_i:
                weights[int(y_i)] += x_i
                weights[y_hat] -= x_i

    return weights


def predict_multiclass(X, weights):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    scores = np.dot(X, weights.T)
    return np.argmax(scores, axis=1)


def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
