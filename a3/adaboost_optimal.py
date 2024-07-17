import tqdm
import numpy as np
from adaboost import load_data, adaboost, plot_results


def preprocess_M(X_train, y_train, X_test, y_test, sequential=False):
    _, d = X_train.shape
    M_train = np.zeros(X_train.shape)
    M_test = np.zeros(X_test.shape)
    pbar = tqdm.trange(d)
    pbar.set_description("Preprocessing Weak Learners")
    for j in pbar:
        # Sort indices of the j-th feature
        sorted_X_j = np.unique(np.sort(X_train[:, j]))

        # Calculate midpoints for thresholds
        thresholds = (sorted_X_j[:-1] + sorted_X_j[1:]) / 2
        thresholds = np.concatenate(
            ([sorted_X_j[0] - 1], thresholds, [sorted_X_j[-1] + 1]))

        # Evaluate all possible thresholds and signs
        min_error = np.inf
        best_s = None
        best_b = None
        for s in [+1, -1]:
            predictions = s * (X_train[:, j, np.newaxis] - thresholds)
            error = \
                np.mean(y_train[:, np.newaxis] * predictions <= 0, axis=0)
            t = thresholds[np.argmin(error)]
            error = np.min(error)
            if error < min_error:
                min_error = error
                best_s = s
                best_b = -s*t

        M_train[:, j] = \
            y_train * np.sign(best_s * X_train[:, j] + best_b)
        M_test[:, j] = \
            y_test * np.sign(best_s * X_test[:, j] + best_b)

    if not sequential:
        M_train /= np.sum(np.abs(M_train), axis=1)[:, np.newaxis]
        M_test /= np.sum(np.abs(M_test), axis=1)[:, np.newaxis]

    return M_train, M_test


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()

    print("Parallel Adaboost:")
    M_train, M_test = preprocess_M(train_x, train_y, test_x, test_y)
    train_losses, train_errors, test_errors = adaboost(M_train, M_test)
    plot_results(train_losses, train_errors, test_errors, dir='plot_optimal')

    print("\nSequential Adaboost:")
    M_train, M_test = preprocess_M(train_x, train_y, test_x, test_y, True)
    train_losses, train_errors, test_errors = adaboost(M_train, M_test, True)
    plot_results(train_losses, train_errors, test_errors,
                 dir='plot_optimal', filename='ex2q4')
