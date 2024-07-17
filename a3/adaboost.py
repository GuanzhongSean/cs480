import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
def load_data():
    with open("train_test_split.pkl", "br") as fh:
        data = pickle.load(fh)
    train_data = np.array(data[0], dtype=np.float64)
    test_data = np.array(data[1], dtype=np.float64)

    train_x = train_data[:, :23]
    train_y = train_data[:, 23]  # labels are either 0 or 1
    test_x = test_data[:, :23]
    test_y = test_data[:, 23]  # labels are either 0 or 1

    # Convert labels to {-1, 1}
    train_y = 2 * train_y - 1
    test_y = 2 * test_y - 1

    return train_x, train_y, test_x, test_y


def h(x):
    signs = np.sign(x)
    signs[signs == 0] = -1  # 0 -> -1
    return signs


def preprocess_M(train_x, train_y, test_x, test_y, sequential=False):
    median = np.median(train_x, axis=0)
    M_train = train_y[:, np.newaxis] * h(train_x - median)
    M_test = test_y[:, np.newaxis] * h(test_x - median)
    if not sequential:
        M_train /= np.sum(np.abs(M_train), axis=1)[:, np.newaxis]
        M_test /= np.sum(np.abs(M_test), axis=1)[:, np.newaxis]

    return M_train, M_test


def adaboost(M_train, M_test, sequential=False):
    n, d = M_train.shape
    w = np.zeros(d)
    p = np.ones(n)
    max_pass = 300

    train_losses = []
    train_errors = []
    test_errors = []

    pbar = tqdm.trange(max_pass)
    pbar.set_description("Running Adaboost")
    for _ in pbar:
        p /= np.sum(p)  # normalize
        epsilon = np.dot(np.maximum(-M_train, 0).T, p)
        gamma = np.dot(np.maximum(M_train, 0).T, p)
        beta = (np.log(gamma) - np.log(epsilon)) / 2

        if sequential:
            j_t = np.argmax(np.abs(np.sqrt(epsilon) - np.sqrt(gamma)))
            alpha = np.zeros(d)
            alpha[j_t] = 1
        else:  # parallel
            alpha = np.ones(d)

        w += alpha * beta
        p *= np.exp(np.dot(-M_train, alpha * beta))

        # Calculate training loss
        train_loss = np.sum(np.exp(-np.dot(M_train, w)))
        train_losses.append(train_loss)

        # Calculate training error
        train_error = np.mean(np.dot(M_train, w) <= 0)
        train_errors.append(train_error)

        # Calculate test error
        test_error = np.mean(np.dot(M_test, w) <= 0)
        test_errors.append(test_error)

    return train_losses, train_errors, test_errors


def plot_results(train_losses, train_errors, test_errors, dir='plot', filename='ex2q3'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.figure()
    _, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax2.plot(train_losses, label='Training Loss', color='red')
    ax.plot(train_errors, label='Training Error', color='blue')
    ax.plot(test_errors, label='Test Error', color='green')

    plt.xlabel('Iteration')
    ax.set_ylabel('Error')
    ax2.set_ylabel('Loss')

    # Create legends for each axis
    ax_lines, ax_labels = ax.get_legend_handles_labels()
    ax2_lines, ax2_labels = ax2.get_legend_handles_labels()
    lines = ax2_lines + ax_lines
    labels = ax2_labels + ax_labels
    ax2.legend(lines, labels, loc='upper right')

    plt.title('Adaboost Training Loss, Training Error, and Test Error')
    plt.savefig(f'{dir}/{filename}.png')
    plt.close()


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()

    print("Parallel Adaboost:")
    M_train, M_test = preprocess_M(train_x, train_y, test_x, test_y)
    train_losses, train_errors, test_errors = adaboost(M_train, M_test)
    plot_results(train_losses, train_errors, test_errors)

    print("\nSequential Adaboost:")
    M_train, M_test = preprocess_M(train_x, train_y, test_x, test_y, True)
    train_losses, train_errors, test_errors = adaboost(M_train, M_test, True)
    plot_results(train_losses, train_errors, test_errors, filename='ex2q4')
