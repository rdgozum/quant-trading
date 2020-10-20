import math
import numpy as np
import matplotlib.pyplot as plt


def plot(X_test, X_test_pred, n=None):
    if n is None:
        n = math.ceil(X_test.shape[0] / X_test.shape[1])
    plt.figure()

    for i, idx in enumerate(list(np.arange(0, X_test.shape[0], X_test.shape[1]))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(X_test[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(X_test_pred[idx])
        ax.get_xaxis().set_visible(False)

        if i == n - 1:
            break

    plt.show()
