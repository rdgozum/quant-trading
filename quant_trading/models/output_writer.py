import math
import numpy as np
import matplotlib.pyplot as plt

from quant_trading import settings


def plot_autoencoder(X, X_pred, n, filename):
    if n is None:
        n = math.ceil(X.shape[0] / X.shape[1])
    plt.figure()

    for i, idx in enumerate(list(np.arange(0, X.shape[0], X.shape[1]))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Original", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(X[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Reconstruction", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(X_pred[idx])
        ax.get_xaxis().set_visible(False)

        if i == n - 1:
            break

    path = settings.results(filename)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
