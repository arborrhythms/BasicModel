import numpy as np
import matplotlib.pyplot as plt

# Global variable for subplot indexing
subplot_index = 0

def plot_things(things=None):
    global subplot_index

    if things is None:
        plt.figure(figsize=(6, 10))
        subplot_index = 0
        return

    subplot_index += 1
    plt.subplot(4, 1, subplot_index)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    if things.ndim == 1 or things.shape[1] == 1:
        plt.plot(things)
    else:
        for vec in things:
            plt.plot([0, vec[0]], [0, vec[1]])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])


def metrics():
    """
    Explore several distance metrics in perceptual and conceptual spaces
    """
    plot_things()  # Initialize figure
    dim = 2
    k = 5

    # === Physical Space ===
    x0 = np.zeros((k, 2))  # Not used further
    objects = 10 * np.random.randn(k, 2)
    plot_things(objects)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    # === Perceptual Space ===
    percepts = objects.copy()
    percepts -= np.min(percepts)
    percepts /= np.max(np.linalg.norm(percepts, axis=1))
    plot_things(percepts)

    # === Conceptual Space ===
    nC = 42
    knowing = 2 * np.random.rand(nC, 2) - 1
    knowing /= np.linalg.norm(knowing, axis=1, keepdims=True)
    activation = percepts @ knowing.T
    activation = activation.sum(axis=0) / k
    activation -= np.mean(activation)
    concepts = (activation[:, np.newaxis] * knowing)
    plot_things(concepts)

    # === Symbolic Space ===
    symbols = np.sign(activation[:, np.newaxis])
    plot_things(symbols)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    metrics()