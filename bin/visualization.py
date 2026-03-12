import numpy as np
import math
import matplotlib.pyplot as plt

def distances():
    def cos_distance(w, X, Y):
        norms = np.sqrt(X ** 2 + Y ** 2) + 1e-8  # avoid divide by zero
        dot = w[0] * Y + w[1] * X
        return dot

    def euclid_distance(w, X, Y):
        # w = [w_y, w_x], so compare against (X, Y) as (w_x, w_y)
        return np.sqrt((X - w[1]) ** 2 + (Y - w[0]) ** 2)

    def pi_distance(w, X, Y):
        return (1 + np.tanh(w[0] * Y)) * (1 + np.tanh(w[1] * X))

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    w = np.array([0.6, 0.8])  # [w_y, w_x] to match MATLAB order
    w = w / np.linalg.norm(w)

    Z1 = cos_distance(w, X, Y)
    Z2 = euclid_distance(w, X, Y)
    Z3 = pi_distance(w, X, Y)

    plot_surface_3d(X, Y, Z1, 'cosDistance (cosine similarity)')
    plot_surface_3d(X, Y, Z2, 'euclidDistance')
    plot_surface_3d(X, Y, Z3, 'piDistance')

    plt.show()


def plot_surface_3d(X, Y, Z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

# Standalone execution entry point
if __name__ == "__main__":
    distances()