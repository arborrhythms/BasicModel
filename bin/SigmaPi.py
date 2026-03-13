"""Minimal experiments for logical functions built from Sigma/Pi layers."""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Model import Layer, SigmaLayer, ReversibleSigmaLayer, PiLayer , ReversiblePiLayer, VQLayer # Import custom layers from Model.py

# This numerically hides the XOR in a set of random vectors,
# which must be identified before the XOR problem can be solved.
embeddingDim  = 1

# Define the neural network
class LogicalFunctionNet(Layer):
    """Two-layer logical network using a Pi hidden layer and Sigma output layer."""
    def __init__(self, nInput, nHidden, nOutput):
        super(LogicalFunctionNet, self).__init__(nInput, nOutput)
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.hidden  = PiLayer(nInput, nHidden, permuteInput=True)  # Hidden layer using PiLayer
        self.output  = SigmaLayer(nHidden, nOutput, permuteInput=True)  # Output layer using SigmaLayer

        self.vq = VQLayer(
            dim = embeddingDim,
            codebookSize = 100,
            numQuantizers = 100)

    def forward(self, x, t=0):
        # The optional VQ stage is kept for experiments but is currently disabled
        # so the logical layers operate directly on the input tensor.
        #x = self.vq(x,t)
        x1 = self.hidden(x, t)  # Pass through PiLayer
        x2 = self.output(x1, t)  # Pass through SigmaLayer
        return x2

def logic(X_train, Y_train):
    """Train the toy network on a hand-built logical truth table."""
    # Hyperparameters
    input_dim  = X_train.shape[1]  # Number of input
    hidden_dim = 3  # Number of hidden
    output_dim = Y_train.shape[1]  # Number of output

    # Initialize the model, loss function, and optimizer
    model     = LogicalFunctionNet(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam Optimizer

    # Training loop
    epochs      = 1000
    mse_history = []

    t = 0.0001
    for epoch in range(epochs):
        optimizer.zero_grad()              # Clear gradients
        outputs = model(X_train, t)        # Forward pass on all truth-table rows
        loss    = criterion(outputs, Y_train) # Compute loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights
        mse_history.append(loss.item())    # Store loss history
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, MSE: {loss.item():.6f}')

    # Plot the Mean Squared Error (MSE) over time
    plt.figure(figsize=(8,5))
    plt.plot(mse_history, label='MSE Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Test the trained model
    with torch.no_grad():
        test_outputs = model(X_train, 0)
        print("\nPredictions after training:")
        for i, (x, y) in enumerate(zip(X_train, test_outputs)):
            print(f"Input: {x.numpy()}, Predicted Output: {y.item():.4f}")
    # Display the learned decision boundary
    if input_dim == 2:
        plot_decision_boundary(model, X_train.numpy(), Y_train.numpy())


# Plot decision boundary
def plot_decision_boundary(model, X, Y):
    """Visualise the learned scalar output over the 2D boolean input plane."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    # The model expects shape (batch, symbols, embeddingDim); for the boolean
    # examples the embedding dimension is 1, so unsqueeze creates that axis.
    grid_tensor = grid_tensor.unsqueeze(2)
    with torch.no_grad():
        Z = model(grid_tensor).reshape(xx.shape)
    Z = Z.squeeze()
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.5)
    plt.colorbar(label="Model Output")

    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), edgecolors='k', cmap="coolwarm", s=100)
    plt.xlabel("Input Feature x1")
    plt.ylabel("Input Feature x2")
    plt.title("Decision Boundary of PiLayer + SigmaLayer")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Define the logical functions training set
    if embeddingDim == 1:
        zero = torch.zeros(1,1,embeddingDim)
        one  = torch.ones(1,1,embeddingDim)
    else:
        # Higher-dimensional experiments use random embeddings for logical 0/1.
        zero = torch.rand(1,1,embeddingDim)
        one  = torch.rand(1,1,embeddingDim)
    X_train = torch.concatenate((
        torch.concatenate((zero, zero), dim=1),
        torch.concatenate((zero, one), dim=1),
        torch.concatenate((one, zero), dim=1),
        torch.concatenate((one, one), dim=1)
    ), dim=0)

    # Corresponding target outputs for AND, OR, XOR, and other logical functions
    target = 'XOR'

    # OR
    if target == 'OR':
        Y_train = torch.tensor([
            [0], [1], [1], [1]], dtype=torch.float32)
    # NOR
    if target == 'NOR':
        Y_train = torch.tensor([
            [1], [0], [0], [0]], dtype=torch.float32)
    # AND
    if target == 'AND':
        Y_train = torch.tensor([
            [0], [0], [0], [1]], dtype=torch.float32)
    # NAND
    if target == 'NAND':
        Y_train = torch.tensor([
            [1], [1], [1], [0]], dtype=torch.float32)
    # XOR
    if target == 'XOR':
        Y_train = torch.tensor([
            [0], [1], [1], [0]], dtype=torch.float32)
    # NXOR
    if target == 'NXOR':
        Y_train = torch.tensor([
            [1], [0], [0], [1]], dtype=torch.float32)
    Y_train = Y_train.unsqueeze(2)
    logic(X_train, Y_train)
