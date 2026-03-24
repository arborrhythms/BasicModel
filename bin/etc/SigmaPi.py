"""Standalone XOR / logic-gate experiment using Sigma and Pi layers.

Builds a two-input truth table, trains a small Pi->Sigma network on it,
and plots the loss curve and (for 2-D inputs) the decision boundary.
Run directly to execute the experiment:  python SigmaPi.py
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import util
util.init_runtime_env()
import matplotlib.pyplot as plt
from Model import Layer, SigmaLayer, PiLayer, VQLayer
from visualize import TheReport

# When embeddingDim == 1, logical 0/1 are scalar tensors.
# Setting embeddingDim > 1 hides the XOR in random high-dimensional
# embeddings, forcing the network to discover the relevant dimensions.
embeddingDim = 1


class LogicalFunctionNet(Layer):
    """Two-layer logical network: Pi hidden layer followed by Sigma output.

    The Pi layer computes multiplicative (product) interactions and the
    Sigma layer computes additive (sum) interactions, together giving the
    network the capacity to learn non-linear logical functions such as XOR.
    """

    def __init__(self, nInput, nHidden, nOutput):
        super(LogicalFunctionNet, self).__init__(nInput, nOutput)
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.hidden  = PiLayer(nInput, nHidden, permuteInput=True)
        self.output  = SigmaLayer(nHidden, nOutput, permuteInput=True)

        # VQ layer is instantiated but currently unused (see forward()).
        self.vq = VQLayer(
            dim=embeddingDim,
            codebookSize=100,
            numQuantizers=100)

    def forward(self, x, t=0):
        """Forward pass through Pi -> Sigma.

        Parameters
        ----------
        x : Tensor
            Input truth-table batch.
        t : int, optional
            Vestigial parameter from the VQ codepath (VQ stage is disabled).
            Kept for interface compatibility; has no effect.
        """
        # VQ quantization stage -- disabled.  Uncomment to re-enable:
        # x = self.vq(x, t)
        x1 = self.hidden(x)
        x2 = self.output(x1)
        return x2


def logic(X_train, Y_train):
    """Train the network on a logical truth table and display results.

    Parameters
    ----------
    X_train : Tensor
        Input rows of the truth table (shape: [4, 1, embeddingDim]).
    Y_train : Tensor
        Target output column (shape: [4, 1, 1]).
    """
    input_dim  = X_train.shape[1]
    hidden_dim = 3
    output_dim = Y_train.shape[1]

    model     = LogicalFunctionNet(input_dim, hidden_dim, output_dim)
    # set_sigma(0) suppresses exploration noise, appropriate for this
    # small standalone experiment where we want reliable convergence.
    model.hidden.set_sigma(0.0001)
    model.output.set_sigma(0.0001)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs      = 1000
    mse_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss    = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        mse_history.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, MSE: {loss.item():.6f}')

    # --- Loss curve ---
    fig = plt.figure(figsize=(8, 5))
    plt.plot(mse_history, label='MSE Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    TheReport.save_figure(fig, "SigmaPi Training Loss")
    TheReport.show_figure(fig)

    # --- Evaluate on the training set (same as the truth table) ---
    with torch.no_grad():
        test_outputs = model(X_train)
        print("\nPredictions after training:")
        for i, (x, y) in enumerate(zip(X_train, test_outputs)):
            print(f"Input: {x.numpy()}, Predicted Output: {y.item():.4f}")

    if input_dim == 2:
        TheReport.plotDecisionBoundary(
            model, X_train.numpy(), Y_train.numpy(),
            title="Decision Boundary of PiLayer + SigmaLayer")

    TheReport.write_html()


if __name__ == '__main__':
    # Build the two-input truth table: (0,0), (0,1), (1,0), (1,1).
    if embeddingDim == 1:
        zero = torch.zeros(1, 1, embeddingDim)
        one  = torch.ones(1, 1, embeddingDim)
    else:
        # Random embeddings -- the network must learn which directions
        # correspond to logical 0 and 1 before solving the gate.
        zero = torch.rand(1, 1, embeddingDim)
        one  = torch.rand(1, 1, embeddingDim)

    X_train = torch.concatenate((
        torch.concatenate((zero, zero), dim=1),
        torch.concatenate((zero, one),  dim=1),
        torch.concatenate((one, zero),  dim=1),
        torch.concatenate((one, one),   dim=1)
    ), dim=0)

    # Select which logical function to learn.
    target = 'XOR'

    if target == 'OR':
        Y_train = torch.tensor([
            [0], [1], [1], [1]], dtype=torch.float32)
    if target == 'NOR':
        Y_train = torch.tensor([
            [1], [0], [0], [0]], dtype=torch.float32)
    if target == 'AND':
        Y_train = torch.tensor([
            [0], [0], [0], [1]], dtype=torch.float32)
    if target == 'NAND':
        Y_train = torch.tensor([
            [1], [1], [1], [0]], dtype=torch.float32)
    if target == 'XOR':
        Y_train = torch.tensor([
            [0], [1], [1], [0]], dtype=torch.float32)
    if target == 'NXOR':
        Y_train = torch.tensor([
            [1], [0], [0], [1]], dtype=torch.float32)

    Y_train = Y_train.unsqueeze(2)
    logic(X_train, Y_train)
