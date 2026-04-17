"""Standalone XOR / logic-gate experiment using Sigma and Pi layers.

Builds a two-input truth table, trains a small Pi->Sigma network on it,
and plots the loss curve and (for 2-D inputs) the decision boundary.
Run directly to execute the experiment:  python SigmaPi.py
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import util
import matplotlib.pyplot as plt
from Layers import Layer, SigmaLayer, PiLayer
from visualize import TheReport

# Minimal in-repo VectorQuantize / ResidualVQ -- covers the subset of the
# vector_quantize_pytorch API VQLayer uses. The external package was
# removed once Codebook owned commitment loss / STE / rotation trick.
class VectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight=1.0,
                 use_cosine_sim=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_cosine_sim = use_cosine_sim
        self.codebook = torch.randn(codebook_size, dim)

    @property
    def codebook(self):
        return self._parameters["_codebook"]

    @codebook.setter
    def codebook(self, value):
        param = value if isinstance(value, nn.Parameter) else nn.Parameter(value.detach().clone())
        if "_codebook" in self._parameters:
            self._parameters["_codebook"] = param
        else:
            self.register_parameter("_codebook", param)
        self.codebook_size = param.shape[0]

    def forward(self, x, return_all_codes=False, **kwargs):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        codebook = self.codebook
        if self.use_cosine_sim:
            flat_cmp = F.normalize(flat, dim=-1)
            codebook_cmp = F.normalize(codebook, dim=-1)
            indices = (flat_cmp @ codebook_cmp.T).argmax(dim=-1)
        else:
            indices = torch.cdist(flat, codebook).argmin(dim=-1)
        quantized_raw = codebook[indices].reshape(original_shape)
        commit_loss = self.commitment_weight * F.mse_loss(
            x, quantized_raw.detach()
        )
        quantized = x + (quantized_raw - x).detach()
        indices = indices.reshape(original_shape[:-1])
        if return_all_codes:
            return quantized, indices, commit_loss, quantized.unsqueeze(0)
        return quantized, indices, commit_loss


class ResidualVQ(nn.Module):
    def __init__(self, dim, codebook_size, num_quantizers=1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=dim,
                codebook_size=codebook_size,
                commitment_weight=kwargs.get("commitment_weight", 1.0),
                use_cosine_sim=kwargs.get("use_cosine_sim", False),
            )
            for _ in range(num_quantizers)
        ])

    def forward(self, x, return_all_codes=False, **kwargs):
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices, all_codes = [], []
        total_loss = x.new_tensor(0.0)
        for layer in self.layers:
            quantized, indices, commit_loss = layer(residual, **kwargs)
            quantized_total = quantized_total + quantized
            residual = residual - quantized.detach()
            total_loss = total_loss + commit_loss
            all_indices.append(indices)
            all_codes.append(quantized)
        stacked_indices = torch.stack(all_indices, dim=0)
        stacked_codes = torch.stack(all_codes, dim=0)
        if return_all_codes:
            return quantized_total, stacked_indices, total_loss, stacked_codes
        return quantized_total, stacked_indices, total_loss


class VQLayer(Layer):
    """Vector-quantization layer backed by a residual VQ codebook.

    Flattens the input to (N, dim), quantizes each vector against a
    learned codebook using cosine similarity, and returns the codes from
    all quantizer stages.  The reverse pass is not implemented because
    codebook lookup is not uniquely invertible.

    Moved from Layers.py into this module because SigmaPi.py is the only
    remaining consumer (and the forward call in LogicalFunctionNet below
    is itself commented out).
    """
    nOutput = 0

    def __init__(self, dim, codebookSize, numQuantizers):
        super(VQLayer, self).__init__(dim, dim)
        self.vq = ResidualVQ(
            dim=dim,
            codebook_size=codebookSize,
            num_quantizers=numQuantizers,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=True,
            rotation_trick=True,  # rotation trick gradient estimator (vs STE)
        )

    def distance(self, x, y):
        """Euclidean distance between two tensors."""
        return torch.sqrt(torch.sum((x - y) ** 2))

    def forward(self, x, t=0):
        batch = len(x)
        x = x.reshape((-1, self.nInput))
        quantized, indices, commit_loss, all_codes = self.vq(x, return_all_codes=True)
        return all_codes

    def reverse(self, y, t=0):
        raise ValueError("VQLayer reverse is not defined; codebook lookup is not invertible.")

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
        self.hidden  = PiLayer(nInput, nHidden)
        self.output  = SigmaLayer(nHidden, nOutput)

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
