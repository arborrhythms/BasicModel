import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 2D Rotation Matrix and Its Derivative
# -------------------------------
def rotation(theta):
    """
    Returns the 2D rotation matrix for angle theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])
def drotation_dtheta(theta):
    """
    Returns the derivative of the 2D rotation matrix with respect to theta.

    d/dθ [cosθ, -sinθ; sinθ, cosθ] = [-sinθ, -cosθ; cosθ, -sinθ]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c],
                     [c, -s]])
# -------------------------------
# Neural Network Definition (NumPy, 2D version)
# -------------------------------
class BidirectionalLinearNumpy:
    """
    A single-layer neural network with weight matrix:
       W = R1 @ D @ R2,
    where:
      - R1 = rotation(theta1) and R2 = rotation(theta2) are 2D rotation matrices,
      - D = diag(diag) is a diagonal matrix.

    Its inverse is computed analytically via:
       W⁻¹ = R2ᵀ @ D⁻¹ @ R1ᵀ.

    The model parameters are:
       theta1, theta2 (scalars) and diag (a 2-element vector).
    """

    def __init__(self):
        self.dim = 2
        # Initialize small random angles.
        self.theta1 = np.random.randn() * 0.01
        self.theta2 = np.random.randn() * 0.01
        # Initialize diag so that D starts as the identity.
        self.diag = np.ones(self.dim)

    def get_weight(self):
        R1 = rotation(self.theta1)
        R2 = rotation(self.theta2)
        D = np.diag(self.diag)
        return R1 @ D @ R2

    def get_inverse_weight(self):
        R1 = rotation(self.theta1)
        R2 = rotation(self.theta2)
        D_inv = np.diag(1.0 / self.diag)
        # Since rotation matrices are orthogonal: R⁻¹ = Rᵀ.
        return R2.T @ D_inv @ R1.T

    def forward(self, x):
        return x @ self.get_weight()

    def inverse(self, y):
        return y @ self.get_inverse_weight()
# -------------------------------
# Loss Function
# -------------------------------
def compute_loss(model, x, y, alpha=0.8, beta = 0.2):
    """
    Computes the loss as the product of:
       L_forward = MSE(x @ W, y)
       L_reverse = MSE(y @ W⁻¹, x)
    """
    W = model.get_weight()
    y_pred = x @ W
    loss_forward = np.mean((y_pred - y) ** 2)

    W_inv = model.get_inverse_weight()
    x_pred = y @ W_inv
    loss_reverse = np.mean((x_pred - x) ** 2)

    # options here for (uninformed) tie-breaking:
    loss = alpha*loss_forward + beta*loss_reverse, loss_forward, loss_reverse
    #if random() > 0.5:
    #    loss = loss_forward, loss_forward, loss_reverse
    #else:
    #    loss = loss_reverse, loss_forward, loss_reverse
    return loss
# -------------------------------
# Analytic Gradient Computation
# -------------------------------
def analytic_gradients(model, x, y):
    """
    Computes analytic gradients of the product loss L = L_forward * L_reverse
    with respect to the parameters theta1, theta2, and diag.

    We first compute the gradients of the forward loss L_forward and reverse loss L_reverse
    with respect to the weight matrices W and W⁻¹.

    Let:
       grad_W = ∂L_forward/∂W = (2/n) * xᵀ (xW - y)
       grad_Winv = ∂L_reverse/∂W⁻¹ = (2/n) * yᵀ (yW⁻¹ - x)

    Then, by the chain rule, for any parameter p:

       dL/dp = L_reverse * <grad_W, dW/dp> + L_forward * <grad_Winv, dW⁻¹/dp>

    where <A, B> denotes the elementwise Frobenius inner product.
    """
    n = x.shape[0]
    # Forward pass
    R1 = rotation(model.theta1)
    R2 = rotation(model.theta2)
    D = np.diag(model.diag)
    D_inv = np.diag(1.0 / model.diag)

    W = R1 @ D @ R2
    W_inv = R2.T @ D_inv @ R1.T

    # Compute predictions and errors.
    y_pred = x @ W
    error_forward = y_pred - y
    L_forward = np.mean(error_forward ** 2)

    x_pred = y @ W_inv
    error_reverse = x_pred - x
    L_reverse = np.mean(error_reverse ** 2)

    # Gradients with respect to W and W_inv.
    grad_W = (2.0 / n) * (x.T @ error_forward)  # shape (2,2)
    grad_Winv = (2.0 / n) * (y.T @ error_reverse)  # shape (2,2)

    # --- Gradients with respect to theta1 ---
    dR1 = drotation_dtheta(model.theta1)  # shape (2,2)
    # dW/dtheta1 = dR1/dtheta1 @ D @ R2.
    dW_dtheta1 = dR1 @ D @ R2
    # dW_inv/dtheta1 = R2ᵀ @ D_inv @ (dR1/dtheta1)ᵀ.
    dWinv_dtheta1 = R2.T @ D_inv @ dR1.T
    grad_theta1 = L_reverse * np.sum(grad_W * dW_dtheta1) + L_forward * np.sum(grad_Winv * dWinv_dtheta1)

    # --- Gradients with respect to theta2 ---
    dR2 = drotation_dtheta(model.theta2)  # shape (2,2)
    # dW/dtheta2 = R1 @ D @ dR2/dtheta2.
    dW_dtheta2 = R1 @ D @ dR2
    # dW_inv/dtheta2 = (dR2/dtheta2)ᵀ @ D_inv @ R1ᵀ.
    dWinv_dtheta2 = dR2.T @ D_inv @ R1.T
    grad_theta2 = L_reverse * np.sum(grad_W * dW_dtheta2) + L_forward * np.sum(grad_Winv * dWinv_dtheta2)

    # --- Gradients with respect to diag ---
    grad_diag = np.zeros(model.dim)
    for i in range(model.dim):
        # dW/d(diag[i]) = R1 @ E_ii @ R2, where E_ii has a 1 at (i,i).
        E = np.zeros((model.dim, model.dim))
        E[i, i] = 1.0
        dW_ddi = R1 @ E @ R2

        # dW_inv/d(diag[i]) = - R2ᵀ @ E @ R1ᵀ / (diag[i]²).
        dWinv_ddi = - R2.T @ E @ R1.T / (model.diag[i] ** 2)

        grad_diag[i] = (L_reverse * np.sum(grad_W * dW_ddi) +
                        L_forward * np.sum(grad_Winv * dWinv_ddi))

    return grad_theta1, grad_theta2, grad_diag, L_forward, L_reverse
# -------------------------------
# Simple Numpy Optimizer (SGD)
# -------------------------------
class NumpySGD:
    """
    A simple SGD optimizer using analytic gradients.
    """

    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr

    def step(self, grads):
        grad_theta1, grad_theta2, grad_diag, _, _ = grads
        self.model.theta1 -= self.lr * grad_theta1
        self.model.theta2 -= self.lr * grad_theta2
        self.model.diag -= self.lr * grad_diag
def get_optimizer(model, lr=0.001):
    """
    Exposes the optimizer as a simple Python module function.
    Returns an instance of NumpySGD for the given model.
    """
    return NumpySGD(model, lr)
# -------------------------------
# Training Function
# -------------------------------
def train_bidirectional_model(num_epochs=100, lr=0.001, num_samples=128):
    """
    Trains the BidirectionalLinearNumpy model on dummy data.

    It records the forward loss, reverse loss, and product loss (L_forward * L_reverse)
    at each epoch, and then plots these losses along with horizontal dashed lines
    representing the best possible (least squares) performance for the forward and reverse mappings.
    """
    # Initialize model and optimizer.
    model = BidirectionalLinearNumpy()
    optimizer = get_optimizer(model, lr)

    # Generate dummy training data.
    x_train = np.random.randn(num_samples, 2)
    true_W = np.random.randn(2, 2)
    y_train = x_train @ true_W + 0.1 * np.random.randn(num_samples, 2)

    forward_losses = []
    reverse_losses = []
    product_losses = []

    for epoch in range(num_epochs):
        loss, L_forward, L_reverse = compute_loss(model, x_train, y_train)
        product_losses.append(loss)
        forward_losses.append(L_forward)
        reverse_losses.append(L_reverse)

        # Compute analytic gradients.
        grads = analytic_gradients(model, x_train, y_train)
        optimizer.step(grads)

        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Product Loss: {loss:.6f}")

    # Compute theoretical best performance via least squares regression.
    # Forward mapping: find W_ls such that x_train @ W_ls ≈ y_train.
    A               = true_W
    forward_pred    = x_train @ A
    loss_forward_ls = np.mean((forward_pred - y_train) ** 2)

    # Reverse mapping: find W_ls such that y_train @ W_ls ≈ x_train.
    A_reverse       = np.linalg.pinv(true_W)
    reverse_pred    = y_train @ A_reverse
    loss_reverse_ls = np.mean((reverse_pred - x_train) ** 2)

    print(f"Theoretical Best Forward LS Loss: {loss_forward_ls:.6f}")
    print(f"Theoretical Best Reverse LS Loss: {loss_reverse_ls:.6f}")

    epochs_arr = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_arr, forward_losses, label="Forward Loss (Network)")
    plt.plot(epochs_arr, reverse_losses, label="Reverse Loss (Network)")
    plt.plot(epochs_arr, product_losses, label="Product Loss (Network)")

    plt.plot(epochs_arr, loss_forward_ls * np.ones_like(epochs_arr), color='blue', linestyle='--',
             label="Best Forward LS Loss")
    plt.plot(epochs_arr, loss_reverse_ls * np.ones_like(epochs_arr), color='orange', linestyle='--',
             label="Best Reverse LS Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses (Product Loss) with Theoretical Best Performance")
    plt.legend()
    plt.show()


    return model, forward_losses, reverse_losses, product_losses
# -------------------------------
# Module Exposure
# -------------------------------
__all__ = ['BidirectionalLinearNumpy', 'get_optimizer', 'train_bidirectional_model', 'NumpySGD']
if __name__ == '__main__':
    train_bidirectional_model(num_epochs=10000, lr=0.001, num_samples=128)
