import numpy as np
import torch
import matplotlib.pyplot as plt

from models.fc_net import FullyConnectedNet
from problems.black_scholes import BlackScholesBarenblatt


def u_exact(t, X, T, r=0.05, sigma=0.4):
    return np.exp((r + sigma**2) * (T - t)) * np.sum(X**2, axis=1, keepdims=True)


def main():
    # Problem settings
    M = 100
    N = 50
    D = 100
    T = 1.0
    Xi = np.array([1.0, 0.5] * (D // 2))[None, :]

    layers = [D + 1] + 4 * [256] + [1]

    # Create model
    net = FullyConnectedNet(layers, activation="ReLU")
    model = BlackScholesBarenblatt(Xi, T, M, N, D, net)

    # Train model
    loss_history = model.train(iterations=20000, lr=1e-3)

    # Evaluate
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    t_np = t_test.cpu().numpy()
    X_np = X_pred.cpu().detach().numpy()
    Y_np = Y_pred.cpu().detach().numpy()
    Y_true = u_exact(t_np.reshape(-1, 1), X_np.reshape(-1, D), T)
    Y_true = Y_true.reshape(M, N + 1, 1)

    # Plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot solution
    plt.figure()
    plt.plot(t_np[0, :, 0], Y_np[0, :, 0], label='Predicted')
    plt.plot(t_np[0, :, 0], Y_true[0, :, 0], '--', label='Exact')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Y(t)')
    plt.title('Solution Trajectory')
    plt.show()


if __name__ == "__main__":
    main()
