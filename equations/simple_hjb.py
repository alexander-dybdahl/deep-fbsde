import torch
from core.fbsde import FBSNN
import numpy as np
import matplotlib.pyplot as plt

class LQRProblem(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)
        self.A = torch.eye(D).to(self.device)
        self.B = torch.eye(D).to(self.device)
        self.Q = torch.eye(D).to(self.device)
        self.R = torch.eye(D).to(self.device)
        self.G = torch.eye(D).to(self.device)

    def phi_tf(self, t, X, Y, Z):
        Rinv = torch.inverse(self.R)
        a = -torch.bmm(Z.unsqueeze(1), Rinv.unsqueeze(0).expand(Z.size(0), -1, -1)).squeeze(1)
        term1 = torch.sum((X @ self.Q) * X, dim=1, keepdim=True)
        term2 = 0.5 * torch.sum(a * (a @ self.R), dim=1, keepdim=True)
        return term1 + term2

    def g_tf(self, X):
        return torch.sum((X @ self.G) * X, dim=1, keepdim=True)

    def mu_tf(self, t, X, Y, Z):
        Rinv = torch.inverse(self.R)
        a = -torch.bmm(Z.unsqueeze(1), Rinv.unsqueeze(0).expand(Z.size(0), -1, -1)).squeeze(1)
        return (X @ self.A.T) + (a @ self.B.T)

    def sigma_tf(self, t, X, Y):
        return torch.eye(self.D).unsqueeze(0).repeat(X.shape[0], 1, 1).to(self.device) * 0.1

def riccati_solution(T, t, A, B, Q, R, G):
    # Here we solve P(t) = solution of dP/dt = -A^T P - P A + P B R^{-1} B^T P - Q, P(T) = G
    # For 1D we solve explicitly
    return G * (1 + T - t)

def u_exact(t, X, P_t):
    return P_t * X**2

def a_exact(X, P_t, Rinv, B):
    return 2 * P_t * X

if __name__ == "__main__":

    M, N, D = 100, 30, 1
    layers = [D + 1] + 3 * [64] + [1]
    Xi = np.ones((1, D))
    T = 1.0
    mode, activation = "FC", "ReLU"
    model = LQRProblem(Xi, T, M, N, D, layers, mode, activation)
    try:
        model_path = "equations/" + f"best_model_{mode}_{activation}.pt"
        model.model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.model.eval()
        print("Pre-trained model loaded.")
    except FileNotFoundError:
        print("No pre-trained model found. Training from scratch.")

    graph = model.train(1000, 1e-3)

    t, W = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t, W)

    t_np = t.cpu().numpy()
    X_np = X_pred.detach().cpu().numpy()
    Y_np = Y_pred.detach().cpu().numpy()

    # exact solution
    P_t = riccati_solution(T, t, model.A, model.B, model.Q, model.R, model.G)
    u_ex = u_exact(t, X_pred, P_t).detach().cpu().numpy()
    a_ex = a_exact(X_pred, P_t, torch.inverse(model.R), model.B).detach().cpu().numpy()

    # predicted control
    _, grad_u = model.net_u(t[:, 0, :], X_pred[:, 0, :])

    a_pred_list = []
    for n in range(N + 1):
        _, grad_u = model.net_u(t[:, n, :], X_pred[:, n, :])  # [M, D]
        a_n = -torch.bmm(grad_u.unsqueeze(1), torch.inverse(model.R).unsqueeze(0).expand(M, -1, -1)).squeeze(1)  # [M, D]
        a_pred_list.append(a_n.unsqueeze(1))  # shape [M, 1, D]

    a_pred = torch.cat(a_pred_list, dim=1)  # shape [M, N+1, D]

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.yscale("log")
    plt.title("Training Loss")

    plt.figure()
    plt.plot(t_np[0, :, 0], Y_np[0, :, 0], label="NN $u$")
    plt.plot(t_np[0, :, 0], u_ex[0, :, 0], '--', label="Exact $u$")
    plt.legend()
    plt.title("Value Function over Time")

    plt.figure()
    plt.plot(t_np[0, :, 0], a_pred[0, :, 0].detach().cpu().numpy(), label="NN $a$")
    plt.plot(t_np[0, :, 0], a_ex[0, :, 0], '--', label="Exact $a$")
    plt.legend()
    plt.title("Optimal Control over Time")

    plt.show()
