import torch
from core.fbsnn_base import FBSNN


class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, net):
        super().__init__(Xi, T, M, N, D, net)

    def phi_tf(self, t, X, Y, Z):
        # Nonlinear driver function
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))

    def g_tf(self, X):
        # Terminal condition: sum of squares
        return torch.sum(X ** 2, dim=1, keepdim=True)

    def mu_tf(self, t, X, Y, Z):
        # Drift term (zero in this example)
        return torch.zeros_like(X)

    def sigma_tf(self, t, X, Y):
        # Diffusion term
        return 0.4 * torch.diag_embed(X)

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list, Y_list = [], []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)
        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            dW = (W1 - W0).unsqueeze(-1)

            sigma = self.sigma_tf(t0, X0, Y0)
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(torch.matmul(sigma, dW), dim=-1)
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(Z0 * torch.squeeze(torch.matmul(sigma, dW), dim=-1), dim=1, keepdim=True)

            Y1, Z1 = self.net_u(t1, X1)

            loss += torch.sum((Y1 - Y1_tilde) ** 2)
            
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1
            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum((Y1 - self.g_tf(X1)) ** 2)
        loss += torch.sum((Z1 - self.Dg_tf(X1)) ** 2)

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]  # include initial prediction for logging
