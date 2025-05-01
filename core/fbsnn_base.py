import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, net, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.Xi = torch.from_numpy(Xi).float().to(self.device)
        self.Xi.requires_grad = True

        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.model = net.to(self.device)

        self.training_loss = []
        self.iteration = []

    def net_u(self, t, X):
        input = torch.cat((t, X), dim=1)
        u = self.model(input)
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u),
                                 allow_unused=True, retain_graph=True, create_graph=True)[0]
        return u, Du

    def Dg_tf(self, X):
        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g),
                                 allow_unused=True, retain_graph=True, create_graph=True)[0]
        return Dg

    def fetch_minibatch(self):
        dt = self.T / self.N
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D))

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(self.M, self.N, self.D))

        t = torch.from_numpy(np.cumsum(Dt, axis=1)).float().to(self.device)
        W = torch.from_numpy(np.cumsum(DW, axis=1)).float().to(self.device)

        return t, W

    def train(self, iterations, lr):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_log = []
        
        for it in range(iterations):
            optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()
            loss, *_ = self.loss_function(t_batch, W_batch, self.Xi)
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())

            if it % 100 == 0:
                print(f"Iteration {it}: Loss = {loss.item():.4e}")

        return loss_log

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_pred, Y_pred, Y0 = self.loss_function(t_star, W_star, Xi_star)
        return X_pred, Y_pred

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        pass

    @abstractmethod
    def loss_function(self, t, W, Xi):
        pass
