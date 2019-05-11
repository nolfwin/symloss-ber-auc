import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class CustomLoss(nn.Module):
    def log_loss(self, z):
        return torch.log(torch.exp(-z)+1)

    def square_loss(self, z):
        return (1-z)**2

    def sigmoid_loss(self, z):
        return torch.sigmoid(-z)

    def hinge_loss(self, z):
        return torch.max(z-z, 1-z)

    def savage_loss(self, z):
        return torch.sigmoid(-2*z)**2

    def unhinged_loss(self, z):
        return -z + 0.5

    def barrier_hinge_loss(self, z):
        return torch.max(-200*(50+z)+50, torch.max(200*(z-50), 50-z))

    def return_loss_func_by_str(self, r_cost=None):
        loss_func = None
        if 'sq' in self.loss:
            loss_func = self.square_loss
        elif 'log' in self.loss:
            loss_func = self.log_loss
        elif 'hinge' in self.loss:
            loss_func = self.hinge_loss
        elif 'sigmoid' in self.loss:
            loss_func = self.sigmoid_loss
        elif 'savage' in self.loss:
            loss_func = self.savage_loss
        elif 'barrier' in self.loss:
            loss_func = self.barrier_hinge_loss
        elif 'unhinged' in self.loss:
            loss_func = self.unhinged_loss

        return loss_func


class CNN(CustomLoss):
    def __init__(self, loss, size=[1, 32, 32]):
        super(CNN, self).__init__()
        self.dropout_rate = 0.5
        self.loss = loss
        self.loss_func = self.return_loss_func_by_str()
        self.conv = nn.Sequential(
            nn.Conv2d(size[0], 18, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(18, 48, 5),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(48*5*5, 800),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(400, 1)
            )

    def forward(self, x):
        y = x.size(0)
        x = self.conv(x)
        x = x.view(y, -1)
        return self.linear(x)

class Cor(CNN):
    def get_ber_loss(self, x_p, x_n, n_p, n_n):
        loss = torch.sum(self.loss_func(self.forward(x_p)))/n_p + torch.sum(self.loss_func(-self.forward(x_n)))/n_n
        return loss

    def get_auc_loss(self, x_p, x_n, n_p, n_n):
        gg = (torch.t(((self.forward(x_p)).repeat(1, n_n))) - ((self.forward(x_n)).repeat(1, n_p))).reshape(-1, 1)
        return 1 / (n_p * n_n) * torch.sum(self.loss_func(gg))

    def __call__(self, x_p, x_n, mode='ber'):
        n_p = len(x_p)
        n_n = len(x_n)
        # print(mode)
        if mode == 'ber':
            loss = self.get_ber_loss(x_p, x_n, n_p, n_n)
        elif mode == 'auc':
            loss = self.get_auc_loss(x_p, x_n, n_p, n_n)
        return loss

    def forward_test(self, x):
        return self.forward(x)
