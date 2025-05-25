# -*- coding: utf-8 -*-
"""
------- Greedy sparse recovery neural networks module
description: The NN module to construct unrolled neural network architectures
for each of the following algorithms:
    1. Orthogonal Matching Pursuit (OMP) data class
    2. Iterative Hard Thresholding (IHT) data class
    3. Compressive Sampling Matching Pursuit (CoSaMP) data class - TBA
For each algorithm it includes a MLP_soft_* nn module that combines n_*_layers of
the original algorithm and n_*_layers of the soft version.
papers:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
    - OMP-Net: Neural network unrolling of weighted Orthogonal Matching Pursuit
    https://ieeexplore.ieee.org/document/10720377
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. complex numbers need to be handled more fluidly.
    2. the code is not fully compatible to include layers of the original algorithm - To be fixed.
    3. the code is not fully compatible with GPU - To be fixed.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
from sort_proxy import soft_sort
import matplotlib.pyplot as plt

#%% #### OMP
## OMP layer
class OMP_layer(nn.Module):
    def __init__(self, A, m, N, s, tau):
        super(OMP_layer, self).__init__()
        self.N = N
        self.A = A
        
    def forward(self, X, Y, S):
        batch_size = X.size(1)
        H = torch.zeros(self.N, batch_size, dtype = self.A.dtype)
        for i in range(batch_size):   
            r = Y[i, :].T - self.A @ X[:, i]
            j = torch.argmax(abs(self.A.conj().T @ r), dim = 0)
            S[j, i] = True
            x_S = torch.linalg.lstsq(self.A[:, S[:, i]], Y[i, :].T).solution
            H[S[:, i], i] = x_S
        return H, S

## soft-OMP layer
class soft_OMP_layer(nn.Module):
    def __init__(self, A, m, N, s, tau):
        super(soft_OMP_layer, self).__init__()
        self.N = N
        self.A = A
        w = torch.ones(self.N)
        # w = torch.randn(N)
        self.W = nn.Parameter(w)
        self.tau = tau
        
    def forward(self, X, Y, P_tensor):
        batch_size = X.size(1)
        H = torch.zeros(self.N, batch_size, dtype = self.A.dtype)
        P1 = torch.zeros(batch_size, P_tensor.size(1) + 1, self.N, dtype = self.A.dtype)
        for i in range(batch_size):
            w_quantity = torch.abs(self.W * (self.A.conj().T @ (Y[i, :].T - self.A @ X[:, i])))
            P_tilde = soft_sort(w_quantity.unsqueeze(1), self.tau, 1)
            P = torch.cat((P_tensor[i, :], P_tilde), dim = 0)
            A_T = self.A @ P.T
            z = torch.linalg.lstsq(A_T, Y[i, :].T).solution
            H[:, i] = (P.T @ z)
            P1[i, :, :] = P
        return H, P1
    
## MLP model with OMP layers
class MLP_soft_OMP(nn.Module):
    def __init__(self, L, D, num_inputs, num_outputs, s, tau, n_OMP_layers, n_soft_OMP_layers):
        super(MLP_soft_OMP, self).__init__()
        self.num_outputs = num_outputs
        self.n_OMP_layers = n_OMP_layers
        self.n_soft_OMP_layers = n_soft_OMP_layers
        
        self.layers_OMP = nn.ModuleList([OMP_layer(D, num_inputs, num_outputs, s, tau) for _ in range(self.n_OMP_layers)])
        # self.layers_soft_OMP = nn.ModuleList([soft_OMP_layer(D, num_inputs, num_outputs, s, tau) for _ in range(self.n_soft_OMP_layers)])
        self.layer_soft_OMP = soft_OMP_layer(D, num_inputs, num_outputs, s, tau)
    
    def forward(self, Y):
        batch_size = Y.size(0)
        X0 = torch.zeros(self.num_outputs, batch_size, dtype = Y.dtype)
        S = torch.zeros(self.num_outputs, batch_size, dtype=bool)
        out = X0
        for layer in self.layers_OMP:
            out, S = layer(out, Y, S)
        P = torch.zeros(batch_size, self.n_OMP_layers, self.num_outputs, dtype = Y.dtype)
        for k in range(batch_size):
            P[k, :] = torch.eye(self.num_outputs)[S[:, k], :]

        for i in range(self.n_soft_OMP_layers):
            out, P = self.layer_soft_OMP(out, Y, P)
        return out.T

#%% IHT
class IHT_layer(nn.Module):
    def __init__(self, A, m, N, s, tau):
        super(IHT_layer, self).__init__()
        self.N = N
        self.A = A
        self.s = s
        
    def forward(self, X, Y):
        batch_size = X.size(1)
        H = torch.zeros(self.N, batch_size)
        H1 = torch.matmul(torch.eye(X.size(0)) - self.A.T @ self.A, X.T) + torch.matmul(self.A.T, Y)  
        for i in range(batch_size):
            H[:, i] = hard_threshold(H1[:, i], self.s, 'Sparsity')
        return H
    
## soft-IHT layer
class soft_IHT_layer(nn.Module):
    def __init__(self, A, m, N, s, tau, eta):
        super(soft_IHT_layer, self).__init__()
        self.N = N
        self.A = A
        self.s = s
        w = torch.ones(N)# + 0.05*torch.randn(N)
        # w = torch.randn(N)
        self.W = nn.Parameter(w)
        # self.W = nn.Parameter(A)
        # self.tau = nn.Parameter(torch.tensor(1e-3))
        self.tau = tau
        self.eta = eta
        # self.eta = nn.Parameter(torch.tensor(eta))
        
    def forward(self, X, Y):
        H = torch.zeros(X.size(), dtype = self.A.dtype)
        H1 = torch.zeros(X.size(), dtype = self.A.dtype)
        batch_size = X.size(1)
        H1 = torch.matmul(torch.eye(X.size(0)) - self.eta * self.A.conj().T @ self.A, X) + self.eta * torch.matmul(self.A.conj().T, Y.T)
        for i in range(batch_size):
            Q = soft_sort((abs(self.W * H1[:, i])).unsqueeze(1), self.tau, self.s)
            H[:, i] = torch.sum(Q, dim = 0) * H1[:, i]
        return H
    
## MLP model with IHT layers
class MLP_soft_IHT(nn.Module):
    def __init__(self, L, D, num_inputs, num_outputs, s, tau, eta, n_IHT_layers, n_soft_IHT_layers):
        super(MLP_soft_IHT, self).__init__()
        self.num_outputs = num_outputs
        self.n_IHT_layers = n_IHT_layers
        self.n_soft_IHT_layers = n_soft_IHT_layers
        
        self.layers_IHT = nn.ModuleList([IHT_layer(D, num_inputs, num_outputs, s, tau) for _ in range(self.n_IHT_layers)])
        # self.layers_soft_IHT = nn.ModuleList([soft_IHT_layer(D, num_inputs, num_outputs, s, tau) for _ in range(self.n_soft_IHT_layers)])
        self.layer_soft_IHT = soft_IHT_layer(D, num_inputs, num_outputs, s, tau, eta)
    
    def forward(self, Y):
        batch_size = Y.size(0)
        X0 = torch.zeros(self.num_outputs, batch_size, dtype = Y.dtype)
        S = torch.zeros(self.num_outputs, batch_size, dtype=bool)
        out = X0
        for layer in self.layers_IHT:
            out = layer(out, Y)
        # for layer in self.layers_soft_IHT:
        #     out = layer(out, Y)
        for i in range(self.n_soft_IHT_layers):
            out = self.layer_soft_IHT(out, Y)
        return out.T
