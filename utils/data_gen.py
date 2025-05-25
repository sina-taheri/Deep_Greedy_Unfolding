# -*- coding: utf-8 -*-
"""
------- Greedy sparse recovery sparsity-in-levels data module
description: The data module to generate synthetic training and test (validation)
pairs of data (x_i, y_i)_{i = 1}^n where x_i are s-sparse vectors of dimension 'N'
and y_i are observation vectors y_i = Ax_i + e of dimension 'm'. The attribute
X_hat is a N x n matrix whose columns are recovered signal from the measurements
y_i using one of the following algorithms:
    1. Orthogonal Matching Pursuit (OMP) data class
    2. Iterative Hard Thresholding (IHT) data class
    3. Compressive Sampling Matching Pursuit (CoSaMP) data class - TBA
papers:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
    - OMP-Net: Neural network unrolling of weighted Orthogonal Matching Pursuit
    https://ieeexplore.ieee.org/document/10720377
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. complex numbers need to be handled more fluidly.
    2. detach algorithms from the compressed sensing.
    3. the code is not fully compatible with GPU - To be fixed.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import math
from algs_module import OMP, IHT, CoSaMP
    
#%% #### OMP
class OMPLData(Dataset):
    def __init__(self, m, N, s, N_levels, s_levels, measure = 'Gaussian', noise_std = 0, num_train = 1024, num_val = 1024, batch_size = 32):
        super().__init__()
        if measure == 'Fourier':
            data_format = torch.cfloat
        else:
            data_format = torch.get_default_dtype()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.zeros(N, n)
        self.X_hat = torch.zeros(N, n)
        self.Y = torch.zeros(m, n, dtype = data_format)
        if measure == 'Fourier':
            # F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(0, m).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(-int(N/2) + 1, int(N/2) + 1).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            self.A = F[torch.sort(torch.randperm(N - 1)[:m])[0], :]/math.sqrt(m)
            self.idx_total = torch.randperm(N)
            for i in range(n):
                for j in range(len(N_levels)):
                    I_i = int(sum(N_levels[:j]))
                    I_f = int(sum(N_levels[:j + 1]))
                    idx_lev = self.idx_total[I_i:I_f]
                    ind = torch.randperm(N_levels[j])
                    idx = idx_lev[ind][:s_levels[j]]
                    self.X[idx, i] = torch.randn(s_levels[j])
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i].to(torch.cfloat)) + torch.randn(m, dtype=torch.cfloat)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = OMP(self.A, self.Y[:, i].unsqueeze(1), max_it = s).squeeze(1)
        else:    
            self.A = torch.randn(m, N)
            self.A = self.A/torch.sqrt(torch.sum(self.A**2, axis = 0))[None, :]
            self.idx_total = torch.range(1, N)
            for i in range(n):
                for j in range(len(N_levels)):
                    idx = torch.randperm(N_levels[j])[:s_levels[j]] + sum(N_levels[0:j]) 
                    xx = torch.randn(s_levels[j])
                    self.X[idx, i] = xx
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i]) + torch.randn(m)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = OMP(self.A, self.Y[:, i].unsqueeze(1), max_it = s).squeeze(1)
    
    def __getitem__(self, index):
        return self.Y.T[index], self.X.T[index]
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle = train)
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.Y.T, self.X.T), train, i)

#%% #### IHT
class IHTLData(Dataset):
    def __init__(self, m, N, s, N_levels, s_levels, eta_iht, measure = 'Gaussian', noise_std = 0, num_train = 1024, num_val = 1024, batch_size = 32, max_it = 30):
        super().__init__()
        if measure == 'Fourier':
            data_format = torch.cfloat
        else:
            data_format = torch.get_default_dtype()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.zeros(N, n)
        self.X_hat = torch.zeros(N, n)
        self.Y = torch.zeros(m, n, dtype = data_format)
        if measure == 'Fourier':
            # F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(0, m).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(-int(N/2) + 1, int(N/2) + 1).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            self.A = F[torch.sort(torch.randperm(N - 1)[:m])[0], :]/math.sqrt(m)
            self.idx_total = torch.randperm(N)
            for i in range(n):
                for j in range(len(N_levels)):
                    I_i = int(sum(N_levels[:j]))
                    I_f = int(sum(N_levels[:j + 1]))
                    idx_lev = self.idx_total[I_i:I_f]
                    ind = torch.randperm(N_levels[j])
                    idx = idx_lev[ind][:s_levels[j]]
                    self.X[idx, i] = torch.randn(s_levels[j])
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i].to(torch.cfloat)) + torch.randn(m, dtype=data_format)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = IHT(self.A, self.Y[:, i].unsqueeze(1), torch.zeros(N, 1, dtype = data_format), p = s, mode = 'Sparsity', eta = eta_iht, max_it = max_it).real.squeeze(1)
        else:
            self.idx_total = torch.arange(N)
            self.A = torch.randn(m, N)
            self.A = self.A/torch.sqrt(torch.sum(self.A**2, axis = 0))[None, :]     
            for i in range(n):
                for j in range(len(N_levels)):
                    idx = torch.randperm(N_levels[j])[:s_levels[j]] + sum(N_levels[0:j]) 
                    xx = torch.randn(s_levels[j])
                    self.X[idx, i] = xx
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i]) + torch.randn(m)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = IHT(self.A, self.Y[:, i].unsqueeze(1), torch.zeros(N, 1, dtype = data_format), p = s, mode = 'Sparsity', eta = eta_iht, max_it = max_it).squeeze(1)
    
    def __getitem__(self, index):
        return self.Y.T[index], self.X.T[index]
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle = train)
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.Y.T, self.X.T), train, i)

#%% #### CoSaMP    
class CoSaMPLData(Dataset):
    def __init__(self, m, N, s, N_levels, s_levels, measure = 'Gaussian', noise_std = 0, num_train = 1024, num_val = 1024, batch_size = 32, max_it = 10):
        super().__init__()
        if measure == 'Fourier':
            data_format = torch.cfloat
        else:
            data_format = torch.get_default_dtype()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.zeros(N, n)
        self.X_hat = torch.zeros(N, n, dtype = data_format)
        self.Y = torch.zeros(m, n, dtype = data_format)
        if measure == 'Fourier':
            # F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(0, m).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(-int(N/2) + 1, int(N/2) + 1).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            self.A = F[torch.sort(torch.randperm(N - 1)[:m])[0], :]/math.sqrt(m)
            self.idx_total = torch.randperm(N)
            for i in range(n):
                for j in range(len(N_levels)):
                    I_i = int(sum(N_levels[:j]))
                    I_f = int(sum(N_levels[:j + 1]))
                    idx_lev = self.idx_total[I_i:I_f]
                    ind = torch.randperm(N_levels[j])
                    idx = idx_lev[ind][:s_levels[j]]
                    self.X[idx, i] = torch.randn(s_levels[j])
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i].to(torch.cfloat)) + torch.randn(m, dtype=data_format)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = CoSaMP(self.A, self.Y[:, i].unsqueeze(1), s1 = s, max_it = max_it, lst_driver = 'gelsd').squeeze(1)
        else:
            self.idx_total = torch.arange(N)
            self.A = torch.randn(m, N)
            self.A = self.A/torch.sqrt(torch.sum(self.A**2, axis = 0))[None, :]     
            for i in range(n):
                for j in range(len(N_levels)):
                    idx = torch.randperm(N_levels[j])[:s_levels[j]] + sum(N_levels[0:j]) 
                    xx = torch.randn(s_levels[j])
                    self.X[idx, i] = xx
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i]) + torch.randn(m)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = CoSaMP(self.A, self.Y[:, i].unsqueeze(1), s1 = s, max_it = max_it, lst_driver = 'gelsd').squeeze(1)
    
    def __getitem__(self, index):
        return self.Y.T[index], self.X.T[index]
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle = train)
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.Y.T, self.X.T), train, i)

"""
@dataclass
class CS_config:
    m: int = None
    N: int = None
    s: int = None
    measurement: str = 'Gaussian'
    noise_std: float = 1e-3
    oracle_support_ratio: int = 1
    K: int = field(init=False)
    N_levels: list = field(init=False)
    s_levels: list = [s, 0]
    n_total = 1024
    
    def __post_init__(self):
        if self.s is None or self.N is None:
            raise ValueError("Both `s` and `N` must be provided.")
        self.K = self.oracle_support_ratio * self.s
        self.N_levels = [self.K, self.N - self.K]

def CS_Data(cfg: CS_config):
    m = cfg.m
    N = cfg.N
    s = cfg.s
    measure = cfg.measurement
    noise_std = cfg.noise_std
    K = cfg.K
    N_levels = cfg.N_levels
    s_levels = cfg.s_levels
    n = cfg.n_total
    
    if measure == 'Fourier':
        data_format = torch.cfloat
    else:
        data_format = torch.get_default_dtype()
        
    X = torch.zeros(N, n)
    Y = torch.zeros(m, n, dtype = data_format)
    
    if measure == 'Fourier':
        F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(-int(N/2) + 1, int(N/2) + 1).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
        A = F[torch.sort(torch.randperm(N - 1)[:m])[0], :]/math.sqrt(m)
        idx_total = torch.randperm(N)
        for i in range(n):
            for j in range(len(N_levels)):
                I_i = int(sum(N_levels[:j]))
                I_f = int(sum(N_levels[:j + 1]))
                idx_lev = idx_total[I_i:I_f]
                ind = torch.randperm(N_levels[j])
                idx = idx_lev[ind][:s_levels[j]]
                X[idx, i] = torch.randn(s_levels[j])
            X[:, i] = X[:, i]/torch.norm(X[:, i])
            Y[:, i] = torch.matmul(A, X[:, i].to(torch.cfloat)) + torch.randn(m, dtype=data_format)*noise_std/math.sqrt(m)
    else:
        idx_total = torch.arange(N)
        A = torch.randn(m, N)
        A = A/torch.sqrt(torch.sum(A**2, axis = 0))[None, :]     
        for i in range(n):
            for j in range(len(N_levels)):
                idx = torch.randperm(N_levels[j])[:s_levels[j]] + sum(N_levels[0:j]) 
                xx = torch.randn(s_levels[j])
                X[idx, i] = xx
            X[:, i] = X[:, i]/torch.norm(X[:, i])
            Y[:, i] = torch.matmul(A, X[:, i]) + torch.randn(m)*noise_std/math.sqrt(m)
    
    return A, X, Y, idx_total[:K]
    
@dataclass
class IHT_CS_config:
    N: int = None
    eta: float = None
    max_it: int = 30
    n_total: int = 1024
    full: bool = False
    
    W: torch.Tensor = field(init=False)
    x0: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.W = torch.zeros(self.N, 1)
        self.x0 = torch.zeros(self.N, 1)

def IHT_CS(A, Y, cfg: IHT_CS_config):
    for i in range(n_total):
        X_hat[:, i] = IHT(A, Y[:, i].unsqueeze(1), torch.zeros(N, 1, dtype = data_format), p = s, mode = 'Sparsity', eta = eta_iht, max_it = max_it).real.squeeze(1)
    
class IHTLData(Dataset):
    def __init__(self, m, N, s, N_levels, s_levels, eta_iht, measure = 'Gaussian', noise_std = 0, num_train = 1024, num_val = 1024, batch_size = 32, max_it = 30):
        super().__init__()
        if measure == 'Fourier':
            data_format = torch.cfloat
        else:
            data_format = torch.get_default_dtype()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.zeros(N, n)
        self.X_hat = torch.zeros(N, n)
        self.Y = torch.zeros(m, n, dtype = data_format)
        if measure == 'Fourier':
            # F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(0, m).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            F = (torch.exp(torch.tensor(-2*torch.pi*1j/N))**(torch.arange(-int(N/2) + 1, int(N/2) + 1).unsqueeze(1) @ torch.arange(0, N).unsqueeze(0)))
            self.A = F[torch.sort(torch.randperm(N - 1)[:m])[0], :]/math.sqrt(m)
            self.idx_total = torch.randperm(N)
            for i in range(n):
                for j in range(len(N_levels)):
                    I_i = int(sum(N_levels[:j]))
                    I_f = int(sum(N_levels[:j + 1]))
                    idx_lev = self.idx_total[I_i:I_f]
                    ind = torch.randperm(N_levels[j])
                    idx = idx_lev[ind][:s_levels[j]]
                    self.X[idx, i] = torch.randn(s_levels[j])
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i].to(torch.cfloat)) + torch.randn(m, dtype=data_format)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = IHT(self.A, self.Y[:, i].unsqueeze(1), torch.zeros(N, 1, dtype = data_format), p = s, mode = 'Sparsity', eta = eta_iht, max_it = max_it).real.squeeze(1)
        else:
            self.idx_total = torch.arange(N)
            self.A = torch.randn(m, N)
            self.A = self.A/torch.sqrt(torch.sum(self.A**2, axis = 0))[None, :]     
            for i in range(n):
                for j in range(len(N_levels)):
                    idx = torch.randperm(N_levels[j])[:s_levels[j]] + sum(N_levels[0:j]) 
                    xx = torch.randn(s_levels[j])
                    self.X[idx, i] = xx
                self.X[:, i] = self.X[:, i]/torch.norm(self.X[:, i])
                self.Y[:, i] = torch.matmul(self.A, self.X[:, i]) + torch.randn(m)*noise_std/math.sqrt(m)
                self.X_hat[:, i] = IHT(self.A, self.Y[:, i].unsqueeze(1), torch.zeros(N, 1, dtype = data_format), p = s, mode = 'Sparsity', eta = eta_iht, max_it = max_it).squeeze(1)
    
    def __getitem__(self, index):
        return self.Y.T[index], self.X.T[index]
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle = train)
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.Y.T, self.X.T), train, i)
"""