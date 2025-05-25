"""
------- Greedy sparse recovery algorithms module
description: The algorithm module includes the following algorithms and their soft (differentiable) versions:
    1. Orthogonal Matching Pursuit (OMP)
    2. Iterative Hard Thresholding (IHT)
    3. Compressive Sampling Matching Pursuit (CoSaMP) - TBA
papers:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
    - OMP-Net: Neural network unrolling of weighted Orthogonal Matching Pursuit
    https://ieeexplore.ieee.org/document/10720377
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. code needs to be changed to numpy.
    2. complex numbers need to be handled more fluidly.
    3. algorithms to be compatible to various sorting proxies than only softsort.
    4. the code is not fully compatible with GPU - To be fixed.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""

import torch
import numpy as np
from sort_proxy import soft_sort
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)

#%% #### OMP
def OMP(A, y, max_it = None, W = None, stopping_val = 0, full = False):
    m, N = A.size()
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if max_it == None:
        max_it = m/2
    if W == None:
        W = torch.ones(N, 1)
    S = torch.zeros(N, dtype = bool)
    i = 0
    rel_err = 1
    r = y
    while (i < max_it) and (rel_err >= stopping_val):
        j = torch.argmax(torch.abs(W * (A.conj().T @ r)), dim = 0)
        S[j] = True
        x_S = torch.linalg.lstsq(A[:, S], y).solution
        x[S, i + 1] = x_S.squeeze(1)
        r = y - A @ x[:, i + 1].unsqueeze(1)
        rel_err = (torch.norm(r)/torch.norm(y))**2
        i += 1
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)
    
def soft_OMP(A, y, tau = 1e-5, max_it = None, W = None, stopping_val = 0, full = False):
    m, N = A.size()
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if max_it == None:
        max_it = m/2
    if W == None:
        W = torch.ones(N, 1)
    i = 0
    r = y
    rel_err = 1
    P = torch.empty(0, N, dtype = A.dtype)
    while (i < max_it) and (rel_err >= stopping_val):
        P_tilde = soft_sort(torch.abs(W * (A.conj().T @ r)), tau, 1)
        P = torch.cat((P, P_tilde), dim = 0)
        B = A @ P.T
        z = torch.linalg.lstsq(B, y).solution
        x[:, i + 1] = (P.T @ z).squeeze(1)
        r = y - A @ x[:, i + 1].unsqueeze(1)
        rel_err = (torch.norm(r)/torch.norm(y))**2
        i += 1  
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)

#%% #### IHT
def hard_threshold(x, p, mode, W):  # weights to be added to the "Threshold" mode
    if mode == 'Threshold':
        y = torch.zeros_like(x)
        bool_ind = torch.gt(torch.abs(x), p)
        y = torch.where(bool_ind, x, y)
        return y
    elif mode == 'Sparsity':
        if not isinstance(p, int):
            raise ValueError("In sparsity mode, the input must be an integer.")
        else:
            _, sorted_indices = torch.sort(torch.abs(W*x), dim=0, descending=True, stable = False)
            y = torch.zeros_like(x)
            selected_indices = sorted_indices[:p]
            y[selected_indices] = x[selected_indices]
        return y, selected_indices

def IHT(A, y, x0 = None, p = 0.1, mode = 'Threshold', eta = 0.1, W = None, max_it = 30, full = False):
    m, N = A.size()
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if x0 == None:
        z = torch.zeros(N, 1, dtype = A.dtype)
    else:
        z = x0
    if W == None:
        W = torch.ones(N, 1)
    
    x[:, 0] = z.squeeze(1)
    n_it = 1
    while(n_it <= max_it):
        z, _ = hard_threshold(z + eta*(A.conj().T)@(y - A@z), p, mode, W)
        x[:, n_it] = z.squeeze(1)
        n_it += 1
        
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)

def soft_IHT(A, y, p, x0 = None, eta = 0.1, W = None, tau = 1e-3, max_it = 30, full = False):
    m, N = A.size()
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if x0 == None:
        z = torch.zeros(N, 1, dtype = A.dtype)
    else:
        z = x0
        
    if W == None:
        W = torch.ones(N, 1)
            
    x[:, 0] = z.squeeze(1)
    n_it = 1
    while(n_it <= max_it):
        u = z + eta*(A.conj().T)@(y - A@z)
        Q = soft_sort(abs(W*u), tau, p)
        z = torch.sum(Q, dim = 0).unsqueeze(1) * u
        x[:, n_it] = z.squeeze(1)
        n_it += 1
        
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)
