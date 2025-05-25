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

def HELU(x, p, alpha):
    return p*torch.relu(x - p + alpha)/alpha + (1 - p/alpha)*torch.relu(x - p) \
            - (p*torch.relu(-x - p + alpha)/alpha + (1 - p/alpha)*torch.relu(-x - p))

#%% #### CoSaMP
def CoSaMP(A, y, s1, s2 = None, x0 = None, W = None, max_it = 20, stopping_val = 0, full = False, lst_driver = None):
    if s2 == None:
        s2 = 2*s1
    m, N = A.size()
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if x0 == None:
        z = torch.zeros(N, 1, dtype = A.dtype)
    else:
        z = x0
    if W == None:
        W = torch.ones(N, 1)
    S = torch.zeros(N, dtype = bool)
    i = 0
    r = y
    rel_err = 1
    while (i < max_it) and (rel_err >= stopping_val):
        _, sorted_indices = torch.sort(torch.abs(W * (A.conj().T @ r)), dim = 0, descending = True, stable = False)
        S[sorted_indices[:s2]] = True
        z = torch.linalg.lstsq(A[:, S], y, driver = lst_driver).solution
        # z = torch.linalg.pinv(A[:, S]) @ y
        u = torch.zeros(N, 1, dtype = z.dtype)
        u[S] = z
        x_S, selected_indices = hard_threshold(u, s1, 'Sparsity', W)
        x[:, i + 1] = x_S.squeeze(1)
        S = torch.zeros(N, dtype = bool)
        S[selected_indices] = True
        r = y - A @ x[:, i + 1].unsqueeze(1)
        rel_err = (torch.norm(r)/torch.norm(y))**2
        i += 1
        
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)

def soft_CoSaMP(A, y, s1, s2 = None, x0 = None, W = None, tau1 = 1e-5, tau2 = None, max_it = 20, stopping_val = 0, full = False, lst_driver = None):
    m, N = A.size()
    if s2 == None:
        s2 = 2*s1
        
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if x0 == None:
        z = torch.zeros(N, 1, dtype = A.dtype)
    else:
        z = x0
        
    if W == None:
        W = torch.ones(N, 1)
        
    if tau2 == None:
        tau2 = tau1
        
    i = 0
    r = y
    rel_err = 1
    P1 = torch.empty(0, N)
    while (i < max_it) and (rel_err >= stopping_val):
        P_tilde = soft_sort(W * torch.abs(A.conj().T @ r), tau1, s2)
        P = torch.cat((P1, P_tilde), dim = 0).to(A.dtype)
        B = A @ P.T
        z = torch.linalg.lstsq(B, y, driver = lst_driver).solution
        
        # if version == 0:
        u = P.T @ z
        Q = soft_sort(abs(W * u), tau2, s1)
        x[:, i + 1] = (torch.sum(Q, dim = 0).unsqueeze(1) * u).squeeze(1)
        P1 = Q
         # version == 1:
        # u = P.T @ z
        # Q = soft_sort(abs(W * u), tau2, s1)
        # x[:, i + 1] = (Q.T @ Q @ u).squeeze(1)
        # P1 = Q
        # version == 2:
        # Q = soft_sort(abs((P @ W) * z), tau2, s1)
        # # x[:, i + 1] = ((Q @ P).T @ Q @ z).squeeze(1)
        # x[:, i + 1] = (P.T @ (torch.sum(Q, dim = 0).unsqueeze(1) * z)).squeeze(1)
        # P1 = Q @ P
        '''else:
            raise ValueError('Soft-CoSaMP version is not valid.')'''
        
        r = y - A @ x[:, i + 1].unsqueeze(1)
        rel_err = (torch.norm(r)/torch.norm(y))**2
        i += 1
        
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)
    
def Perm_matrix(indices):
    N = indices.size(0)
    P = torch.zeros(N, N)
    P[torch.arange(N), indices[:, 0]] = 1
    return P

def my_lst_qr(y, A):
    Q, R = torch.linalg.qr(A)
    x = torch.linalg.solve(R, Q.T @ y)
    return x

def pCoSaMP(A, y, s1, s2 = None, x0 = None, W = None, max_it = 20, stopping_val = 0, full = False, driver = None):
    m, N = A.size()
    if s2 == None:
        s2 = 2*s1
        
    x = torch.zeros(N, max_it + 1, dtype = A.dtype)
    if x0 == None:
        z = torch.zeros(N, 1, dtype = A.dtype)
    else:
        z = x0
        
    if W == None:
        W = torch.ones(N, 1)
        
    i = 0
    r = y
    rel_err = 1
    P1 = torch.empty(0, N)
    
    r1 = y
    S = torch.zeros(N, dtype = bool)
    x1 = torch.zeros(N, max_it + 1, dtype = A.dtype)
    eps = 10**-5
    while (i < max_it) and (rel_err >= stopping_val):
        # print(torch.norm(r - r1))
        _, sorted_indices1 = torch.sort(torch.abs(W * (A.conj().T @ r1)), dim = 0, descending = True, stable = False) # cosamp
        # ------
        sorted_indices = torch.argsort(torch.abs(W * (A.conj().T @ r)), dim = 0, descending = True, stable = False)
        print_indices = sorted_indices[:s2, :] - sorted_indices1[:s2, :]
        if len(print_indices[print_indices != 0]) != 0: 
            print(i, sorted_indices[:s2, :].T, sorted_indices1[:s2, :].T)
            
        S[sorted_indices1[:s2]] = True # cosamp
        # ------
        P_tilde = Perm_matrix(sorted_indices)[:s2, :]
        P = torch.cat((P1, P_tilde), dim = 0)
        # P = torch.unique(P, dim = 0)
        # p = 1/(torch.sum(P, dim = 0) + eps)
        # D = torch.diag(p)
        # B = A @ D @ (P.T)
        B = A @ P.T
        
        z1 = torch.linalg.lstsq(A[:, S], y, driver = lst_driver).solution # cosamp
        # z1 = my_lst_qr(y, A[:, S])
        # z1 = torch.linalg.pinv(A[:, S]) @ y # cosamp
        # ------
        z = torch.linalg.lstsq(B, y, driver = lst_driver).solution
        # z = my_lst_qr(y, B)
        # z = torch.linalg.pinv(B) @ y
        
        u1 = torch.zeros(N, 1) # cosamp
        u1[S] = z1 # cosamp
        # ------
        # u = D @ P.T @ z
        u = P.T @ z
        # print(torch.norm(u - u1))
        
        x_S, selected_indices1 = hard_threshold(u1, s1, 'Sparsity', W) # cosamp
        # ------
        selected_indices = torch.argsort(abs(W * u), dim = 0, descending = True, stable = False)
        
        x1[:, i + 1] = x_S.squeeze(1) # cosamp
        S = torch.zeros(N, dtype = bool) # cosamp
        S[selected_indices1] = True # cosamp
        # ------
        Q = Perm_matrix(selected_indices)[:s1, :]
        x[:, i + 1] = (torch.sum(Q, dim = 0).unsqueeze(1) * u).squeeze(1)
        P1 = Q
    
        # print(torch.norm(x1[:, i + 1] - x[:, i + 1]))
        
        r1 = y - A @ x1[:, i + 1].unsqueeze(1) # cosamp
        # ------
        r = y - A @ x[:, i + 1].unsqueeze(1)
        
        
        # p = torch.sum(P, dim = 0)
        # D = torch.diag(p)
        # B = A @ D @ P.T
        # P = torch.unique(P, dim = 0)
        
          
        # if version == 0:
        # u = D @ P.T @ z
        
        
        
        print_indices = selected_indices[:s1, :] - selected_indices1
        if len(print_indices[print_indices != 0]) != 0: 
            print(i, selected_indices.T, selected_indices[:s1, :].T)
        # print()
        '''elif version == 1:
            u = P.T @ z
            Q = Perm_matrix(torch.argsort(abs(W * u), dim = 0, descending = True))[:s1, :]
            x[:, i + 1] = (Q.T @ Q @ u).squeeze(1)
            P = Q
        elif version == 2:
            Q = Perm_matrix(torch.argsort(abs((P @ W) * z), dim = 0, descending = True))[:s1, :]
            # x[:, i + 1] = ((Q @ P).T @ Q @ z).squeeze(1)
            x[:, i + 1] = (P.T @ (torch.sum(Q, dim = 0).unsqueeze(1) * z)).squeeze(1)
            P = Q @ P
        else:
            raise ValueError('Soft-CoSaMP version is not valid.')'''
        
        rel_err = (torch.norm(r)/torch.norm(y))**2
        i += 1
        
    if full == True:
        return x
    else:
        return x[:, -1].unsqueeze(1)