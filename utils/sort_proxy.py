# -*- coding: utf-8 -*-
"""
------- Soft-sorting strategies
description: The module generates the softsort to approximate the Permutation
matrix for the input vector. The module includes the following sorting techniques:
    1. Soft-sort
    2. Sinkhorn - Beta version
    3. Sparse soft-sort: based on sparse max - Beta version
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. 
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import time
import sys
import os

def soft_sort(s, tau, k = None, descending_bool = True):
    if k == None:
        s_sorted = s.sort(descending=descending_bool, dim=0)[0]
    else:
        if descending_bool == True:
            s_sorted, _ = torch.topk(s, k, dim = 0)
        else:
            s_sorted, _ = -torch.topk(-s, k, dim = 0)
            
    pairwise_distances = (s.T - s_sorted).abs().neg() / tau
    P_hat = pairwise_distances.softmax(1)
    return P_hat

def sinkhorn(X, l, nr = None):
    S = torch.exp(X)
    one_vec = torch.ones(X.size(0), 1)
    for i in range(l):
        S = S/(S@one_vec@one_vec.T)
        S = S/(one_vec@one_vec.T@S)
    if nr == None:
        S = S
    else:
        S = S[:nr, :]
    return S

class Sparsemax(torch.autograd.Function):
    def forward(ctx, input):
        K = len(input)
        k = torch.arange(K) + 1
        sorted_input, _ = torch.sort(input, descending = True)
        positive_indices = ((k * sorted_input) + 1 - torch.cumsum(sorted_input, dim = 0) > 0).nonzero()
        kz = positive_indices[-1]
        th = (torch.sum(sorted_input[:kz + 1]) - 1)/(kz + 1)
        output = nn.functional.relu(input - th)
        ctx.save_for_backward(output)
        return output

    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        positive_indices = (output > 0).nonzero().squeeze()
        hot_vector = torch.zeros_like(output)
        hot_vector[positive_indices] = 1
        grad_input = torch.diag(hot_vector) - (hot_vector.unsqueeze(1) @ hot_vector.unsqueeze(1).T)/torch.sum(hot_vector)  # Example backward operation
        return grad_input

def sparse_soft_sort(s, tau, descending_bool):
    s_sorted = s.sort(descending = descending_bool, dim=0)[0]
    pairwise_distances = (torch.ones_like(s) @ s.T - s_sorted @ torch.ones_like(s).T).abs().neg() / tau
    P_hat = torch.stack([Sparsemax.apply(row) for row in pairwise_distances])
    return P_hat