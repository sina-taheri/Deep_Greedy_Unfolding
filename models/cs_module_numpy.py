"""
------- Greedy sparse recovery algorithms module
description: TBA
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 08-06-2026
comments and future updates:
    - complex numbers need to be handled more fluidly.
    - algorithms to be compatible to various sorting proxies than only softsort.
    - the code is not fully compatible with GPU - To be fixed.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""
import numpy as np
from dataclasses import field, dataclass
from typing import Any, Optional, Union, Callable
import pywt
from sort_proxy import soft_sort

#%% #### OMP
def OMP(A, y, max_it = None, W = None, x0 = None, stopping_val = 0, full = False, lstsq = 'svd', **lstsq_kwargs):
    """
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    max_it : TYPE, optional
        DESCRIPTION. The default is None.
    W : TYPE, optional
        DESCRIPTION. The default is None.
    stopping_val : TYPE, optional
        DESCRIPTION. The default is 0.
    full : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    m, N = A.shape
    if y.ndim == 1:
        y = y[:, None]
    if x0 is None:
        x0 = np.zeros((N, 1))
    if full:
        x = x0
    if max_it is None:
        max_it = int(m/2)
    else: 
        assert max_it<int(m/2), "'max_it' can be at most equal to 'int(m/2)'"
    if W is None:
        W = np.ones((N, 1))
        
    i = 0
    r = y - A @ x0
    rel_err = 1
    S = np.zeros(N, dtype=bool)
    while (i < max_it) and (rel_err >= stopping_val):
        j = np.argmax(np.abs(W * (A.conj().T @ r)), axis=0)
        S[j] = True
        
        z = np.zeros_like(x0)
        if lstsq.lower() == 'svd':
            z[S,:] = np.linalg.lstsq(A[:, S], y)[0]
        elif lstsq.lower() == 'richardson':
            z[S,:] = Richardson(
                A[:, S], y, x0 = None, 
                max_it=lstsq_kwargs['max_it'], alpha=lstsq_kwargs['step_size']
                )
        if full:
            x = np.concatenate((x, z), axis=1)
        else:
            x = z
            
        r = y - A @ x[:, -1][:, None]
        rel_err = (np.linalg.norm(r) / np.linalg.norm(y)) ** 2
        i += 1  
        
    return x
    
def Soft_OMP(A, y, tau = 1e-5, max_it = None, x0 = None, W = None, 
        full = False, stopping_val = 0, lstsq = 'svd', **lstsq_kwargs):
    """
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    tau : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    max_it : TYPE, optional
        DESCRIPTION. The default is None.
    x0 : TYPE, optional
        DESCRIPTION. The default is None.
    W : TYPE, optional
        DESCRIPTION. The default is None.
    full : TYPE, optional
        DESCRIPTION. The default is False.
    stopping_val : TYPE, optional
        DESCRIPTION. The default is 0.
    lstsq : TYPE, optional
        DESCRIPTION. The default is None.
    **lstsq_kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    m, N = A.shape
    if y.ndim == 1:
        y = y[:, None]
    if x0 is None:
        x0 = np.zeros((N, 1))
    if full:
        x = x0
    if max_it is None:
        max_it = int(m/2)
    else: 
        assert max_it<int(m/2), "'max_it' can be at most equal to 'int(m/2)'"
    if W is None:
        W = np.ones((N, 1))
        
    i = 0
    r = y - A @ x0
    rel_err = 1
    P = np.empty((0, N), dtype=A.dtype)
    while (i < max_it) and (rel_err >= stopping_val):
        v = np.abs(W * (A.conj().T @ r))
        P_tilde = soft_sort(v.T, tau, 1)
        P = np.concatenate((P, P_tilde), axis=0)
        B = A @ P.T
        
        if lstsq.lower() == 'svd':
            z = np.linalg.lstsq(B, y)[0]
        elif lstsq.lower() == 'richardson':
            z = Richardson(B, y, x0 = None, max_it=lstsq_kwargs['max_it'], alpha=lstsq_kwargs['step_size'])
        if full:
            x = np.concatenate((x, P.T @ z), axis=1)
        else:
            x = P.T @ z
            
        r = y - A @ x[:, -1][:, None]
        rel_err = (np.linalg.norm(r)/np.linalg.norm(y))**2
        i += 1  
    
    return x

def Richardson(A, y, x0 = None, step_size = 0.1, max_it = 30, full = False):
    m, N = A.shape
    x = np.zeros((N, max_it + 1), dtype = A.dtype)
    if x0 == None:
        z = np.zeros((N, 1), dtype = A.dtype)
    else:
        z = x0
    
    x[:, 0] = z.squeeze(1)
    n_it = 1
    while(n_it <= max_it):
        z += step_size * (A.conj().T) @ (y - A@z)
        x[:, n_it] = z.squeeze(1)
        n_it += 1
        
    if full == True:
        return x
    else:
        return np.expand_dims(x[:, -1],1)
