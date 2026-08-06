"""
------- Greedy sparse recovery algorithms module
description: TBA
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 07-02-2026
Updates:
    - The CompSense module is in numpy.
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
    

#%% #### Compressive sensing class
@dataclass
class CompSense:
    """
    '''Class discription (F: the field R or C):
        'CS': compressed sensing y = Ax + e (x \in F^N, A \in F^{m x N}, y \in F^m)
        'SC': sparse coding x = Dz + e (x \in F^N, D \in F^{N x M}, z \in F^M)
        'CS-SC': compressed sensing + dictionary y = ADz + e (z \in F^M, D \in F^{N x M}, A \in {m x N}, y \in F^m)
    Updates:
        - The class is written in numpy to increase compatibility and efficienty
    Future updates:
        - See sensingDictionary object in matlab and implement its attributes.
    """
    problem_type: str = 'CS'
    N: Optional[int] = 128
    m: Optional[int] = None
    s: Optional[int] = None
    M: Optional[int] = None
    A: Optional[float] = None
    D: Optional[float] = None
    dict_type: Optional[list] = field(default_factory=list)
    measure_type: Optional[str] = 'Gaussian'  # 'Gaussian', 'Fourier'
    noise_type: Optional[str] = 'Gaussian'    # 'Gaussian'
    primes: Optional[list] = None
    K: Optional[int] = None
    noise_std: Optional[float] = 0
    n_data: Optional[int] = 2048
    wave_level: Optional[int] = None
    tau: Optional[float] = 1e-3
    alg: Callable[..., float] = OMP
    
    def __post_init__(self):
        self.data_format = np.complex64 if self.measure_type == 'Fourier' else np.float64
        if (self.problem_type=='CS') or (self.problem_type=='CS-SC') and (self.A==None): self.D = self._sensingDictionary_().astype(self.data_format)
        if (self.problem_type=='SC') or (self.problem_type=='CS-SC') and (self.D==None): self.D = self._sparsifyDictionary_().astype(self.data_format)
        self.alg_list = {
            'OMP', 'Soft_OMP',
            'GrLASSO', 'Soft_GrLASSO'
            'IHT', 'Soft_IHT',
            'CoSaMP', 'Soft_CoSaMP'
            }
        
    #--------- Measurement (sensing) matrix
    def _sensingDictionary_(self):
        # future update: return function handle if available
        if self.measure_type == 'Fourier':
            F = np.exp(-2*np.pi*1j/self.N) ** (
                np.arange(-int(self.N/2) + 1, int(self.N/2) + 1).reshape(-1,1) \
                    @ np.arange(-int(self.N/2) + 1, int(self.N/2) + 1).reshape(1,-1)
                    )
            A = F[np.sort(np.random.permutation(self.N - 1)[:self.m]), :]/np.sqrt(self.m)
            return A
        elif self.measure_type == 'Gaussian':
            A = np.random.randn(self.m, self.N)
            return A/np.linalg.norm(A, axis=0)
        elif self.measure_type == 'Deterministic':
            return self._MR_matrix_(self.N, self.s, self.K, self.primes)
        else:
            raise TypeError('measure not defined')
    
    @staticmethod
    def _MR_matrix_(N, s, K, primes):
        primes_K = primes[primes > s][:K]
        m = sum(primes_K)
        if m > N:
            raise ValueError("m > N invalid in compressed sensing, consider choosing different values for 's' and 'K' or 'N'.")
        M = np.zeros((m, N), dtype = bool)
        ind_mat = np.zeros((K, N), dtype = int)
        r = 0
        for j in range(K):
            for h in range(0, primes_K[j]):
                for n in range(N):
                    if n % primes_K[j] == h % primes_K[j]:
                        M[r, n] = True
                        ind_mat[j, n] = r
                r += 1
        return M, ind_mat

#--------- Dictionary  
    def _sparsifyDictionary_(self):
        # future update: return function handle
        D = np.empty((self.N, 0))
        wavelist = []
        for wv in pywt.families(): wavelist += pywt.wavelist(wv)
        for name in self.dict_type:
            if name=='dct':
                D = np.concatenate((D, self._make_dct_basis_(self.N)), axis=1)
            elif name in wavelist:
                D = np.concatenate((D, self._make_wavelet_basis_(self.N, name, self.wave_level)), axis=1)
            elif name=='eye':
                D = np.concatenate((D, np.eye(self.N)), axis=1)
            else:
                raise ValueError(f"dictionary {name} is not defined")
        return D
    
    def _make_dct_basis_(self, n):
        ind_vec = np.arange(n).reshape(-1, 1) + 0.5
        D = np.cos(np.pi * ind_vec @ ind_vec.T / n)
        return D/np.linalg.norm(D, axis=0)
    
    def _make_wavelet_basis_(self, n, wavetype, n_level):
        # n must be powers of 2.
        X = np.eye(n)
        coeffs = pywt.wavedec(X, wavetype, level=n_level, mode='per', axis=0)
        D = np.concatenate(tuple(coeffs), axis=0).T
        return D

#--------- Solver
    def solve(self, Y, alg=None, dictionary=None, **kwargs):
        if dictionary is None:
            if self.problem_type=='SC': dictionary = self.D
            elif self.problem_type=='CS': dictionary = self.A
            elif self.problem_type=='CS-SC': dictionary = self.A @ self.D
            else: raise TypeError("Problem type not defined.")
        
        alg_dict = {
            'OMP':OMP, 'Soft_OMP':Soft_OMP,
            'GrLASSO':GrLASSO, 'Soft_GrLASSO':Soft_GrLASSO,
            'IHT':IHT, 'Soft_IHT':Soft_IHT,
            'CoSaMP':CoSaMP, 'Soft_CoSaMP':Soft_CoSaMP
            }
        
        if alg is not None: solver = alg if callable(alg) else alg_dict[alg]
        elif (self.alg is not None) & callable(self.alg): solver = self.alg
        else: raise ValueError("Algorithm must be defined.")
        
        if ('soft' in solver.__name__.lower()) & ('tau' not in kwargs.keys()): kwargs['tau']=self.tau
        
        n = Y.shape[1]
        N = dictionary.shape[1]
        X_hat = np.zeros((N, n))
        
        for i in range(n): X_hat[:, i] = solver(dictionary, Y[:, i].reshape(-1, 1), **kwargs).squeeze(1).real

        return X_hat

#--------- Estimate sparsity
    def sparsity(self, X, method='pwr', param=None):
        # Methods: 'pwr', 'thresh', 'proxy'
        if method.lower()=='pwr':
            if param==None: param = 0.9    # pwr=\|x_S\|^2/\|x - x_S\|^2=0.90 is equivalent to SNR=40dB
            X_sorted = np.sort(np.abs(X), axis=0)[-1::-1]
            X_norm = X_sorted/np.linalg.norm(X, axis=0)
            X_cum = np.cumsum(X_norm**2, axis=0)
            M = np.ones_like(X_cum)
            M[X_cum>param] = 0.0
            s_vec = np.sum(M, axis=0)
        
        elif method.lower()=='thresh':
            n_th = 50
            n_sig = X.shape[1]
            f = np.zeros((n_th, n_sig))
            t_range = np.linspace(0, np.max(np.abs(X)), n_th)
            for i, t in enumerate(t_range):
                Y, supp = self.hard_threshold(X, t, 'thresh')
                f[i, :] = (np.linalg.norm(X - Y, axis=0)**2 + t**2*np.sum(supp, axis=0))
            
            T = t_range[np.argmin[f]]
            _, supp = self.hard_threshold(X, T, 'thresh')
            s_vec = np.sum(supp, axis=0)
                        
        elif method.lower()=='proxy':
            s_vec = (np.linalg.norm(X, 1, axis=0)/np.linalg.norm(X, 2, axis=0))**2
        
        else: raise ValueError(f"Method '{method}' is not defined.")
        
        return int(np.median(s_vec))
    
    @staticmethod
    def hard_threshold(X, p, mode):
        if mode.lower()=='thresh':
            X = np.where(np.abs(X)>p, X, 0)
            supp = np.abs(X)>p
        elif mode.lower()=='sparse':
            idx = np.argsort(np.abs(X), axis=0)[:-p]
            X[idx, np.arange(X.shape[1])[None,:]] = 0
            supp = np.zeros_like(X, dtype=bool)
            supp[idx, np.arange(X.shape[1])[None,:]] = False
        else: raise ValueError(f"Mode '{mode}' is not defined.")
        return X, supp
    
#--------- Random signal generator
    def rand_sparse(self, n_data=None):
        if n_data==None: n_data = self.n_data
        if self.problem_type=='CS':
            if isinstance(self.N, list) & isinstance(self.s, list):
                N = sum(self.N)
                N_levels = self.N
                s_levels = self.s
            else:
                N = self.N
                N_levels = [self.N, 0]
                s_levels = [self.s, 0]
        elif (self.problem_type=='SC') | (self.problem_type=='CS_SC'):
            if isinstance(self.M, list) & isinstance(self.s, list):
                N = sum(self.M)
                N_levels = self.M
                s_levels = self.s
            else:
                N = self.M
                N_levels = [self.M, 0]
                s_levels = [self.s, 0]
                
        X = np.zeros((N, n_data), dtype=self.data_format)
        idx_total = np.random.permutation(N)
        for i in range(n_data):
            for j in range(len(N_levels)):
                I_i = int(sum(N_levels[:j]))
                I_f = int(sum(N_levels[:j + 1]))
                idx_lev = idx_total[I_i:I_f]
                ind = np.random.permutation(N_levels[j])
                idx = idx_lev[ind][:s_levels[j]]
                X[idx, i] = np.random.randn(s_levels[j])
        return X/np.linalg.norm(X, axis=0), idx_total
            
    def add_noise(self, X, sigma = None, complex_noise = False):
        if sigma is None: sigma = self.noise_std
        if self.noise_type=='Gaussian':
            if not complex_noise:
                noise = np.random.randn(*X.shape)*sigma/np.sqrt(X.shape[0])
            else:
                noise = np.random.randn(*X.shape).astype(self.data_format)*sigma/np.sqrt(2*X.shape[0])\
                    + 1j*np.random.randn(*X.shape).astype(self.data_format)*sigma/np.sqrt(2*X.shape[0])
        else:
            raise ValueError(f"Noise type '{self.noise_type}' not defined.")
        return X + noise
    
    def generate_synth_data(self, alg:str='OMP', alg_kwargs:dict=None,
                            soft_alg:str='Soft_OMP', soft_alg_kwargs:dict=None,
                            dictionary:float=None):
        
        if dictionary is None:
            if self.problem_type=='SC': dictionary = self.D
            elif self.problem_type=='CS': dictionary = self.A
            elif self.problem_type=='CS-SC': dictionary = self.A @ self.D
            else: raise TypeError("Problem type not defined.")
        
        sig_dict = {}
        
        sig_dict['Z'], sig_dict['splits'] = self.rand_sparse(self.n_data)
        X = dictionary @ sig_dict['Z']
        X = self.add_noise(X)
        sig_dict['X'] = X/np.linalg.norm(X, axis=0)
        
        sig_dict['Z_hat'] = self.solve(sig_dict['X'], alg, **alg_kwargs)
        if soft_alg: sig_dict['Z_hat_soft'] = self.solve(sig_dict['X'], soft_alg, **soft_alg_kwargs)
            
        return sig_dict