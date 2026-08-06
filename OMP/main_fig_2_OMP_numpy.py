# -*- coding: utf-8 -*-
"""
------- Main for Figure_2 OMP recoverry error vs. tau, and difference error vs. tau
description: This is the code to generate the figures for OMP vs. Soft-OMP in
Figure 2 of the following paper:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
and Figure 1 of the following:
    - OMP-Net: Neural network unrolling of weighted Orthogonal Matching Pursuit
    https://ieeexplore.ieee.org/document/10720377
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 8-6-2026
comments and future updates:
    1. complex numbers need to be handled more fluidly.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""
if __name__ == '__my_MLP_OMP__': pass

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

#%% ### Add dierectory
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../your_project/OMP
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(os.path.dirname(script_dir), "utils"))
sys.path.append(os.path.join(os.path.dirname(script_dir), "models"))

from cs_module_numpy import OMP, Soft_OMP

#%% ### Experimental setup
experiment_num = 1  # 0: recoverry error vs. tau, 1: difference error vs. tau
N = 400
m = 200
s = 15

n_repeat = 50
tau_range = 10**np.arange(-10, 2, 0.5)
n_tau = len(tau_range)
measure = 'Gauss'     # 'Gauss' or 'Fourier'

if experiment_num == 0:
    std_range = [1e-5, 1e-3, 1e-1]
    n_std = len(std_range)
    IT = s
    core_save_name = f'experiment_{experiment_num}_N_{N}_m_{m}_s_{s}_max_it_{IT}_n_repeat_{n_repeat}'
elif experiment_num == 1:
    it_range = [5, 15, 30]
    n_it = len(it_range)
    eps = 10**-12
    noise_std = 1e-3
    core_save_name = f'experiment_{experiment_num}_N_{N}_m_{m}_s_{s}_noise_{noise_std}_n_repeat_{n_repeat}'
else:
    raise ValueError(f"{experiment_num} not valid. 'experiment num' takes only '0' or '1'.")
    
data_save_dir = os.path.join(script_dir, 'data', 'Fig_2', 'OMP_' + core_save_name + '.pt')

#%% ### Main data function
def main_fig2_data(key_val):
    if key_val == 0:
        err_omp = np.zeros((n_repeat, n_std))
        err_soft_omp = np.zeros((n_repeat, n_std, n_tau))
        for i in range(n_repeat):
            if measure.lower()=='gauss':
                A = np.random.randn(m, N)
            elif measure.lower()=='fourier': 
                F = np.exp(-2*np.pi*1j/N) ** (
                    np.arange(-int(N/2) + 1, int(N/2) + 1).reshape(-1,1) \
                        @ np.arange(-int(N/2) + 1, int(N/2) + 1).reshape(1,-1)
                        )
                A = F[np.sort(np.random.permutation(N - 1)[:m]), :]/np.sqrt(m)
            
            A /= np.linalg.norm(A)
            
            x = np.zeros((N, 1))
            idx = np.random.randint(1, N, [s])
            xx = np.random.randn(s, 1)
            x[idx, :] = xx/np.linalg.norm(xx)
            
            for j in range(n_std):
                noise = np.random.randn(m, 1)*std_range[j]/math.sqrt(m)
                
                y = A@x + noise
                
                x_hat = OMP(A, y, max_it = IT)
                err_omp[i, j] = np.linalg.norm(x_hat - x)
                
                for k, t in enumerate(tau_range):
                    x_hat = Soft_OMP(A, y, tau=t, max_it=IT)
                    err_soft_omp[i, j, k] = np.linalg.norm(x_hat - x)
        
        torch.save({'err_omp':err_omp, 'err_soft_omp': err_soft_omp}, data_save_dir)
        
    elif key_val == 1:
        err_diff = np.zeros((n_repeat, n_it, n_tau))
        for i in range(n_repeat):
            if measure.lower()=='gauss':
                A = np.random.randn(m, N)
            elif measure.lower()=='fourier': 
                F = np.exp(-2*np.pi*1j/N) ** (
                    np.arange(-int(N/2) + 1, int(N/2) + 1).reshape(-1,1) \
                        @ np.arange(-int(N/2) + 1, int(N/2) + 1).reshape(1,-1)
                        )
                A = F[np.sort(np.random.permutation(N - 1)[:m]), :]/np.sqrt(m)
            
            A /= np.linalg.norm(A)
            
            x = np.zeros((N, 1))
            idx = np.random.randint(1, N, [s])
            xx = np.random.randn(s, 1)
            x[idx, :] = xx/np.linalg.norm(xx)
            
            noise = np.random.randn(m, 1)*noise_std/math.sqrt(m)
                
            y = A@x + noise
                
            x_omp = OMP(A, y, max_it = it_range[-1], full = True).real
                
            for k, t in enumerate(tau_range):
                x_soft_omp = Soft_OMP(A, y, tau = t, max_it = it_range[-1], full = True).real
                for j in range(n_it):
                    err_diff[i, j, k] = np.linalg.norm(
                        x_omp[:, it_range[j]] - x_soft_omp[:, it_range[j]]
                        )/np.linalg.norm(x_omp[:, it_range[j]])

        torch.save({'err_diff': err_diff}, data_save_dir)
    else:
        raise ValueError("'main_fig2_data' takes only '0' or '1' as input.")  
    pass

#%% ### Main figure plot function
def main_fig2_plot(key_val):
    color_profile = [[(1, 0, 0), (0.25, 0, 0)], [(0, 1, 0), (0, 0.25, 0)], [(0, 0, 1), (0, 0, 0.25)]]
    plt.figure()
    plt.gca().set_prop_cycle(None)
    data_dict = torch.load(data_save_dir, weights_only=False)
    
    if key_val == 0:
        err_omp = data_dict['err_omp']
        err_soft_omp = data_dict['err_soft_omp']
        mu_omp = np.log10(err_omp).mean(axis = 0)
        mu_soft_omp= np.log10(err_soft_omp).mean(axis = 0)
        std_omp = np.log10(err_omp).std(axis = 0)
        std_soft_omp = np.log10(err_soft_omp).std(axis = 0)
        for i in range(n_std):
            formatted_number = "{:e}".format(std_range[i])
            a, b = formatted_number.split('e')
            # if wanted to show it in scientific notation: "${:.2f} \\times 10^{{{}}}$.format(float(a), int(b)))"
            y_omp = 10**(mu_omp[i]*np.ones(tau_range.size()))
            plt.plot(tau_range, y_omp, color = color_profile[i][0], alpha = 1, label = 'OMP ($ \sigma $ = ' + "$10^{{{}}}$)".format(int(b)))
            y_omp_up = 10**((mu_omp[i] + std_omp[i])*np.ones(tau_range.size()))
            plt.plot(tau_range, y_omp_up, color = color_profile[i][0], alpha = 0.2)
            y_omp_down = 10**((mu_omp[i] - std_omp[i])*np.ones(tau_range.size()))
            plt.plot(tau_range, y_omp_down, color = color_profile[i][0], alpha = 0.2)
            plt.fill_between(tau_range, y_omp_up, y_omp_down, interpolate=True, color = color_profile[i][0], alpha=0.2)
            
            y_soft_omp = 10**(mu_soft_omp[i, :])
            plt.plot(tau_range, y_soft_omp, color = color_profile[i][1], alpha = 1, label = 'Soft-OMP ($ \sigma $ = ' + "$10^{{{}}}$)".format(int(b)))
            y_soft_omp_up = 10**(mu_soft_omp[i, :] + std_soft_omp[i, :])
            plt.plot(tau_range, y_soft_omp_up, color = color_profile[i][1], alpha = 0.2)
            y_soft_omp_down = 10**(mu_soft_omp[i, :] - std_soft_omp[i, :])
            plt.plot(tau_range, y_soft_omp_down, color = color_profile[i][1], alpha = 0.2)
            plt.fill_between(tau_range, y_soft_omp_up, y_soft_omp_down, interpolate=True, color = color_profile[i][1], alpha=0.2)
        
        plt.ylim(10**-6.5, 10)
        plt.ylabel('Relative $ \ell^2 $-error')
        plt.legend(loc = 'upper left')
        fig_save_dir = os.path.join(script_dir, 'figs', 'Fig_2', 'OMP_shaded_plot_' + core_save_name + '.png')
    
    elif key_val == 1:
        err_diff = data_dict['err_diff']
        log_err = np.log10(eps + err_diff) # eps as the machine precision is added to avoid 0 which is problematic with log
        std_err_diff = np.std(log_err, axis = 0)
        mu_err_diff = np.mean(log_err, axis=0)
        for i in range(n_it):
            # if wanted to show it in scientific notation: "${:.2f} \\times 10^{{{}}}$.format(float(a), int(b)))"
            y_err_diff = 10**(mu_err_diff[i, :])
            plt.plot(tau_range, y_err_diff, color = color_profile[i][0], alpha = 1, label = fr"$n = {it_range[i]}$")
            y_err_diff_up = 10**(mu_err_diff[i, :] + std_err_diff[i, :])
            plt.plot(tau_range, y_err_diff_up, color = color_profile[i][0], alpha = 0.2)
            y_err_diff_down = 10**(mu_err_diff[i, :] - std_err_diff[i, :])
            plt.plot(tau_range, y_err_diff_down, color = color_profile[i][0], alpha = 0.2)
            plt.fill_between(tau_range, y_err_diff_up, y_err_diff_down, interpolate=True, color = color_profile[i][0], alpha=0.2)
            
        plt.ylim(10**-14, 10)
        plt.ylabel(r'$ \|x^{(n)} - \tilde{x}^{(n)}\|_{\ell^2}/\|x^{(n)}\|_{\ell^2} $')
        plt.legend(loc = 'upper left')
        fig_save_dir = os.path.join(script_dir, 'figs', 'Fig_2', 'OMP_shaded_plot_' + core_save_name + '.png')
    else:
        raise ValueError("'main_fig2_plot' takes only '0' or '1' as input.")
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(tau_range[0], tau_range[-1])
    plt.title("(Soft-)OMP")
    plt.xlabel(r'$ \tau $')
    fig = plt.gcf()
    fig.savefig(fig_save_dir, bbox_inches='tight', dpi=300)
    
    pass

#%% #### Main
if __name__ == "__main__":  # if the data is already generated you don't need to run 'main_fig2_data'
    main_fig2_data(experiment_num)
    main_fig2_plot(experiment_num)