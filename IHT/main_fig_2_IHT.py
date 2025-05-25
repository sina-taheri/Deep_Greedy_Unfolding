# -*- coding: utf-8 -*-
"""
------- Main for Figure_2 IHT recoverry error vs. tau, and difference error vs. tau
description: This is the code to generate the figures for IHT vs. Soft-IHT in
Figure 2 of the following paper:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
and Figure 1 of the following:
    - OMP-Net: Neural network unrolling of weighted Orthogonal Matching Pursuit
    https://ieeexplore.ieee.org/document/10720377
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. complex numbers need to be handled more fluidly.
DISCLAIMER:
    This code is provided for academic and educational purposes only. While we strive to match the original
    paper's methodology and results as closely as possible, minor discrepancies may exist due to implementation
    differences, randomness, or environment settings.
"""
if __name__ == '__my_MLP_IHT__': pass

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

#%% ### Add dierectory
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../your_project/IHT
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(os.path.dirname(script_dir), "utils"))
sys.path.append(os.path.join(os.path.dirname(script_dir), "models"))

from algs_module import IHT, soft_IHT

#%% ### Experimental setup
# torch.set_default_dtype(torch.float64)
experiment_num = 1  # 0: recoverry error vs. tau, 1: difference error vs. tau
N = 400
m = 200
s = 15
eta = 0.6

n_repeat = 100
tau_range = 10**torch.arange(-10, 2, 0.5, dtype=float)
n_tau = len(tau_range)

if experiment_num == 0:
    std_range = [1e-5, 1e-3, 1e-1]
    n_std = len(std_range)
    IT = 30
    core_save_name = f'experiment_{experiment_num}_N_{N}_m_{m}_s_{s}_max_it_{IT}_n_repeat_{n_repeat}'
elif experiment_num == 1:
    it_range = [1, 15, 45]
    n_it = len(it_range)
    eps = 10**-12
    noise_std = 1e-3
    core_save_name = f'experiment_{experiment_num}_N_{N}_m_{m}_s_{s}_noise_{noise_std}_n_repeat_{n_repeat}'
else:
    raise ValueError(f"{experiment_num} not valid. 'experiment num' takes only '0' or '1'.")
    
data_save_dir = os.path.join(script_dir, 'data', 'Fig_2', 'IHT_' + core_save_name + '.pt')

#%% ### Main data function
def main_fig2_data(key_val):
    if key_val == 0:
        err_iht = torch.zeros(n_repeat, n_std)
        err_soft_iht = torch.zeros(n_repeat, n_std, n_tau)
        for i in range(n_repeat):
            A = torch.randn(m, N)
            A = A/torch.sqrt(torch.sum(A**2, axis = 0))[None, :]
            
            x = torch.zeros(N, 1)
            idx = torch.randint(1, N, [s])
            xx = torch.randn(s, 1)
            x[idx, :] = xx/torch.norm(xx)
                
            for j in range(n_std):
                noise = torch.randn(m, 1)*std_range[j]/math.sqrt(m)
                
                y = torch.matmul(A, x) + noise
                
                x_hat = IHT(A, y, p = s, mode = 'Sparsity', eta = eta, max_it = IT)
                err_iht[i, j] = torch.norm(x_hat - x)
                
                for k, t in enumerate(tau_range):
                    x_hat = soft_IHT(A, y, p = s, eta = eta, tau = t, max_it = IT)
                    err_soft_iht[i, j, k] = torch.norm(x_hat - x)
        
        torch.save({'err_iht':err_iht, 'err_soft_iht': err_soft_iht}, data_save_dir)
        
    elif key_val == 1:
        err_diff = torch.zeros(n_repeat, n_it, n_tau)
        for i in range(n_repeat):
            A = torch.randn(m, N)
            A = A/torch.sqrt(torch.sum(A**2, axis = 0))[None, :]
            
            x = torch.zeros(N, 1)
            idx = torch.randint(1, N, [s])
            xx = torch.randn(s, 1)
            x[idx, :] = xx/torch.norm(xx)
                
            noise = torch.randn(m, 1)*noise_std/math.sqrt(m)
                
            y = torch.matmul(A, x) + noise
                
            x_iht = IHT(A, y, p = s, mode = 'Sparsity', eta = eta, max_it = it_range[-1], full = True)
                
            for k, t in enumerate(tau_range):
                x_soft_iht = soft_IHT(A, y, p = s, eta = eta, tau = t, max_it = it_range[-1], full = True)
                for j in range(n_it):
                    err_diff[i, j, k] = torch.norm(x_iht[:, it_range[j]] - x_soft_iht[:, it_range[j]])/torch.norm(x_iht[:, it_range[j]])

        torch.save({'err_diff': err_diff}, data_save_dir)
    else:
        raise ValueError("'main_fig2_data' takes only '0' or '1' as input.")  
    pass

#%% ### Main figure plot function
def main_fig2_plot(key_val):
    color_profile = [[(1, 0, 0), (0.25, 0, 0)], [(0, 1, 0), (0, 0.25, 0)], [(0, 0, 1), (0, 0, 0.25)]]
    plt.figure()
    plt.gca().set_prop_cycle(None)
    data_dict = torch.load(data_save_dir)
    
    if key_val == 0:
        err_iht = data_dict['err_iht']
        err_soft_iht = data_dict['err_soft_iht']
        mu_iht = torch.log10(err_iht).mean(dim = 0)
        mu_soft_iht= torch.log10(err_soft_iht).mean(dim = 0)
        std_iht = torch.log10(err_iht).std(dim = 0)
        std_soft_iht = torch.log10(err_soft_iht).std(dim = 0)
        for i in range(n_std):
            formatted_number = "{:e}".format(std_range[i])
            a, b = formatted_number.split('e')
            # if wanted to show it in scientific notation: "${:.2f} \\times 10^{{{}}}$.format(float(a), int(b)))"
            y_iht = 10**(mu_iht[i]*torch.ones(tau_range.size()))
            plt.plot(tau_range, y_iht, color = color_profile[i][0], alpha = 1, label = 'IHT ($ \sigma $ = ' + "$10^{{{}}}$)".format(int(b)))
            y_iht_up = 10**((mu_iht[i] + std_iht[i])*torch.ones(tau_range.size()))
            plt.plot(tau_range, y_iht_up, color = color_profile[i][0], alpha = 0.2)
            y_iht_down = 10**((mu_iht[i] - std_iht[i])*torch.ones(tau_range.size()))
            plt.plot(tau_range, y_iht_down, color = color_profile[i][0], alpha = 0.2)
            plt.fill_between(tau_range, y_iht_up, y_iht_down, interpolate=True, color = color_profile[i][0], alpha=0.2)
            
            y_soft_iht = 10**(mu_soft_iht[i, :])
            plt.plot(tau_range, y_soft_iht, color = color_profile[i][1], alpha = 1, label = 'Soft-IHT ($ \sigma $ = ' + "$10^{{{}}}$)".format(int(b)))
            y_soft_iht_up = 10**(mu_soft_iht[i, :] + std_soft_iht[i, :])
            plt.plot(tau_range, y_soft_iht_up, color = color_profile[i][1], alpha = 0.2)
            y_soft_iht_down = 10**(mu_soft_iht[i, :] - std_soft_iht[i, :])
            plt.plot(tau_range, y_soft_iht_down, color = color_profile[i][1], alpha = 0.2)
            plt.fill_between(tau_range, y_soft_iht_up, y_soft_iht_down, interpolate=True, color = color_profile[i][1], alpha=0.2)
        
        plt.ylim(10**-6.5, 10)
        plt.ylabel('Relative $ \ell^2 $-error')
        plt.legend(loc = 'upper left')
        fig_save_dir = os.path.join(script_dir, 'figs', 'Fig_2', 'IHT_shaded_plot_' + core_save_name + '.png')
    
    elif key_val == 1:
        err_diff = data_dict['err_diff']
        log_err = torch.log10(eps + err_diff) # eps as the machine precision is added to avoid 0 which is problematic with log
        std_err_diff = torch.std(log_err, dim = 0)
        mu_err_diff = torch.mean(log_err, dim=0)
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
        fig_save_dir = os.path.join(script_dir, 'figs', 'Fig_2', 'IHT_shaded_plot_' + core_save_name + '.png')
    else:
        raise ValueError("'main_fig2_plot' takes only '0' or '1' as input.")
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(tau_range[0], tau_range[-1])
    plt.title("(Soft-)IHT")
    plt.xlabel(r'$ \tau $')
    fig = plt.gcf()
    fig.savefig(fig_save_dir, bbox_inches='tight', dpi=300)
    
    pass

#%% #### Main
if __name__ == "__main__":  # if the data is already generated you don't need to run 'main_fig2_data'
    main_fig2_data(experiment_num)
    main_fig2_plot(experiment_num)