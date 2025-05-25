# -*- coding: utf-8 -*-
"""
------- Main for Figure_3 IHT MSE-Loss, oracle weights, learned weights and relative ℓ2-error boxplots
description: This is the code to generate the figures in Figure 3 of the following paper:
    - Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms
    https://arxiv.org/abs/2505.15661
    and Figure 2 of the following:
    The code includes:
        - Preamble: to set hyperparameters, generate the dataset and model, set the
        optimizer and loss and give an initial evaluation before training.
        - main_train(): to train the network from scratch.
        - best_checkpoint_finder(): to find the best checkpoint and set it on the model.
        - main_fig_gen(): to generate figures.
    Note: "comment out 'main_train()' and 'best_checkpoint_finder()' if the data
    is provided or the best checkpoint is already found."
author: Sina Mohammad-Taheri (sina.mohammadtaheri@concordia.ca)
last revised: 5-23-2025
comments and future updates:
    1. complex numbers need to be handled more fluidly.
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

#%% #### Directory management
### Add dierectory
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../your_project/IHT
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(os.path.dirname(script_dir), "utils"))
sys.path.append(os.path.join(os.path.dirname(script_dir), "models"))

from data_gen import IHTLData
from algs_module import IHT, soft_IHT
from nnet_module import MLP_soft_IHT
from train_module import trainer
from my_config import train_config
#%% #### Preliminaries
### device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## hyperparameters
N = 256
m = 36
s = 10
measure = 'Fourier'
noise_std = 1e-3
K = int(2*s)
N_levels = [K, N - K]   # sum(N_levels) = N
s_levels = [s, 0]  # sum(s_levels) = s
eta_gl = 0.5
tau = 1e-3

input_size = m
output_size = N
num_layers = 30
num_it = num_layers     # number of iterations of IHT and soft-IHT

n_val = 512
n_train = 2*n_val
batch_size = 64
num_epochs = 1000
learning_rate = 1e-2
every_n_checkpoint = 10

core_data_name = f'N_{N}_m_{m}_s_{s}_K_{K}_noise_{noise_std}_eta_{eta_gl}_ntrain_{n_train}_nval_{n_val}_batch_{batch_size}'
core_name = core_data_name + f'_tau_{tau}_lr_{learning_rate}'
# %% #### IHT data
## Generate dataset
data_name = 'IHTL_data_' + core_data_name + '.pt'
data_name = os.path.join(os.path.join(script_dir, 'data', 'Fig_3'), data_name)
# dataset = IHTLData(m, N, s, N_levels, s_levels, eta_gl, measure, noise_std, n_train, n_val, batch_size, num_it)
# torch.save(dataset, data_name)
dataset = torch.load(data_name)

## Generate dataloader
train_dataloader = dataset.get_dataloader(True)
val_dataloader = dataset.get_dataloader(False)

## Verify dataloader
'''
# Generate dataloader
train_dataloader = dataset.get_dataloader(True)
val_dataloader = dataset.get_dataloader(False)

# Verify dataloader
example = iter(train_dataloader)
Y, X = next(example)
print(Y.shape, X.shape)    

_, axs = plt.subplots(3, 2)
for i in range(3):
    axs[i, 0].stem(X[i, :])
    axs[i, 1].stem(Y[i, :])
plt.show()
'''

#### MLP IHT
model = MLP_soft_IHT(L = num_layers, D = dataset.A, num_inputs = input_size,\
                num_outputs = output_size, s = s, tau = tau, eta = eta_gl, n_IHT_layers = 0,\
                    n_soft_IHT_layers = num_layers)
### Loss and optimizer
def vec_MSELoss(outputs, targets):
    return (torch.norm(outputs - targets, dim = 1)**2).mean()
optimizer = torch.optim.RMSprop

#### Evaluation before training
X_model_before = model((dataset.Y[:, n_train:]).T).detach().real
loss_model_before = vec_MSELoss(X_model_before, dataset.X[:, n_train:n_train + n_val].T)

loss_IHT = vec_MSELoss(dataset.X_hat[:, n_train:].T, dataset.X[:, n_train:].T)

w_oracle = torch.zeros(N, 1)
w_oracle[dataset.idx_total[:K], :] = 1
X_wiht_oracle = torch.zeros(N, n_val)
for i in range(n_val):
    X_wiht_oracle[:, i] = IHT(dataset.A, dataset.Y[:, n_train + i].unsqueeze(1), p = s, mode = 'Sparsity', eta = eta_gl, W = w_oracle, max_it = num_it).real.squeeze(1)
loss_wiht_oracle = vec_MSELoss(X_wiht_oracle.T, dataset.X[:, n_train:].T)

X_oracle_soft = torch.zeros(N, n_val)
for i in range(n_val):
    X_oracle_soft[:, i] = soft_IHT(dataset.A, dataset.Y[:, n_train + i].unsqueeze(1), s, eta = eta_gl, W = w_oracle, tau = tau, max_it = num_it).real.squeeze(1)
loss_wsoft_oracle = vec_MSELoss(X_oracle_soft.T, dataset.X[:, n_train:].T)

print("Model loss (before training):", loss_model_before, "\n IHT loss", loss_IHT,\
      "\n IHT_median:", torch.median(torch.norm(dataset.X_hat[:, n_train:].T - dataset.X[:, n_train:].T, dim = 1)**2),\
          "\n WIHT_oracle_median:", torch.median(torch.norm(X_wiht_oracle.T - dataset.X[:, n_train:].T, dim = 1)**2),\
          "\n WIHT-oracle loss", loss_wiht_oracle, "\n WIHT-soft oracle loss", loss_wsoft_oracle)

#%% #### Main files
### Training loop
def main_train():
    checkpoint_name = 'IHT_checkpoint_' + core_name
    config = train_config(
        optimizer_cls = optimizer,
        loss_cls = vec_MSELoss,
        max_epochs = num_epochs,
        lr_cls = learning_rate,
        grad_max_norm = 1.0,
        checkpoint_save_dir = os.path.join(script_dir, 'data','Fig_3', 'checkpoints', checkpoint_name),
        scheduler_cls = lr_scheduler.StepLR,
        scheduler_kwargs = {"step_size": 500, "gamma": 0.5}
    )

    train_iter_data, train_loss_data, val_iter_data, val_loss_data = trainer(model, dataset, config)        
    train_loss_name = 'IHTNet_train_loss_' + core_name + '.pt'
    train_loss_name = os.path.join(script_dir, 'data', 'Fig_3', train_loss_name)
    train_loss_dict = {"train_iter_data": train_iter_data, "val_iter_data": val_iter_data, "train_loss_data": train_loss_data, "val_loss_data": val_loss_data}
    torch.save(train_loss_dict, train_loss_name)
    
    pass
        
### Generate figures
def best_checkpoint_finder():
    loss_model_all = []
    checkpoint_epoch_range = np.arange(every_n_checkpoint, num_epochs, every_n_checkpoint)
    W_evolution = torch.zeros(N, len(checkpoint_epoch_range))
    for i, epoch in enumerate(checkpoint_epoch_range):
        checkpoint_name = 'IHT_checkpoint_' + core_name + f'_epoch_{epoch}.pt'
        checkpoint_name = os.path.join(script_dir, 'data', 'Fig_3', 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_dict'])
        X_model = model((dataset.Y[:, n_train:]).T).real.detach()
        loss_model = vec_MSELoss(X_model, dataset.X[:, n_train:].T)      
        loss_model_all.append(loss_model)
        W_evolution[:, i] = checkpoint['model_dict']['layer_soft_IHT.W']
    
    i_best_model = torch.argmin(torch.tensor(loss_model_all))
    loss_best_model = loss_model_all[i_best_model]
    # print(loss_best_model)
    
    epoch_best_model = checkpoint_epoch_range[i_best_model]
    checkpoint_name = 'IHT_checkpoint_' + core_name + f'_epoch_{epoch_best_model}.pt'
    checkpoint_name = os.path.join(script_dir, 'data', 'Fig_3', 'checkpoints', checkpoint_name)
    checkpoint = torch.load(checkpoint_name)
    epoch_best = checkpoint_epoch_range[i_best_model]
    
    best_checkpoint_name = 'IHT_best_checkpoint_' + core_name + '.pt'
    best_checkpoint_dict = {"best_checkpoint": checkpoint, "W_evolution": W_evolution, "loss_best_model": loss_best_model}
    torch.save(best_checkpoint_dict, os.path.join(script_dir, 'data', 'Fig_3', 'checkpoints', best_checkpoint_name))
    
    pass

def main_fig_gen():
    train_loss_name = 'IHTNet_train_loss_' + core_name + '.pt'
    train_loss_name = os.path.join(script_dir, 'data', 'Fig_3', train_loss_name)
    # train_loss_name = os.path.join(script_dir, 'data', 'Fig_3', 'IHTNet_train_loss_N_256_m_22_s_10_noise_0.001_batch_64_ntrain_1024_nval_512_lr_0.01.pt')
    train_loss_dict = torch.load(train_loss_name)
    train_iter_data = train_loss_dict['train_iter_data']
    val_iter_data = train_loss_dict['val_iter_data']
    train_loss_data = train_loss_dict['train_loss_data']
    val_loss_data = train_loss_dict['val_loss_data']
    
    # torch.load(best)
    best_checkpoint_name = 'IHT_best_checkpoint_' + core_name + '.pt'
    best_checkpoint_dict = torch.load(os.path.join(script_dir, 'data', 'Fig_3', 'checkpoints', best_checkpoint_name))
    checkpoint = best_checkpoint_dict['best_checkpoint']
    # W_evolution = best_checkpoint_dict['W_evolution']
    
    loss_best_model = best_checkpoint_dict['loss_best_model']
    epoch_best = checkpoint['epoch']
    w_learned = checkpoint['model_dict']['layer_soft_IHT.W']
    
    model.load_state_dict(checkpoint['model_dict'])
    X_model = model((dataset.Y[:, n_train:]).T).real.detach()
    
    # post-processing: this block solves the least-squares on top s entries of the output - result not reported in the images
    X_model_ls = torch.zeros(X_model.size())
    _, indices_model = torch.topk(abs(X_model.T.real), s, dim = 0)
    for i in range(n_val):
        z = torch.linalg.lstsq(dataset.A[:, indices_model[:, i]], dataset.Y[:, n_train + i].T).solution.real
        X_model_ls[i, indices_model[:, i]] = torch.real(z)
        
    loss_model_ls = vec_MSELoss(X_model_ls, dataset.X[:, n_train:].T)
    
    
    X_wiht_learned = torch.zeros(N, n_val)
    for i in range(n_val):
        X_wiht_learned[:, i] = IHT(dataset.A, dataset.Y[:, n_train + i].unsqueeze(1), p = s, mode = 'Sparsity', eta = eta_gl, max_it = num_it, W = w_learned.unsqueeze(1)).real.squeeze(1)
    loss_wiht_learned = vec_MSELoss(X_wiht_learned.T, dataset.X[:, n_train:].T)
    
    v = np.array(val_iter_data)
    val_iter_v = v[v < epoch_best]
    val_loss_v = np.array(val_loss_data)[v < epoch_best]
    
    v = np.array(train_iter_data)
    train_iter_v = v[v < epoch_best]
    train_loss_v = np.array(train_loss_data)[v < epoch_best]
    
    plt.figure()
    plt.plot(train_iter_v, train_loss_v, linewidth = 2, label = "IHT-Net (training)")
    plt.plot(val_iter_v, val_loss_v, linewidth = 2, label = "IHT-Net (validation)")
    # plt.plot(val_iter_v, loss_wiht_oracle*torch.ones(len(val_iter_v)), linewidth = 2, label = "WIHT (oracle)")
    # plt.plot(val_iter_v, loss_wiht_learned*torch.ones(len(val_iter_v)), linewidth = 2, linestyle = '--', label = "WIHT (learned)")
    plt.plot(val_iter_v, loss_best_model*torch.ones(len(val_iter_v)), linewidth = 2, label = "IHT-Net")
    plt.plot(val_iter_v, loss_IHT*torch.ones(len(val_iter_v)), linewidth = 2, label = "IHT")
    plt.yscale('log')
    plt.xlim([0, epoch_best + 0.1])
    plt.ylabel("MSE-Loss")
    plt.xlabel("epoch")
    plt.legend(loc = 'upper right')
    fig_name = 'IHT_fig_' + core_name + '.png'
    fig_name = os.path.join(script_dir, 'figs', 'Fig_3', fig_name)
    fig = plt.gcf()
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    print("IHT loss:", loss_IHT, "\n WIHT-oracle loss:", loss_wiht_oracle, "\n WIHT-soft oracle loss:", loss_wsoft_oracle,\
          "\n WIHT_learned loss:", loss_wiht_learned, "\n Model loss (before training):", loss_model_before, \
              "\n Model loss (after training):", loss_best_model, "\n Model loss with LS:", loss_model_ls)
        
    plt.figure()
    w_init = torch.zeros(N)
    w_init[:N] = 1
    fig, axs = plt.subplots(2, 1)
    axs[0].stem(w_oracle)
    axs[0].set_title("Oracle weights", fontname = 'Times New Roman')
    axs[1].stem(w_init)
    axs[1].stem(torch.abs(w_learned), linefmt='r-', markerfmt='ro', basefmt=' ')
    axs[1].set_title("Initial weights & Learned weights", fontname = 'Times New Roman')
    fig.subplots_adjust(hspace=0.4)
    fig_name = 'IHT_weights_' + core_name + '.png'
    fig_name = os.path.join(script_dir, 'figs', 'Fig_3', fig_name)
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    
    plt.figure(figsize = (6.4, 3.2))
    err_iht = torch.norm((dataset.X_hat[:, n_train:].real.T - dataset.X[:, n_train:].T), dim = 1)\
        /torch.norm(dataset.X[:, n_train:].T, dim = 1)
    err_wiht_oracle = torch.norm((X_wiht_oracle.real.T - dataset.X[:, n_train:].T), dim = 1)\
        /torch.norm(dataset.X[:, n_train:].T, dim = 1)
    err_wiht_learned = torch.norm((X_wiht_learned.real.T - dataset.X[:, n_train:].T), dim = 1)\
        /torch.norm(dataset.X[:, n_train:].T, dim = 1)
    err_model = torch.norm((X_model.real - dataset.X[:, n_train:].T), dim = 1)\
        /torch.norm(dataset.X[:, n_train:].T, dim = 1)
    # plt.boxplot(torch.cat((err_iht.unsqueeze(0), err_wiht_learned.unsqueeze(0), err_wiht_oracle.unsqueeze(0), err_model.unsqueeze(0))))
    # plt.xticks([1, 2, 3, 4], ['IHT', 'WIHT (learned)', 'WIHT (oracle)', 'IHT-Net'])
    plt.boxplot(torch.cat((err_iht.unsqueeze(0), err_model.unsqueeze(0))), vert = False)
    plt.yticks([1, 2], ['IHT', 'IHT-Net'])
    plt.xscale('log')
    plt.xlabel('Relative $\ell^2$-error')
    fig_name = 'IHT_boxplot_' + core_name + '.png'
    fig_name = os.path.join(script_dir, 'figs', 'Fig_3', fig_name)
    fig = plt.gcf()
    fig.savefig(fig_name, bbox_inches='tight', dpi=300)
    
    pass

#%% #### Main
if __name__ == "__main__":  # if the data is already generated you don't need to run 'main_train'
    # main_train()
    # best_checkpoint_finder()
    main_fig_gen()

