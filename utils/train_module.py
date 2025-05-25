# -*- coding: utf-8 -*-
"""
------- Greedy sparse recovery train module
description: The train module includes 'trainer' function that takes the train 
config as the input and fits the input dataset on the input (unrolled) neural network.
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
from my_config import train_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def trainer(model, dataset, config: train_config):
    num_epochs = config.max_epochs
    learning_rate = config.lr_cls
    
    optimizer = config.optimizer_cls(model.parameters(), lr = learning_rate)
    vec_MSELoss = config.loss_cls
    
    scheduler = None
    grad_max_norm = None
    if config.scheduler_cls:
        scheduler = config.scheduler_cls(optimizer, **(config.scheduler_kwargs or {}))
    if config.grad_max_norm:
        grad_max_norm = config.grad_max_norm
    
    running_loss_train = config.running_loss_train
    running_loss_val = config.running_loss_val
    every_n_checkpoint = config.every_n_checkpoint
    train_iter_data = config.train_iter_data
    train_loss_data = config.train_loss_data
    val_iter_data = config.val_iter_data
    val_loss_data = config.val_loss_data
    
    checkpoint_save_dir = config.checkpoint_save_dir
    train_dataloader = dataset.get_dataloader(True)
    val_dataloader = dataset.get_dataloader(False)
    train_to_val_ratio = len(train_dataloader)/len(val_dataloader)
    n_total_steps = len(train_dataloader)
    every_n_train = n_total_steps/2
    every_n_val = n_total_steps
    
    for epoch in range(num_epochs):
        iter_val = iter(val_dataloader)
        for i, batch_train in enumerate(train_dataloader):
            ## training
            Y_train, X_train = batch_train
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            
            # forward pass
            outputs_train = model(Y_train)
            loss_train = vec_MSELoss(outputs_train.real, X_train)
            
            grads = torch.zeros(64)
            for i in range(64):
                loss = (torch.norm(outputs_train[i, :].real - X_train[i, :])**2)
                grads[i] = torch.norm(torch.autograd.grad(loss, model.parameters(), retain_graph=True)[0])
                
            # backward pass
            loss_train.backward()
            # print("grad norm:", torch.norm(model.layer_soft_OMP.W.grad))
            
            if grad_max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)
                
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss_train += loss_train.item()
            
            ## validation
            if (i + 1) % train_to_val_ratio == 0:
                with torch.no_grad():
                    Y_val, X_val = next(iter_val)
                    X_val = X_val.to(device)
                    Y_val = Y_val.to(device)
                    
                    # forward pass
                    outputs_val = model(Y_val)
                    loss_val = vec_MSELoss(outputs_val.real, X_val)
                    
                    running_loss_val += loss_val.item()
            
            ## save loss values and plot
            if (((i + 1) % every_n_train == 0) and (i != 0)):
                train_iter_data.append(float(epoch*n_total_steps + i)/n_total_steps)
                train_loss_data.append(running_loss_train/every_n_train)
                
                print(f'iteration = {i + 1}/{epoch + 1}, loss = {running_loss_train/every_n_train:.3e}')
                running_loss_train = 0.0
                  
            if (((i + 1) % every_n_val == 0) and (i != 0)):
                val_iter_data.append(float(epoch*n_total_steps + i)/n_total_steps)
                val_loss_data.append(running_loss_val/(every_n_val/train_to_val_ratio))
                
                plt.plot(train_iter_data, train_loss_data, linewidth = 1.5, label = "train loss")
                plt.plot(val_iter_data, val_loss_data, linewidth = 1.5, label = "val loss")
                plt.yscale('log')
                plt.xlim([0, num_epochs + 0.1])
                plt.ylabel("MSE-Loss")
                plt.xlabel("epoch")
                plt.legend()
                plt.pause(0.01)
                
                running_loss_val = 0.0
        
        ## create checkpoint
        if (epoch % every_n_checkpoint == 0) and (epoch != 0):
            checkpoint = {
                "epoch": epoch,
                "optim_dict": optimizer.state_dict(),
                "model_dict": model.state_dict()}
            torch.save(checkpoint, checkpoint_save_dir + f'_epoch_{epoch}.pt')
        
        if scheduler:
            scheduler.step()
        
    return train_iter_data, train_loss_data, val_iter_data, val_loss_data
