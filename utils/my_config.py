# -*- coding: utf-8 -*-
"""
------- Config for the training module
description: 
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
from dataclasses import dataclass, field
import os

def vec_MSELoss(outputs, targets):
    return (torch.norm(outputs - targets, dim = 1)**2).mean()
    
@dataclass
class train_config:
    loss_cls: callable = vec_MSELoss
    optimizer_cls: callable = torch.optim.RMSprop
    
    max_epochs: int = 100
    lr_cls: float = 1e-2
    scheduler_cls: callable = None
    scheduler_kwargs: dict = None
    grad_max_norm: float = None
    running_loss_train: float = 0.0
    running_loss_val: float = 0.0
    every_n_checkpoint: int = 10
    train_iter_data: list = field(default_factory=list)
    train_loss_data: list = field(default_factory=list)
    val_iter_data: list = field(default_factory=list)
    val_loss_data: list = field(default_factory=list)
    checkpoint_save_dir: str = os.path.abspath(os.path.dirname(__file__))
    
    