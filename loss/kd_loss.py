
import torch.nn as nn
from torch.nn import functional as F

import torch

class KD(nn.Module):
    def __init__(self):
        super(KD, self).__init__()
    
    def forward(self, student, teacher, temperature, factor = None):
        pred_student = F.log_softmax(student / temperature, dim=1)
        pred_teacher = F.softmax(teacher / temperature, dim=1)
        if factor == None:
            loss = F.kl_div(pred_student, pred_teacher, reduction="none").sum(1).mean()
        else:
            loss = F.kl_div(pred_student, pred_teacher, reduction="none").sum(1) * factor
            loss = loss.mean()
        return loss



def RefineLoss(targets, student, teacher):
    K = student.size(1)
    pred_student = F.softmax(student, dim=1)
    pred_teacher = F.softmax(teacher, dim=1)
    # w=1 for correct class, w=1/(K-1) for wrong classes (Eq. 10-11)
    weights = student.new_full(student.size(), 1.0 / (K - 1))
    weights.scatter_(1, targets.view(-1, 1), 1.0)
    loss = (weights * F.relu(pred_teacher - pred_student)).sum(1).mean()
    return loss

import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def mixup_data(x, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
    index = torch.randperm(x.size(0)).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, lam, index

def Mixup(net, inputs, outputs_student, temperature, alpha):
    """L_KD2 (Eq. 9): τ²·KL(f(x̂)/τ ∥ (1-λ)·p(x)/τ + λ·p(x')/τ)"""
    mixed_x, lam, index = mixup_data(inputs, alpha=alpha)
    logit = net(mixed_x)
    if isinstance(logit, list):
        logit = logit[0][0]
    elif isinstance(logit, tuple):
        logit = logit[0]
    soft_x  = F.softmax(outputs_student.detach() / temperature, dim=1)
    soft_xp = F.softmax(outputs_student.detach()[index] / temperature, dim=1)
    soft_teacher = (1 - lam) * soft_x + lam * soft_xp
    log_pred = F.log_softmax(logit / temperature, dim=1)
    loss = F.kl_div(log_pred, soft_teacher, reduction='batchmean') * (temperature ** 2)
    return logit, loss

