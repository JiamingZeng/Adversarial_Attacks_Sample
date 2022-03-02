import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def pgd(model, x, y, step_size=0.007, epsilon=0.031,
            num_steps=10, random_start = True):
    model.eval()

    if random_start:
        perturb = torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).to(x.device)
        x_adv = x.clone() + perturb
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x.clone()

    x_adv.requires_grad_()
    batch_size = len(x)

    ce_loss = nn.CrossEntropyLoss()

    for i in range(num_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            loss = ce_loss(model(x_adv), y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), 
                          x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv
