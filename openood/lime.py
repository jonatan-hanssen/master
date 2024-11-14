import torch
import os
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM
from typing import Callable, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dataloaders, mask_image, display_pytorch_image, lime_explanation
import pickle

id_name = 'cifar10'
device = 'cuda'

dataloaders = get_dataloaders(id_name)

# load the model

net = ResNet18_32x32(num_classes=10)
net.load_state_dict(
    torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
)
net.cuda()
net.eval()

betas_per = dict()

for key in ('id', 'near', 'far'):
    pbar = tqdm(dataloaders[key][0], total=dataloaders[key][1])

    all_betas = list()

    for i, batch in enumerate(pbar):
        data = batch['data'].to(device)
        betas = lime_explanation(net, data, block_size=2, mask_prob=0.4)
        all_betas.append(betas)
        if i > 20:
            pass
            # break

    betas = torch.cat(all_betas, dim=0)
    print(betas.shape)
    betas_per[key] = betas

with open('saved_metrics/lime_betas1.pkl', 'wb') as file:
    pickle.dump(betas_per, file)
