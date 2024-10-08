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
from utils import get_dataloaders, mask_image

id_name = 'cifar10'

dataloaders = get_dataloaders(id_name)

# load the model

net = ResNet18_32x32(num_classes=10)
net.load_state_dict(
    torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
)
net.cuda()
net.eval()

mask_tensor = torch.tensor(
    [
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    ]
)


masked_image = mask_image(mask_tensor)

pbar = tqdm(dataloaders['id'][0], total=dataloaders['id'][1])

for batch in pbar:
    print(batch.size)
