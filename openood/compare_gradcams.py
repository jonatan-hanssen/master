import torch
import os
import sys
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget as CO
from typing import Callable, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    get_dataloaders,
    display_pytorch_image,
    occlusion,
    numpify,
    lime_explanation,
    overlay_saliency,
    get_network,
    GradCAMWrapper,
)
import pickle
import matplotlib.cm as cm

from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from typing import Callable, List, Tuple, Optional


id_name = sys.argv[1]
device = 'cuda'
batch_size = 128

dataloaders = get_dataloaders(id_name, batch_size=batch_size)

# load the model

net = get_network(id_name)

grad_list = list()
for key in ('id', 'near', 'far'):
    pbar = tqdm(dataloaders[key][0], total=dataloaders[key][1])

    target_layers = [net.layer4[-1]]
    camm = GradCAMWrapper(model=net, target_layer=target_layers[0])

    grads = list()
    for i, batch in enumerate(pbar):
        data = batch['data'].to(device)
        cams = camm(data)
        print(grads)
        grads.append(cams)

    grads = torch.cat(grads)
    grad_list.append(grads)

with open(f'saved_metrics/{id_name}_grads_own.pkl', 'wb') as file:
    pickle.dump(grad_list, file)
