import torch
import os, sys
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, AblationCAM
import argparse
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget as CO
from typing import Callable, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    get_dataloaders,
    display_pytorch_image,
    numpify,
    overlay_saliency,
    get_network,
    get_saliency_generator,
    fontsize,
)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--id', '-i', type=str, default='ImageWoof')
parser.add_argument('--near', '-n', type=str, default='Stanford Dogs')
parser.add_argument('--far', '-f', type=str, default='Places365')
parser.add_argument('--fontsize', '-s', type=int, default=22)

args = parser.parse_args(sys.argv[1:])

fontsize(args.fontsize)

id_name = args.dataset

dataloaders = get_dataloaders(id_name, batch_size=8, full=False, shuffle=True)

while True:
    id_images = next(dataloaders['id'][0])['data']
    near_images = next(dataloaders['near'][0])['data']
    far_images = next(dataloaders['far'][0])['data']

    cols = 5

    fig, axes = plt.subplots(nrows=3, ncols=cols)

    for i, images in enumerate([id_images, near_images, far_images]):
        for j in range(cols):
            ax = axes[i][j]
            display_pytorch_image(images[j], ax=ax)

    axes[0][cols // 2].set_title(f'ID: {args.id}')
    axes[1][cols // 2].set_title(f'Near-OOD: {args.near}')
    axes[2][cols // 2].set_title(f'Far-OOD: {args.far}')

    plt.tight_layout()
    plt.show()
