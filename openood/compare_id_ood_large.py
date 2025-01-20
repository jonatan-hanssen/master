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
)
import pickle
import matplotlib.cm as cm
import shap

dogs = {
    0: 'Shih-Zu',
    1: 'Rhod. Ridgeback',
    2: 'Beagle',
    3: 'Eng. Foxhound',
    4: 'Border_terrier',
    5: 'Aus. Terrier',
    6: 'Golden_retriever',
    7: 'Old_English_sheepdog',
    8: 'Samoyed',
    9: 'Dingo',
}

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', type=bool, default=True)
parser.add_argument('--ood', '-o', type=str, default='near')
parser.add_argument(
    '--interpolation',
    '-i',
    type=str,
    default='bilinear',
    choices=['bilinear', 'nearest', 'none'],
)

args = parser.parse_args(sys.argv[1:])
print(args)

id_name = args.dataset
device = 'cuda'

dataloaders = get_dataloaders(id_name, batch_size=8, full=False, shuffle=True)

# load the model
net = get_network(id_name)

saliency_dict = dict()

generator_func = get_saliency_generator(args.generator, net, args.repeats)

while True:
    id_batch = next(dataloaders['id'][0])
    ood_batch = next(dataloaders[args.ood][0])

    id_images = id_batch['data'].to(device)
    ood_images = ood_batch['data'].to(device)

    id_saliencies = generator_func(id_images)
    ood_saliencies = generator_func(ood_images)
    print(id_saliencies.shape)

    normalize = False
    interpolation = args.interpolation
    opacity = 1.6

    for i in range(8):
        plt.subplot(4, 8, i * 2 + 1)
        plt.title('id')
        display_pytorch_image(id_images[i])
        plt.subplot(4, 8, i * 2 + 2)
        plt.title(f'{torch.mean(id_saliencies[i]):.10f}')
        overlay_saliency(
            id_images[i],
            id_saliencies[i],
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )
        print(torch.max(id_saliencies[i]))

        plt.subplot(4, 8, i * 2 + 17)
        plt.title('ood')
        display_pytorch_image(ood_images[i])
        plt.subplot(4, 8, i * 2 + 18)
        plt.title(f'{torch.mean(ood_saliencies[i]):.10f}')
        overlay_saliency(
            ood_images[i],
            ood_saliencies[i],
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
            previous_maxval=torch.max(id_saliencies[i]),
        )

    plt.tight_layout()
    plt.show()
