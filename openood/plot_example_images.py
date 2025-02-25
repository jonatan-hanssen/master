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
    set_fontsize,
)
import matplotlib

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--fontsize', '-s', type=int, default=22)
parser.add_argument('--num_images', '-n', type=int, default=3)
parser.add_argument('--skips', type=int, default=0)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args(sys.argv[1:])

if args.pgf:
    matplotlib.use('pgf')
    matplotlib.rcParams.update(
        {
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        }
    )

# set_fontsize(args.fontsize)

dataloaders = get_dataloaders(
    args.dataset, batch_size=args.num_images, full=True, shuffle=False
)

keypairs = list()

for key in dataloaders:
    for second_key in dataloaders[key]:
        keypairs.append((key, second_key))

print(keypairs)

rows = len(keypairs)
cols = args.num_images
print(rows)


fig, axes = plt.subplots(nrows=len(keypairs), ncols=args.num_images)


for i, keypair in enumerate(keypairs):
    key, second_key = keypair

    iterator = iter(dataloaders[key][second_key])

    for _ in range(args.skips):
        next(iterator)

    images = next(iterator)['data']
    for j in range(cols):
        ax = axes[i][j]
        display_pytorch_image(images[j], ax=ax)
        if args.pgf:
            file_path = (
                f'../master/figure/{args.dataset}_examples/image-img{i * cols + j}.png'
            )
            display_pytorch_image(images[j], save_path=file_path)

    Key, Second_key = key.capitalize(), second_key.capitalize()

    if key == 'near' or key == 'far':
        axes[i][cols // 2].set_title(f'{Key}-OOD: {Second_key}')
    else:
        axes[i][cols // 2].set_title(f'{Key}: {Second_key}')

plt.tight_layout()

if args.pgf:
    exit()
    directory = f'../master/figure/{args.dataset}_examples'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/image.pgf')
else:
    plt.show()
