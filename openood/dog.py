import torch
import os, argparse
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_dataloaders,
    display_pytorch_image,
    overlay_saliency,
    get_network,
    get_saliency_generator,
    GradCAMWrapper,
)
import pickle
from pytorch_grad_cam import AblationCAM
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--return_dim', type=int, default=3)
parser.add_argument('--early_stop', '-e', type=int, default=10000)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args(sys.argv[1:])
print(args)

id_name = args.dataset
device = 'cuda'
batch_size = args.batch_size

dataloaders = get_dataloaders(id_name, batch_size=batch_size, full=False, shuffle=True)


# load the model
net = get_network(id_name)

saliency_dict = dict()
score_dict = dict()


dataloader = dataloaders['id'][0]


cam_wrapper = GradCAMWrapper(model=net, target_layer=net.layer4[-1], normalize=False)

all_scores = list()
all_saliencies = list()

for i, batch in enumerate(dataloader):
    print(i, end='\r', flush=True)
    data = batch['data'].to(device)
    preds = net(data).detach().cpu()

    saliencies = cam_wrapper(data).detach().cpu()
    saliencies = saliencies.mean(dim=-1).mean(dim=-1)
    all_saliencies.append(saliencies)
    scores = torch.max(preds, dim=1)[0]

    all_scores.append(scores)

all_scores = torch.cat(all_scores).numpy()
all_saliencies = torch.cat(all_saliencies).numpy()
print(all_saliencies.shape)
print(all_scores.shape)
print(np.corrcoef(all_saliencies, all_scores))
