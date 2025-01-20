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
)
import pickle
from pytorch_grad_cam import AblationCAM

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', type=bool, default=True)

args = parser.parse_args(sys.argv[1:])

id_name = args.dataset
device = 'cuda'
batch_size = args.batch_size

dataloaders = get_dataloaders(
    id_name, batch_size=batch_size, full=args.full, shuffle=True
)

# load the model
net = get_network(id_name)
generator_func = get_saliency_generator(args.generator, net, args.repeats)

saliency_dict = dict()


for key in ('id', 'near', 'far'):
    if isinstance(dataloaders[key], dict):
        saliency_dict[key] = dict()

        for second_key in dataloaders[key]:
            saliencies = list()
            for i, batch in enumerate(tqdm(dataloaders[key][second_key])):
                if i > 2:
                    break
                data = batch['data'].to(device)
                saliencies.append(generator_func(data))

            saliency_dict[key][second_key] = torch.cat(saliencies)

    else:
        pbar = tqdm(dataloaders[key][0], total=dataloaders[key][1])

        saliencies = list()
        for i, batch in enumerate(pbar):
            data = batch['data'].to(device)
            saliencies.append(generator_func(data))

        saliency_dict[key] = torch.cat(saliencies)

with open(
    f'saved_saliencies/{args.dataset}_{args.generator}_{args.repeats}.pkl', 'wb'
) as file:
    pickle.dump(saliency_dict, file)
