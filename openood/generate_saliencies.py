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
parser.add_argument('--early_stop', '-e', type=int, default=10000)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '--just_mean', '-j', action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args(sys.argv[1:])

id_name = args.dataset
device = 'cuda'
batch_size = args.batch_size

dataloaders = get_dataloaders(
    id_name, batch_size=batch_size, full=args.full, shuffle=True
)

# load the model
net = get_network(id_name)
generator_func = get_saliency_generator(
    args.generator,
    net,
    args.repeats,
    just_mean=args.just_mean,
    normalize=args.normalize,
)

saliency_dict = dict()
score_dict = dict()


for key in ('id', 'near', 'far'):
    if isinstance(dataloaders[key], dict):
        saliency_dict[key] = dict()
        score_dict[key] = dict()

        for second_key in dataloaders[key]:
            saliencies = list()
            scores = list()
            for i, batch in enumerate(tqdm(dataloaders[key][second_key])):
                if i > args.early_stop:
                    break
                data = batch['data'].to(device)
                preds = net(data).detach().cpu()
                scores.append(preds)
                saliencies.append(generator_func(data))

            saliency_dict[key][second_key] = torch.cat(saliencies)
            score_dict[key][second_key] = torch.cat(scores)

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

with open(f'saved_scores/{args.dataset}.pkl', 'wb') as file:
    pickle.dump(score_dict, file)
