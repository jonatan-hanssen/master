import torch
import os, argparse
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_dataloaders,
    display_pytorch_image,
    occlusion,
    numpify,
    lime_explanation,
    overlay_saliency,
    get_network,
    GradCAMWrapper,
    segmented_occlusion,
)
import pickle
import captum

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='segocclusion')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', type=bool, default=True)

args = parser.parse_args(sys.argv[1:])

id_name = args.dataset
device = 'cuda'
batch_size = args.batch_size

dataloaders = get_dataloaders(id_name, batch_size=batch_size, full=args.full)

# load the model
net = get_network(id_name)

aggregate_dict = dict()

# if args.generator == 'gradcam':
#     cam_wrapper = GradCAMWrapper(model=net, target_layer=net.layer4[-1])
#     generator_func = cam_wrapper
# elif args.generator == 'occlusion':
#     generator_func = lambda data: occlusion(net, data, repeats=args.repeats)
# elif args.generator == 'lime':
#     generator_func = lambda data: lime_explanation(net, data, args.repeats)


for key in ('id', 'near', 'far'):
    if isinstance(dataloaders[key], dict):
        aggregate_dict[key] = dict()

        for second_key in dataloaders[key]:
            all_sets = list()
            all_saliencies = list()
            for i, batch in tqdm(enumerate(dataloaders[key][second_key])):
                data = batch['data'].to(device)
                preds = torch.argmax(net(data), dim=1)
                if i > 100:
                    break
                if args.generator == 'segocclusion':
                    saliencies = segmented_occlusion(net, data)
                    sets = list()
                    for saliency in saliencies:
                        sets.append(torch.unique(saliency))

                    all_sets += sets

                elif args.generator == 'shapley':
                    ablator = captum.attr.GradientShap(net)
                    baselines = torch.randn(20, 3, 224, 224)
                    saliencies = (
                        ablator.attribute(data, baselines.cuda(), target=preds)
                        .mean(dim=1)
                        .cpu()
                    )
                    all_saliencies.append(saliencies)

            aggregate_dict[key][second_key] = torch.cat(all_saliencies)

    else:
        pbar = tqdm(dataloaders[key][0], total=dataloaders[key][1])

        saliencies = list()
        for i, batch in enumerate(pbar):
            data = batch['data'].to(device)
            saliencies.append(generator_func(data))

        aggregate_dict[key] = torch.cat(saliencies)

with open(
    f'saved_saliencies/{args.dataset}_{args.generator}_{args.repeats}.pkl', 'wb'
) as file:
    pickle.dump(aggregate_dict, file)
