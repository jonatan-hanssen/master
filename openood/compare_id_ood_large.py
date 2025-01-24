import torch
import argparse
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

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--ood', '-o', type=str, default='near')
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    '--interpolation',
    '-i',
    type=str,
    default='bilinear',
    choices=['bilinear', 'nearest', 'none'],
)

args = parser.parse_args()

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

    id_preds = torch.max(torch.nn.functional.softmax(net(id_images), dim=0), dim=-1)[0]
    ood_preds = torch.max(torch.nn.functional.softmax(net(ood_images), dim=0), dim=-1)[
        0
    ]

    normalize = args.normalize
    interpolation = args.interpolation
    opacity = 1.6

    for i in range(8):
        plt.subplot(4, 8, i * 2 + 1)
        plt.title('id')
        display_pytorch_image(id_images[i])
        plt.subplot(4, 8, i * 2 + 2)
        plt.title(
            f'{torch.mean(id_saliencies[i] - torch.min(id_saliencies[i])):.3f}, {id_preds[i]:.3f}'
        )
        overlay_saliency(
            id_images[i],
            id_saliencies[i],
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )

        plt.subplot(4, 8, i * 2 + 17)
        plt.title('ood')
        display_pytorch_image(ood_images[i])
        plt.subplot(4, 8, i * 2 + 18)
        plt.title(
            f'{torch.mean(ood_saliencies[i] - torch.min(ood_saliencies[i])):.3f}, {ood_preds[i]:.3f}'
        )
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
