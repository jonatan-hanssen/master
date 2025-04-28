import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils import (
    get_dataloaders,
    get_labels,
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
parser.add_argument('--skips', type=int, default=0)
parser.add_argument('--opacity', type=float, default=1.6)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--ood', '-o', type=str, default='near')
parser.add_argument('--batch_size', '-b', type=int, default=8)
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
print(args)


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


id_name = args.dataset
device = 'cuda'

dataloaders = get_dataloaders(
    id_name, batch_size=args.batch_size, full=False, shuffle=args.shuffle
)
labels = get_labels(id_name)

# load the model
net = get_network(id_name)

saliency_dict = dict()

generator_func = get_saliency_generator(
    args.generator, net, args.repeats, relu=args.relu
)

for i in range(args.skips):
    id_batch = next(dataloaders['id'][0])
    ood_batch = next(dataloaders[args.ood][0])


while True:
    id_batch = next(dataloaders['id'][0])
    ood_batch = next(dataloaders[args.ood][0])

    id_images = id_batch['data'].to(device)
    ood_images = ood_batch['data'].to(device)

    id_saliencies = generator_func(id_images)
    ood_saliencies = generator_func(ood_images)

    id_preds, id_labels = torch.max(
        torch.nn.functional.softmax(net(id_images), dim=-1), dim=-1
    )
    ood_preds, ood_labels = torch.max(
        torch.nn.functional.softmax(net(ood_images), dim=-1), dim=-1
    )

    maxval = max(torch.max(id_saliencies), torch.max(ood_saliencies))

    normalize = args.normalize
    interpolation = args.interpolation
    opacity = args.opacity

    for i in range(args.batch_size):
        plt.subplot(args.batch_size, 2, i + 1)
        if labels is not None:
            plt.title(f'{labels[id_labels[i]]}, {torch.mean(id_saliencies[i]):.3f}')
        else:
            plt.title(f'{torch.mean(id_saliencies[i]):.3f}')
        overlay_saliency(
            id_images[i],
            id_saliencies[i],
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )
        # plt.colorbar()

        plt.subplot(args.batch_size, 2, i + args.batch_size + 1)
        if labels is not None:
            plt.title(f'{labels[ood_labels[i]]}, {torch.mean(ood_saliencies[i]):.3f}')
        else:
            plt.title(f'{torch.mean(ood_saliencies[i]):.3f}')
        overlay_saliency(
            ood_images[i],
            ood_saliencies[i],
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
            previous_maxval=torch.max(id_saliencies[i]),
        )
        # plt.colorbar()

    plt.tight_layout()
    if args.pgf:
        plt.savefig(
            f'../master/figure/imagenet_heatmaps/{args.dataset}_{args.generator}_heatmaps_{"" if args.normalize else "un"}normalized.pgf'
        )
        exit()
    else:
        plt.show()
