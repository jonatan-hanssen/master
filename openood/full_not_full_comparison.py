import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse
import torch
import scipy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils import calculate_auc, get_palette
import utils
import matplotlib


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--auc', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--negate', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--linewidth', type=float, default=2)
parser.add_argument(
    '--full', '-f', action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--smoothing', type=float, default=0.3)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)

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

else:
    plt.rcParams.update({'font.size': 22})

plot = lambda data, linestyle, label: sns.kdeplot(
    data,
    linestyle=linestyle,
    bw_method=args.smoothing,
    label=label,
    linewidth=args.linewidth,
)

sns.set_palette(get_palette())


filename = f'saved_saliencies/{args.dataset}_{args.generator}_{args.repeats}.pkl'
with open(filename, 'rb') as file:
    saliency_dict = pickle.load(file)

score_filename = f'saved_scores/{args.dataset}.pkl'
with open(score_filename, 'rb') as file:
    score_dict = pickle.load(file)

with open(score_filename, 'rb') as file:
    score_dict2 = pickle.load(file)

if not args.full:
    new_saliency_dict = dict()
    new_score_dict = dict()
    for key in saliency_dict:
        new_saliency_dict[key] = dict()
        saliencies = [
            saliency_dict[key][second_key] for second_key in saliency_dict[key]
        ]
        if isinstance(saliencies[0], dict):
            pass
        else:
            saliencies = torch.cat(saliencies, dim=0)
        second_key = ', '.join(saliency_dict[key].keys())
        new_saliency_dict[key][second_key] = saliencies

    saliency_dict = new_saliency_dict

    for key in score_dict:
        new_score_dict[key] = dict()
        scores = [score_dict[key][second_key] for second_key in score_dict[key]]
        scores = torch.cat(scores, dim=0)
        second_key = ', '.join(score_dict[key].keys())
        new_score_dict[key][second_key] = scores

    score_dict = new_score_dict

print()


for key in score_dict2:
    for second_key in score_dict2[key]:
        score = score_dict2[key][second_key]

        logit_score = torch.max(score, dim=-1)[0]
        softmax_score = torch.max(torch.nn.functional.softmax(score, dim=-1), dim=-1)[0]

        plt.subplot(121)

        if key == 'id':
            key == 'ID'

        label = f'{key.capitalize()}: {second_key.capitalize()}'
        if key == 'id':
            plot(logit_score, linestyle='-', label=label)
        elif key == 'near':
            plot(logit_score, linestyle='--', label=label)
        elif key == 'far':
            plot(logit_score, linestyle=':', label=label)

for key in score_dict:
    for second_key in score_dict[key]:
        score = score_dict[key][second_key]

        logit_score = torch.max(score, dim=-1)[0]
        softmax_score = torch.max(torch.nn.functional.softmax(score, dim=-1), dim=-1)[0]

        plt.subplot(122)

        if key == 'id':
            label = 'ID'
        elif key == 'near':
            label = 'Near-OOD'
        elif key == 'far':
            label = 'Far-OOD'

        plot(logit_score, linestyle='-', label=label)


plt.subplot(121)
plt.legend(loc='upper left')
plt.xlabel('Maximum Logit Score')

plt.subplot(122)
plt.ylabel('')
plt.legend()
plt.xlabel('MLS after combining Near and Far')


if args.pgf:
    plt.savefig(f'../master/figure/{args.dataset}_logits_distribution.pgf')
    plt.clf()
else:
    plt.show()
