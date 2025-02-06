import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse
import torch
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
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--smoothing', type=float, default=0.3)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    '--show_scores', '-s', action=argparse.BooleanOptionalAction, default=False
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

plot = lambda data, label: sns.kdeplot(
    data, bw_method=args.smoothing, label=label, linewidth=1.5
)

sns.set_palette(get_palette())


filename = f'saved_saliencies/{args.dataset}_{args.generator}_{args.repeats}.pkl'
with open(filename, 'rb') as file:
    saliency_dict = pickle.load(file)

if not args.full:
    new_saliency_dict = dict()
    for key in saliency_dict:
        new_saliency_dict[key] = dict()
        saliencies = [
            saliency_dict[key][second_key] for second_key in saliency_dict[key]
        ]
        saliencies = torch.cat(saliencies, dim=0)
        print(saliencies.shape)
        second_key = ', '.join(saliency_dict[key].keys())
        new_saliency_dict[key][second_key] = saliencies

    saliency_dict = new_saliency_dict

if args.show_scores:
    score_filename = f'saved_scores/{args.dataset}.pkl'
    with open(score_filename, 'rb') as file:
        score_dict = pickle.load(file)

    for key in saliency_dict:
        for second_key in saliency_dict[key]:
            score = score_dict[key][second_key]
            score = torch.max(score, dim=-1)[0]
            saliency = saliency_dict[key][second_key]
            b, h, w = saliency.shape
            saliency = saliency.reshape((b, h * w))

            aggregate = torch.mean(saliency, dim=-1)
            correlation = np.corrcoef(score.cpu(), aggregate.cpu())[0][1]
            plt.scatter(score, aggregate, label=f'{second_key}: {correlation=:.4f}')

    plt.xlabel('Maximum Logit Score')
    plt.ylabel('Mean Saliency')
    plt.legend()
    plt.show()

    score_filename = f'saved_scores/{args.dataset}.pkl'
    with open(score_filename, 'rb') as file:
        score_dict = pickle.load(file)

    for key in saliency_dict:
        for second_key in saliency_dict[key]:
            score = score_dict[key][second_key]
            score = torch.max(torch.nn.functional.softmax(score, dim=-1), dim=-1)[0]
            saliency = saliency_dict[key][second_key]
            b, h, w = saliency.shape
            saliency = saliency.reshape((b, h * w))

            aggregate = torch.mean(saliency, dim=-1)
            correlation = np.corrcoef(score.cpu(), aggregate.cpu())[0][1]
            plot(score, label=f'{second_key}')

    plt.xlabel('Logit')
    plt.legend()
    plt.show()
    exit()


functions = (
    ('Mean', torch.mean),
    ('Spread', utils.spread),
    ('Variance', utils.norm_std),
    ('Recon', utils.pca_recon_loss()),
    ('Gini', utils.gini),
    ('PCA', utils.pca_wrapper()),
    ('Max', lambda data, dim: torch.max(data, dim=-1)[0]),
    ('Entropy', utils.norm_std),
)


for name, function in functions:
    id_aggregate = None
    id_score = None
    near_aucs = list()
    far_aucs = list()
    for key in ('id', 'near', 'far'):
        if isinstance(saliency_dict[key], dict):
            for second_key in saliency_dict[key]:
                saliency = saliency_dict[key][second_key]
                saliency = saliency.cpu()

                if len(saliency.shape) == 1:
                    aggregate = saliency
                else:
                    b, h, w = saliency.shape
                    saliency = saliency.reshape((b, h * w))

                    if args.relu:
                        saliency = torch.nn.functional.relu(saliency)

                    if args.normalize:
                        saliency -= torch.mean(saliency, dim=0)
                        # saliency /= torch.std(saliency, dim=0)[0]

                    aggregate = function(saliency, dim=-1)

                if isinstance(aggregate, np.ndarray):
                    aggregate = torch.tensor(aggregate)

                if args.show_scores:
                    score = score_dict[key][second_key]
                    score = torch.max(score, dim=1)[0]

                if key == 'id':
                    id_aggregate = aggregate
                    plot(aggregate, f'{key}: {second_key}'.upper())

                    if args.show_scores:
                        id_score = score
                        plot(score, f'{key}: {second_key} scores')

                elif key == 'near':
                    auc = calculate_auc(id_aggregate, aggregate)
                    near_aucs.append(auc)

                    if args.auc:
                        plot(
                            aggregate,
                            f'{key}: {second_key}, {auc=:.2f}'.upper().replace(
                                '_', '\_'
                            ),
                        )
                    else:
                        plot(
                            aggregate, f'{key}: {second_key}'.upper().replace('_', '\_')
                        )

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)
                        plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

                elif key == 'far':
                    auc = calculate_auc(id_aggregate, aggregate)
                    far_aucs.append(auc)

                    if args.auc:
                        plot(
                            aggregate,
                            f'{key}: {second_key}, {auc=:.2f}'.upper().replace(
                                '_', '\_'
                            ),
                        )
                    else:
                        plot(
                            aggregate, f'{key}: {second_key}'.upper().replace('_', '\_')
                        )

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)
                        plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

    near_auc = np.array(near_aucs).mean()
    far_auc = np.array(far_aucs).mean()

    if args.auc:
        plt.title(
            f'{name.capitalize()} for {args.generator.capitalize()} saliencies on {args.dataset.capitalize()}.\nNear AUC {near_auc:.2f}, Far AUC {far_auc:.2f}'
        )
    else:
        plt.title(
            f'{name.capitalize()} for {args.generator.capitalize()} saliencies on {args.dataset.capitalize()}.'
        )
    plt.xlabel(name)
    plt.grid()
    plt.legend()

    if args.pgf:
        plt.savefig(
            f'../master/figure/{args.dataset}_{args.generator}_{name.lower()}.pgf'
        )
        plt.clf()
    else:
        plt.show()
