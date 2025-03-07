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
parser.add_argument('--dont_plot', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--negate', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--linewidth', type=float, default=1.5)
parser.add_argument(
    '--full', '-f', action=argparse.BooleanOptionalAction, default=False
)
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

if args.dont_plot:
    args.show_scores = True
    args.full = True


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
    data, bw_method=args.smoothing, label=label, linewidth=args.linewidth
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
        if isinstance(saliencies[0], dict):
            pass
        else:
            saliencies = torch.cat(saliencies, dim=0)
        print(f'{key.upper()} length: {saliencies.shape[0]}')
        second_key = ', '.join(saliency_dict[key].keys())
        new_saliency_dict[key][second_key] = saliencies

    saliency_dict = new_saliency_dict

print()

score_filename = f'saved_scores/{args.dataset}.pkl'
with open(score_filename, 'rb') as file:
    score_dict = pickle.load(file)

if args.show_scores and not args.dont_plot:
    for key in saliency_dict:
        for second_key in saliency_dict[key]:
            score = score_dict[key][second_key]
            score = torch.max(score, dim=-1)[0]
            saliency = saliency_dict[key][second_key]
            if isinstance(saliency, dict):
                print(saliency)
                aggregate = saliency['Mean']

            else:
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

bogo = True


for name, function in utils.get_aggregate_functions():
    inner = saliency_dict['id'][next(iter(saliency_dict['id']))]
    if isinstance(inner, dict):
        spaces = 11 - len(name)
        if name not in inner.keys():
            print(f'{name}:{" "*spaces} NOT FOUND')
            continue

    id_aggregate = None
    id_score = None
    id_softmax_score = None
    near_aucs = list()
    far_aucs = list()

    near_mls_aucs = list()
    far_mls_aucs = list()

    near_msp_aucs = list()
    far_msp_aucs = list()
    for key in ('id', 'near', 'far'):
        if isinstance(saliency_dict[key], dict):
            for second_key in saliency_dict[key]:
                saliency = saliency_dict[key][second_key]

                if isinstance(saliency, dict):
                    aggregate = saliency[name]
                    aggregate.cpu()

                elif len(saliency.shape) == 1:
                    saliency = saliency.cpu()
                    aggregate = saliency
                else:
                    saliency = saliency.cpu()
                    b, h, w = saliency.shape
                    saliency = saliency.reshape((b, h * w))

                    if args.relu:
                        saliency = torch.nn.functional.relu(saliency)

                    if args.normalize:
                        saliency -= torch.mean(saliency, dim=0)
                        # saliency /= torch.std(saliency, dim=0)[0]

                    aggregate = function(saliency, dim=-1)

                if args.negate:
                    aggregate = -1 * aggregate

                if isinstance(aggregate, np.ndarray):
                    aggregate = torch.tensor(aggregate)

                if args.show_scores:
                    logits = score_dict[key][second_key]
                    softmaxes = torch.nn.functional.softmax(logits, dim=1)
                    score = torch.max(logits, dim=1)[0]
                    softmax_score = torch.max(softmaxes, dim=1)[0]

                if key == 'id':
                    id_aggregate = aggregate
                    if not args.dont_plot:
                        plot(aggregate, f'{key}: {second_key}'.upper())

                    if args.show_scores:
                        id_score = score
                        id_softmax_score = softmax_score
                        plot(score, f'{key}: {second_key} scores')

                elif key == 'near':
                    auc = calculate_auc(id_aggregate, aggregate)
                    near_aucs.append(auc)

                    if not args.dont_plot:
                        if args.auc:
                            plot(
                                aggregate,
                                f'{key}: {second_key}, {auc=:.2f}'.upper().replace(
                                    '_', '\_'
                                ),
                            )
                        else:
                            plot(
                                aggregate,
                                f'{key}: {second_key}'.upper().replace('_', '\_'),
                            )

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)
                        softmax_score_auc = calculate_auc(
                            id_softmax_score, softmax_score
                        )

                        near_msp_aucs.append(softmax_score_auc)
                        near_mls_aucs.append(score_auc)
                        if not args.dont_plot:
                            plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

                elif key == 'far':
                    auc = calculate_auc(id_aggregate, aggregate)
                    far_aucs.append(auc)

                    if not args.dont_plot:
                        if args.auc:
                            plot(
                                aggregate,
                                f'{key}: {second_key}, {auc=:.2f}'.upper().replace(
                                    '_', '\_'
                                ),
                            )
                        else:
                            plot(
                                aggregate,
                                f'{key}: {second_key}'.upper().replace('_', '\_'),
                            )

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)

                        softmax_score_auc = calculate_auc(
                            id_softmax_score, softmax_score
                        )

                        far_msp_aucs.append(softmax_score_auc)
                        far_mls_aucs.append(score_auc)
                        if not args.dont_plot:
                            plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

    near_auc = np.array(near_aucs).mean()
    far_auc = np.array(far_aucs).mean()

    if args.dont_plot:
        spaces = 11 - len(name)
        if args.show_scores:
            near_mls_scores = np.array(near_mls_aucs).mean()
            far_mls_scores = np.array(far_mls_aucs).mean()

            near_msp_scores = np.array(near_msp_aucs).mean()
            far_msp_scores = np.array(far_msp_aucs).mean()
            if bogo:
                print(
                    f'Max Logit:   Near-AUC {near_mls_scores:.3f}   Far-AUC {far_mls_scores:.3f}'
                )
                print(
                    f'Max Softmax: Near-AUC {near_msp_scores:.3f}   Far-AUC {far_msp_scores:.3f}\n'
                )
                bogo = False
        if near_auc > 0.5:
            print(
                f'{name}:{" "*spaces} Near-AUC {near_auc:.3f}   Far-AUC {far_auc:.3f}'
            )
        else:
            print(
                f'{name}:{" "*spaces} Near-AUC {1 - near_auc:.3f}   Far-AUC {1 - far_auc:.3f}     (if negated)'
            )

        continue

    if not args.pgf:
        if args.auc:
            plt.title(
                f'{name.capitalize()} for {args.generator.capitalize()} saliencies on {args.dataset.capitalize()}.\nNear AUC {near_auc:.2f}, Far AUC {far_auc:.2f}'
            )
        else:
            plt.title(
                f'{name.capitalize()} for {args.generator.capitalize()} saliencies on {args.dataset.capitalize()}.'
            )
    plt.xlabel(name)
    # plt.grid()
    # plt.legend(loc='lower right')
    plt.legend(fontsize='x-small')

    if args.pgf:
        plt.savefig(
            f'../master/figure/{args.dataset}_{args.generator}_{name.lower()}.pgf'
        )
        plt.clf()
    else:
        plt.show()
