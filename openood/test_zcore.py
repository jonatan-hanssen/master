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
parser.add_argument('--negate', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--linewidth', type=float, default=1.5)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--smoothing', type=float, default=0.3)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--table', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args(sys.argv[1:])

results_dict = dict()

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

score_filename = f'saved_scores/{args.dataset}.pkl'
with open(score_filename, 'rb') as file:
    score_dict = pickle.load(file)


once = True
for name, function in utils.get_aggregate_functions(args.relu):
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

    correlations = list()
    soft_correlations = list()

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

                    if args.normalize:
                        saliency -= torch.mean(saliency, dim=0)
                        # saliency /= torch.std(saliency, dim=0)[0]

                    aggregate = function(saliency, dim=-1)

                if args.negate:
                    aggregate = -1 * aggregate

                if isinstance(aggregate, np.ndarray):
                    aggregate = torch.tensor(aggregate)

                logits = score_dict[key][second_key]
                score = torch.max(logits, dim=1)[0]
                softmaxes = torch.nn.functional.softmax(logits, dim=1)
                softmax_score = torch.max(softmaxes, dim=1)[0]
                # plt.scatter(score, aggregate)
                # plt.show()

                soft_correlation = np.corrcoef(softmax_score.cpu(), aggregate.cpu())[
                    0, 1
                ]
                correlation = np.corrcoef(score.cpu(), aggregate.cpu())[0, 1]
                correlations.append(correlation)
                soft_correlations.append(soft_correlation)

                if key == 'id':
                    id_aggregate = aggregate

                    id_mean = aggregate.mean()
                    id_std = aggregate.std()

                    id_score = score
                    id_score_mean = score.mean()
                    id_score_std = score.std()

                    id_softmax_score = softmax_score
                    plot(score, f'{key}: {second_key} scores')

                else:
                    auc = calculate_auc(
                        (id_aggregate) / id_std + (id_score) / id_score_std,
                        (aggregate) / id_std + (score) / id_score_std,
                    )
                    score_auc = calculate_auc(id_score, score)
                    softmax_score_auc = calculate_auc(id_softmax_score, softmax_score)

                    if key == 'near':
                        near_aucs.append(auc)
                        near_msp_aucs.append(softmax_score_auc)
                        near_mls_aucs.append(score_auc)
                    elif key == 'far':
                        far_aucs.append(auc)
                        far_msp_aucs.append(softmax_score_auc)
                        far_mls_aucs.append(score_auc)

                # print(np.corrcoef(score.cpu(), aggregate.cpu()))

    correlations = np.array(correlations).mean()
    soft_correlations = np.array(soft_correlations).mean()
    near_auc = np.array(near_aucs).mean()
    far_auc = np.array(far_aucs).mean()

    spaces = 11 - len(name)

    near_mls_scores = np.array(near_mls_aucs).mean()
    far_mls_scores = np.array(far_mls_aucs).mean()

    near_msp_scores = np.array(near_msp_aucs).mean()
    far_msp_scores = np.array(far_msp_aucs).mean()

    if once:
        print(
            f'Max Logit:   Near-AUC {near_mls_scores:.3f}   Far-AUC {far_mls_scores:.3f}'
        )
        print(
            f'Max Softmax: Near-AUC {near_msp_scores:.3f}   Far-AUC {far_msp_scores:.3f}\n'
        )
        once = False

        results_dict['\\ac{mls}'] = {
            'near': near_mls_scores * 100,
            'far': far_mls_scores * 100,
            'corr': 2,
            'soft_corr': 2,
        }

        results_dict['\\ac{msp}'] = {
            'near': near_msp_scores * 100,
            'far': far_msp_scores * 100,
            'corr': 2,
            'soft_corr': 2,
        }

    if near_auc > 0.5:
        print(
            f'{name}:{" "*spaces} Near-AUC {near_auc:.3f}   Far-AUC {far_auc:.3f} Corr:  {correlations:.2f}, SCorr:  {soft_correlations:.2f}'
        )

        results_dict[name] = {
            'near': near_auc * 100,
            'far': far_auc * 100,
            'corr': correlations,
            'soft_corr': soft_correlations,
        }
    else:
        print(
            f'{name}:{" "*spaces} Near-AUC {1 - near_auc:.3f}   Far-AUC {1 - far_auc:.3f} Corr: {correlations:.2f}, SCorr: {soft_correlation:.2f}   (if negated)'
        )

        results_dict[f'{name}$\\downarrow$'] = {
            'near': (1 - near_auc) * 100,
            'far': (1 - far_auc) * 100,
            'corr': correlations,
            'soft_corr': soft_correlations,
        }

max_near = max(results_dict, key=lambda name: results_dict[name]['near'])
max_far = max(results_dict, key=lambda name: results_dict[name]['far'])

columns = '|m{5em}|' + 'c|' * len(results_dict)
names = 'Aggregate ' + ''.join([f'& {name} ' for name in results_dict])
near = 'Near-\\ac{ood} \\ac{auroc} ' + ''.join(
    [
        f'&\\textbf{{ {results_dict[name]["near"]:.1f} }}'
        if name == max_near
        else f'& {results_dict[name]["near"]:.1f} '
        for name in results_dict
    ]
)
far = 'Far-\\ac{ood} \\ac{auroc} ' + ''.join(
    [
        f'&\\textbf{{ {results_dict[name]["far"]:.1f} }}'
        if name == max_far
        else f'& {results_dict[name]["far"]:.1f} '
        for name in results_dict
    ]
)
corr = 'Correlation with \\ac{mls}' + ''.join(
    [
        '& - '
        if results_dict[name]['corr'] == 2
        else f'& {results_dict[name]["corr"]:.2f} '
        for name in results_dict
    ]
)
soft_corr = 'Correlation with \\ac{msp}' + ''.join(
    [
        '& - '
        if results_dict[name]['soft_corr'] == 2
        else f'& {results_dict[name]["soft_corr"]:.2f} '
        for name in results_dict
    ]
)


latex_table = f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |p{{5em}}|c c|c c c c c c|c c c| }}
    \hline
     \centering Aggregation type & \multicolumn{{2}}{{c|}}{{Baselines}} & \multicolumn{{6}}{{c|}}{{Magnitude of saliencies}} & \multicolumn{{3}}{{p{{8em}}|}}{{\centering Statistical dispersion}} \\\\
    \hline
    {names} \\\\
    \hline
    \\rowcolor{{near!50}}
    {near} \\\\
    \hline
    \\rowcolor{{far!50}}
    {far} \\\\
    \hline
    {corr} \\\\
    \hline
    {soft_corr} \\\\
    \hline
    \end{{tabular}}
    \caption[\\ac{{auroc}} scores for {args.generator} on {args.dataset}]{{\\ac{{auroc}} scores for {args.generator} on {args.dataset}. The highest value for Near- and Far-\\ac{{ood}} is highlighted in bold. $\downarrow$ denotes that \\ac{{id}} data points more often have a lower score with this aggregation, and thus the output values have been negated (as described in section \\ref{{section:aurocfpr95}})}}
    \label{{table:{args.dataset}_{args.generator}_metrics}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}
"""
if args.table:
    print(latex_table)
