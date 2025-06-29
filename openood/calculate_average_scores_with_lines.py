import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse
import torch
import copy
import scipy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils import calculate_auc, get_palette
import utils
import matplotlib
from utils import prettify, pp


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--negate', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args(sys.argv[1:])


sns.set_palette(get_palette())

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

generators = ['lime', 'occlusion', 'gradcam', 'integratedgradients', 'gbp']

results_dict_list = list()
for generator in generators:
    once = True
    results_dict = dict()
    filename = f'saved_saliencies/{args.dataset}_{generator}_{args.repeats}.pkl'
    with open(filename, 'rb') as file:
        saliency_dict = pickle.load(file)

    score_filename = f'saved_scores/{args.dataset}.pkl'
    with open(score_filename, 'rb') as file:
        score_dict = pickle.load(file)

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

                    soft_correlation = np.corrcoef(
                        softmax_score.cpu(), aggregate.cpu()
                    )[0, 1]
                    correlation = np.corrcoef(score.cpu(), aggregate.cpu())[0, 1]
                    correlations.append(correlation)
                    soft_correlations.append(soft_correlation)

                    if key == 'id':
                        id_aggregate = aggregate

                        id_score = score
                        id_softmax_score = softmax_score

                    elif key == 'near':
                        auc = calculate_auc(id_aggregate, aggregate)
                        near_aucs.append(auc)

                        score_auc = calculate_auc(id_score, score)
                        softmax_score_auc = calculate_auc(
                            id_softmax_score, softmax_score
                        )

                        near_msp_aucs.append(softmax_score_auc)
                        near_mls_aucs.append(score_auc)

                    elif key == 'far':
                        auc = calculate_auc(id_aggregate, aggregate)
                        far_aucs.append(auc)

                        score_auc = calculate_auc(id_score, score)

                        softmax_score_auc = calculate_auc(
                            id_softmax_score, softmax_score
                        )

                        far_msp_aucs.append(softmax_score_auc)
                        far_mls_aucs.append(score_auc)

                    # print(np.corrcoef(score.cpu(), aggregate.cpu()))

        correlations = np.array(correlations).mean()
        soft_correlations = np.array(soft_correlations).mean()
        near_auc = np.array(near_aucs).mean()
        far_auc = np.array(far_aucs).mean()

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

        spaces = 11 - len(name)

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

            results_dict[name] = {
                'near': (1 - near_auc) * 100,
                'far': (1 - far_auc) * 100,
                'corr': correlations,
                'soft_corr': soft_correlations,
            }
    results_dict_list.append(results_dict)

mean_values_dict = copy.deepcopy(results_dict_list[0])

for key in results_dict_list[0]:
    for second_key in results_dict_list[0][key]:
        values = [
            results_dict_list[i][key][second_key] for i in range(len(results_dict_list))
        ]

        mean_values_dict[key][second_key] = np.array(values).mean()

for key in mean_values_dict:
    mean_values_dict[key]['mean'] = (
        mean_values_dict[key]['near'] + mean_values_dict[key]['far']
    ) / 2

print(mean_values_dict)

results_dict = mean_values_dict

results_dict2 = copy.deepcopy(results_dict)

results_dict2.pop('\\ac{mls}')
results_dict2.pop('\\ac{msp}')

max_near = max(results_dict2, key=lambda name: results_dict2[name]['near'])
max_far = max(results_dict2, key=lambda name: results_dict2[name]['far'])
max_mean = max(results_dict2, key=lambda name: results_dict2[name]['mean'])

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

mean = 'Mean \\ac{auroc} ' + ''.join(
    [
        f'&\\textbf{{ {results_dict[name]["mean"]:.1f} }}'
        if name == max_mean
        else f'& {results_dict[name]["mean"]:.1f} '
        for name in results_dict
    ]
)


latex_table = f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |p{{5.1em}}|c c|c c c c c c|c c c| }}
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
    {mean} \\\\
    \hline
    {corr} \\\\
    \hline
    {soft_corr} \\\\
    \hline
    \end{{tabular}}
    \caption[Average \\ac{{auroc}} scores over all \\ac{{xai}} saliency methods on {pp(args.dataset)}]{{Average \\ac{{auroc}} scores for all \\ac{{xai}} saliency methods on {pp(args.dataset)}. The highest non-baseline value for Near- and Far-\\ac{{ood}} is highlighted in bold. $\downarrow$ denotes that \\ac{{id}} data points more often have a lower score with this aggregation, and thus the output values have been negated (as described in section \\ref{{section:aurocfpr95}})}}
    \label{{table:{args.dataset}_all_metrics}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}
"""
print(latex_table)

mean_values_dict.pop('CV')
mean_values_dict.pop('RMD')
mean_values_dict.pop('QCD')

baseline_best_near = (
    max(mean_values_dict['\\ac{mls}']['near'], mean_values_dict['\\ac{msp}']['near'])
    / 100
)
baseline_best_far = (
    max(mean_values_dict['\\ac{mls}']['far'], mean_values_dict['\\ac{msp}']['far'])
    / 100
)

labels = list(mean_values_dict.keys())[2:]
nears = [mean_values_dict[key]['near'] / 100 for key in labels]
fars = [mean_values_dict[key]['far'] / 100 for key in labels]

avg = np.array(nears) * 0.5 + np.array(fars) * 0.5

x_ticks = np.arange(len(labels))


plt.rc('axes', axisbelow=True)
# plt.axvspan(-0.5, 1.5, color='gray', alpha=0.2, label='Baselines')
plt.bar(x_ticks - 0.2, nears, 0.4, label='Near-OOD', color=get_palette()[1])
plt.bar(x_ticks + 0.2, fars, 0.4, label='Far-OOD', color=get_palette()[2])
# plt.scatter(x_ticks, avg, label='Average', color=get_palette()[3])
plt.axhline(
    y=baseline_best_near,
    color=get_palette()[1],
    linestyle='--',
    label='Near-OOD Baseline',
)
plt.axhline(
    y=baseline_best_far,
    color=get_palette()[2],
    linestyle='--',
    label='Far-OOD Baseline',
)
plt.ylim(bottom=0.5, top=1)
plt.grid(axis='y', linestyle='--')

plt.xticks(x_ticks, labels)
plt.xlabel('Aggregate')
plt.ylabel('AUROC')


plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 3, 0, 1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])


if args.pgf:
    plt.savefig(f'../master/figure/{args.dataset}_all_metrics_barplot.pgf')
    plt.clf()

elif args.plot:
    plt.show()


nears = list()
fars = list()

for result_dict in results_dict_list:
    results_dict2 = copy.deepcopy(result_dict)

    results_dict2.pop('\\ac{mls}')
    results_dict2.pop('\\ac{msp}')

    max_near = max(results_dict2, key=lambda name: results_dict2[name]['near'])
    max_far = max(results_dict2, key=lambda name: results_dict2[name]['far'])

    nears.append(result_dict[max_near]['near'] / 100)
    fars.append(result_dict[max_far]['far'] / 100)

labels = ['MLS', 'MSP', 'LIME', 'Occlusion', 'GradCAM', 'IG', 'GBP']
labels = ['LIME', 'Occlusion', 'GradCAM', 'IG', 'GBP']
print(nears)
print(fars)
# nears = [
#     results_dict_list[0]['\\ac{mls}']['near'] / 100,
#     results_dict_list[0]['\\ac{mls}']['near'] / 100,
# ] + nears
# fars = [
#     results_dict_list[0]['\\ac{mls}']['far'] / 100,
#     results_dict_list[0]['\\ac{mls}']['far'] / 100,
# ] + fars

x_ticks = np.arange(len(labels))


plt.rc('axes', axisbelow=True)
# plt.axvspan(-0.5, 1.5, color='gray', alpha=0.2, label='Baselines')
plt.bar(x_ticks - 0.2, nears, 0.4, label='Near-OOD', color=get_palette()[1])
plt.bar(x_ticks + 0.2, fars, 0.4, label='Far-OOD', color=get_palette()[2])
# plt.scatter(x_ticks, avg, label='Average', color=get_palette()[3])
plt.ylim(bottom=0.5, top=1)
plt.grid(axis='y', linestyle='--')

plt.xticks(x_ticks, labels)
plt.xlabel('XAI method')
plt.ylabel('AUROC')

plt.axhline(
    y=baseline_best_near,
    # color=get_palette()[1],
    color='black',
    linestyle='--',
    label='Near-OOD Baseline',
)
plt.axhline(
    y=baseline_best_far,
    color='black',
    linestyle=':',
    label='Far-OOD Baseline',
)


handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 3, 0, 1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])


if args.pgf:
    plt.savefig(f'../master/figure/{args.dataset}_all_metrics_barplot.pgf')
    plt.clf()

elif args.plot:
    plt.show()
