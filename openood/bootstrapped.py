import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, argparse, pickle
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--better_postprocessor', '-b', type=str, default='vim')
parser.add_argument('--worse_postprocessor', '-w', type=str, default='gradmean')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--aggregator', '-a', type=str, default='Norm')

args = parser.parse_args(sys.argv[1:])


def get_metrics(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    names = list(all_metrics[0].index)
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])
    return metrics, names


if args.better_postprocessor == 'salagg':
    better_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_{args.better_postprocessor}_{args.generator}_{args.aggregator}_bootstrapped.pkl'
    )

else:
    better_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_{args.better_postprocessor}_bootstrapped.pkl'
    )
worse_metric, _ = get_metrics(
    f'saved_metrics/{args.dataset}_{args.worse_postprocessor}_bootstrapped.pkl'
)

print(args.better_postprocessor)
print(better_metric.mean(axis=0))

print(args.worse_postprocessor)
print(worse_metric.mean(axis=0))

print(
    f'\n{args.worse_postprocessor} has higher AUROC than {args.better_postprocessor} with the following probabilities (null hypothesis)'
)

for i in range(8):
    better = better_metric[:, i, 1]
    worse = worse_metric[:, i, 1]

    pval = ttest_ind(better, worse, alternative='greater').pvalue

    padding = 10 - len(names[i])

    mean_diff = better.mean() - worse.mean()
    mean_scale = better.mean() / worse.mean()
    print(
        f'{names[i]}: {" "*padding} pval: {pval:.5f}    mean_diff: {" " if mean_diff > 0 else ""}{mean_diff:.3f}%   mean_scale: {mean_scale:.3f}'
    )
