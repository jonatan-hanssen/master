import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, argparse, pickle
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--first_postprocessor', '-f', type=str, default='vim')
parser.add_argument('--second_postprocessor', '-s', type=str, default='gradmean')

args = parser.parse_args(sys.argv[1:])


def get_metrics(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    names = list(all_metrics[0].index)
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])
    return metrics, names


first_metric, names = get_metrics(
    f'saved_metrics/{args.dataset}_{args.first_postprocessor}_bootstrapped.pkl'
)
second_metric, _ = get_metrics(
    f'saved_metrics/{args.dataset}_{args.second_postprocessor}_bootstrapped.pkl'
)

for i in range(8):
    a = first_metric[:, i, 1]
    b = second_metric[:, i, 1]

    pval = ttest_ind(a, b, alternative='less').pvalue

    padding = 10 - len(names[i])

    mean_diff = a.mean() - b.mean()
    mean_scale = a.mean() / b.mean()
    print(
        f'{names[i]}: {" "*padding} pval: {pval:.5f}    mean_diff: {" " if mean_diff > 0 else ""}{mean_diff:.3f}%   mean_scale: {mean_scale:.3f}'
    )
