import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import ttest_ind


def get_metrics(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])
    return metrics


if len(sys.argv) > 1:
    # filename = f'saved_metrics/{sys.argv[1]}.pkl'
    filename = sys.argv[1]
else:
    filename = 'saved_metrics/neovim_bootstrapped2.pkl'

vim = get_metrics(f'saved_metrics/vim_bootstrapped.pkl')
other = get_metrics(filename)
print('vim')
print(vim.mean(axis=0))

print(filename)
print(other.mean(axis=0))

names = [
    'cifar100',
    'tin',
    'nearood',
    'mnist',
    'svhn',
    'texture',
    'places365',
    'farood',
]

for i in range(8):
    a = vim[:, i, 1]
    b = other[:, i, 1]

    pval = ttest_ind(a, b, alternative='less').pvalue

    padding = 10 - len(names[i])

    mean_diff = a.mean() - b.mean()
    mean_scale = a.mean() / b.mean()
    print(
        f'{names[i]}: {" "*padding} pval: {pval:.5f}    mean_diff: {" " if mean_diff > 0 else ""}{mean_diff:.3f}%   mean_scale: {mean_scale:.3f}'
    )
