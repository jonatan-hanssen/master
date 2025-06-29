import pickle
import numpy as np


def get_metrics(dataset):
    filename = f'saved_metrics/{dataset}_mls_bootstrapped.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        dataframe = data[0][0]

    filename = f'saved_metrics/{dataset}_salpluslogit_gbpnored_norm_bootstrapped.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])

    dataframe.values[:] = metrics.mean(axis=0)
    near_auroc, far_auroc = dataframe.loc[['nearood', 'farood'], 'AUROC'].values / 100

    print(dataset, near_auroc, far_auroc)

    return near_auroc, far_auroc


get_metrics('imagenet200')
get_metrics('imagenet')
get_metrics('cifar10')
get_metrics('cifar100')


def get_metrics(dataset):
    filename = f'saved_metrics/{dataset}_salpluslogit_gbpnored_norm.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data

    near_auroc, far_auroc = all_metrics.loc[['nearood', 'farood'], 'AUROC'].values / 100

    print(dataset, near_auroc, far_auroc)

    return near_auroc, far_auroc


print()

get_metrics('imagenet200')
get_metrics('cifar10')
get_metrics('imagenet')
get_metrics('cifar100')
