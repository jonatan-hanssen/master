import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils import calculate_auc

plt.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    '--show_scores', '-s', action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args(sys.argv[1:])

smoothing = 0.3
plot = lambda data, label: sns.kdeplot(
    data, bw_method=smoothing, label=label, linewidth=3
)

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
            score = torch.max(torch.nn.functional.softmax(score, dim=-1), dim=-1)[0]
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


def entropy(saliencies, dim=-1):
    entropy = saliencies - saliencies.min(-1)[0].unsqueeze(1)
    entropy = entropy / entropy.sum(-1).unsqueeze(1)

    entropy = -1 * entropy * torch.log(entropy + 1e-10)
    entropy = entropy.sum(-1)
    return entropy


def gini(saliencies, dim=-1):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    ginies = list()

    for row in saliencies:
        array = row.numpy()
        array = array.flatten()  # all values are treated equally, arrays must be 1d
        if np.amin(array) < 0:
            array -= np.amin(array)  # values cannot be negative
        array += 0.0000001  # values cannot be 0
        array = np.sort(array)  # values must be sorted
        index = np.arange(1, array.shape[0] + 1)  # index per array element
        n = array.shape[0]  # number of array elements
        gini = (np.sum((2 * index - n - 1) * array)) / (
            n * np.sum(array)
        )  # Gini coefficient
        ginies.append(gini)

    return torch.tensor(ginies)


def norm_std(saliencies, dim=-1):
    saliencies = saliencies / saliencies.mean(dim=-1).unsqueeze(1)

    return saliencies.std(dim=-1)


pca = PCA(n_components=1)
did_pca = False


class pca_wrapper:
    def __init__(self):
        self.pca = None

    def __call__(self, saliencies, dim=-1):
        # saliencies = saliencies - torch.min(saliencies, dim=0)[0]
        # saliencies = saliencies / torch.max(saliencies, dim=0)[0]
        if self.pca is None:
            self.pca = PCA(n_components=1)
            return self.pca.fit_transform(saliencies).squeeze()
        else:
            return self.pca.transform(saliencies).squeeze()


class pca_recon_loss:
    def __init__(self):
        self.pca = None

    def __call__(self, saliencies, dim=-1):
        # saliencies = saliencies - torch.min(saliencies, dim=0)[0]
        # saliencies = saliencies / torch.max(saliencies, dim=0)[0]
        if self.pca is None:
            self.pca = PCA(n_components=1)
            self.pca.fit(saliencies)

        recon = self.pca.inverse_transform(self.pca.transform(saliencies))
        recon = torch.tensor(recon)

        recon_loss = torch.mean((saliencies - recon) ** 2, dim=1)

        return recon_loss


# print(score_dict)

functions = (
    ('Mean', torch.mean),
    ('Recon', pca_recon_loss()),
    ('Gini', gini),
    ('Std', torch.std),
    ('PCA', pca_wrapper()),
    ('Max', lambda data, dim: torch.max(data, dim=-1)[0]),
    ('Entropy', norm_std),
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
                    plot(aggregate, f'{key}: {second_key}')

                    if args.show_scores:
                        id_score = score
                        plot(score, f'{key}: {second_key} scores')

                elif key == 'near':
                    auc = calculate_auc(id_aggregate, aggregate)
                    near_aucs.append(auc)

                    plot(aggregate, f'{key}: {second_key}. AUC: {auc:.2f}')

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)
                        plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

                elif key == 'far':
                    auc = calculate_auc(id_aggregate, aggregate)
                    far_aucs.append(auc)
                    plot(aggregate, f'{key}: {second_key}. AUC: {auc:.2f}')

                    if args.show_scores:
                        score_auc = calculate_auc(id_score, score)
                        plot(score, f'{key}: {second_key} scores {score_auc:.2f}')

    near_auc = np.array(near_aucs).mean()
    far_auc = np.array(far_aucs).mean()

    plt.title(
        f'{name} for {args.generator} saliencies on {args.dataset}.\nNear AUC {near_auc:.2f}, far AUC {far_auc:.2f}'
    )
    plt.xlabel(name)
    plt.legend()
    plt.show()
