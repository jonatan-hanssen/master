import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse
import torch
from sklearn.decomposition import PCA
import pandas as pd
from utils import calculate_auc

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', type=bool, default=True)

args = parser.parse_args(sys.argv[1:])

filename = f'saved_saliencies/{args.dataset}_{args.generator}_{args.repeats}.pkl'

plt.rcParams.update({'font.size': 22})


def entropy(saliencies, dim=-1):
    entropy = saliencies - saliencies.min(-1)[0].unsqueeze(1)
    entropy = entropy / entropy.sum(-1).unsqueeze(1)

    entropy = -1 * entropy * torch.log(entropy + 1e-10)
    entropy = entropy.sum(-1)
    return entropy


pca = PCA(n_components=1)
did_pca = False


class pca_wrapper:
    def __init__(self):
        self.pca = None

    def __call__(self, saliencies, dim=-1):
        if self.pca is None:
            self.pca = PCA(n_components=1)
            return self.pca.fit_transform(saliencies).squeeze()
        else:
            return self.pca.transform(saliencies).squeeze()


with open(filename, 'rb') as file:
    saliency_dict = pickle.load(file)

functions = (
    ('Mean', torch.mean),
    ('Max', lambda data, dim: torch.max(data, dim=-1)[0]),
    ('Std', torch.std),
    ('Entropy', entropy),
    ('PCA', pca_wrapper()),
)

smoothing = 0.3
plot = lambda data, label: sns.kdeplot(
    data, bw_method=smoothing, label=label, linewidth=3
)

for name, function in functions:
    id_aggregate = None
    near_aucs = list()
    far_aucs = list()
    for key in ('id', 'near', 'far'):
        if isinstance(saliency_dict[key], dict):
            for second_key in saliency_dict[key]:
                saliency = saliency_dict[key][second_key]
                saliency = saliency.cpu()
                if len(saliency.shape) != 2:
                    b, h, w = saliency.shape
                    saliency = saliency.reshape((b, h * w))
                aggregate = function(saliency, dim=-1)

                if isinstance(aggregate, np.ndarray):
                    aggregate = torch.tensor(aggregate)

                if key == 'id':
                    id_aggregate = aggregate
                    plot(aggregate, f'{key}: {second_key}')
                elif key == 'near':
                    auc = calculate_auc(id_aggregate, aggregate)
                    near_aucs.append(auc)
                    plot(aggregate, f'{key}: {second_key}, AUC: {auc:.2f}')

                elif key == 'far':
                    auc = calculate_auc(id_aggregate, aggregate)
                    far_aucs.append(auc)
                    plot(aggregate, f'{key}: {second_key}, AUC: {auc:.2f}')

    near_auc = np.array(near_aucs).mean()
    far_auc = np.array(far_aucs).mean()

    plt.title(
        f'{name} for {args.generator} saliencies on {args.dataset}.\nNear AUC {near_auc:.2f}, far AUC {far_auc:.2f}'
    )
    plt.xlabel(name)
    plt.legend()
    plt.show()
