import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys
import torch
from sklearn.decomposition import PCA
from openTSNE import TSNE
import pandas as pd
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor


filename = 'saved_metrics/lime_betas_per_dataset.pkl'

with open(filename, 'rb') as file:
    betas = pickle.load(file)


ids = torch.vstack([betas['id'][key] for key in betas['id']])
nears = torch.vstack([betas['near'][key] for key in betas['near']])
fars = torch.vstack([betas['far'][key] for key in betas['far']])
print(ids.shape)
exit()

pca = PCA(n_components=64)
tsne = TSNE(
    perplexity=30,
    metric='euclidean',
    n_jobs=8,
    negative_gradient_method='fft',
    random_state=0,
    verbose=True,
)

print('hello')
ids = pca.fit_transform(ids)
print('done')
nears = pca.transform(nears)
fars = pca.transform(fars)

ids = tsne.fit(ids)
nears = tsne.transform(nears)
fars = tsne.transform(fars)

with open('saved_metrics/shit.com', 'wb') as file:
    pickle.dump([ids, nears, fars, tsne], file)

#
# with open('saved_metrics/shit.com', 'rb') as file:
#     ids, nears, fars = pickle.load(file)
#
#
# plot = lambda data, label : plt.scatter(data[:, 0], data[:, 1], s=1, label=label)
#
# plot(ids, 'id')
# plot(nears, 'near')
# plot(fars, 'far')
# plt.legend()
# plt.savefig('temp.png', bbox_inches='tight')
