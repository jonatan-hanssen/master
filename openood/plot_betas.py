import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
from sklearn.decomposition import PCA


filename = 'saved_metrics/lime_betas.pkl'

plt.rcParams.update({'font.size': 22})

with open(filename, 'rb') as file:
    betas = pickle.load(file)

pca = PCA(n_components=5)

id_betas = betas['id']
near_betas = betas['near']
far_betas = betas['far']



id_stds = torch.std(id_betas, dim=1)
near_stds = torch.std(near_betas, dim=1)
far_stds = torch.std(far_betas, dim=1)

smoothing = 0.3
plot = lambda data, label : sns.kdeplot(data, bw_method=smoothing, label=label, linewidth=3)

id_betas = pca.fit_transform(betas['id'])
near_betas = pca.transform(betas['near'])
far_betas = pca.transform(betas['far'])

plot(id_stds, 'id')
plot(near_stds, 'near')
plot(far_stds, 'far')
plt.legend()
plt.show()

