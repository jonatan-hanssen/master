import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys
import torch
from sklearn.decomposition import PCA
import pandas as pd
matplotlib.use('pdf')


filename = 'saved_metrics/lime_betas.pkl'

plt.rcParams.update({'font.size': 22})

with open(filename, 'rb') as file:
    betas = pickle.load(file)

pca = PCA(n_components=5)

id_betas = betas['id']
near_betas = betas['near']
far_betas = betas['far']

summary = lambda data : print(pd.DataFrame(data).describe())


summary(id_betas)
summary(near_betas)
summary(far_betas)



id_stds = torch.std(id_betas, dim=1)
near_stds = torch.std(near_betas, dim=1)
far_stds = torch.std(far_betas, dim=1)

smoothing = 0.3
plot = lambda data, label : sns.kdeplot(data, bw_method=smoothing, label=label, linewidth=3)

id_betas = pca.fit_transform(betas['id'])
near_betas = pca.transform(betas['near'])
far_betas = pca.transform(betas['far'])

summary(id_betas)
summary(near_betas)
summary(far_betas)

plot(id_betas[:, 0], 'id')
plot(near_betas[:, 0], 'near')
plot(far_betas[:, 0], 'far')
plt.xlabel('betas')
plt.title('Hello')
plt.legend()
plt.savefig('temp.png', bbox_inches='tight')

