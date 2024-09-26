import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys


filename = f'saved_metrics/{sys.argv[1]}.pkl'

plt.rcParams.update({'font.size': 22})

with open(filename, 'rb') as file:
    scores = pickle.load(file)

id_scores = scores['id']['test'][1]

near_ood_scores = np.array([])
for key in scores['ood']['near']:
    near_ood_scores = np.concatenate([near_ood_scores, scores['ood']['near'][key][1]])

far_ood_scores = np.array([])
for key in scores['ood']['far']:
    far_ood_scores = np.concatenate([far_ood_scores, scores['ood']['far'][key][1]])

bins = 1000
# plt.hist(id_scores, bins=bins, histtype='step', density=True, label='id')
# plt.hist(near_ood_scores, bins=bins, histtype='step', density=True, label='near ood')
# plt.hist(far_ood_scores, bins=bins, histtype='step', density=True, label='far ood')
smoothing = 0.1
width = 3

sns.kdeplot(id_scores, bw_method=smoothing, label='id', linewidth=3)
sns.kdeplot(near_ood_scores, bw_method=smoothing, label='near ood', linewidth=3)
sns.kdeplot(far_ood_scores, bw_method=smoothing, label='far ood', linewidth=3)
plt.title(f'Density plot for {sys.argv[1]} method')
plt.xlabel(f'Value of metric used for separation')
plt.legend()
plt.show()
