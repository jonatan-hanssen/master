import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys
import torch
from sklearn.decomposition import PCA
import pandas as pd


filename = f'saved_metrics/{sys.argv[1]}_grads_own.pkl'

plt.rcParams.update({'font.size': 22})


def entropy(saliencies):
    entropy = saliencies - saliencies.min(-1)[0].unsqueeze(1)
    entropy = entropy / entropy.sum(-1).unsqueeze(1)

    entropy = -1 * entropy * torch.log(entropy + 1e-10)
    entropy = entropy.sum(-1)
    return entropy


with open(filename, 'rb') as file:
    grads = pickle.load(file)

pca = PCA(n_components=3)

id_grads = grads[0]
near_grads = grads[1]
far_grads = grads[2]

b, h, w = id_grads.shape
id_grads = id_grads.reshape((b, h * w))

b, h, w = near_grads.shape
near_grads = near_grads.reshape((b, h * w))

b, h, w = far_grads.shape
far_grads = far_grads.reshape((b, h * w))

summary = lambda data: print(pd.DataFrame(data).describe())

id_grads_mean = id_grads.mean(dim=-1)
id_grads_max = id_grads.max(dim=-1)[0]
id_grads_diff = id_grads.max(dim=-1)[0] - id_grads.min(dim=-1)[0]
id_grads_stds = id_grads.std(dim=-1)
id_grads_entropy = entropy(id_grads)

near_grads_mean = near_grads.mean(dim=-1)
near_grads_max = near_grads.max(dim=-1)[0]
near_grads_diff = near_grads.max(dim=-1)[0] - near_grads.min(dim=-1)[0]
near_grads_stds = near_grads.std(dim=-1)
near_grads_entropy = entropy(near_grads)

far_grads_mean = far_grads.mean(dim=-1)
far_grads_max = far_grads.max(dim=-1)[0]
far_grads_diff = far_grads.max(dim=-1)[0] - far_grads.min(dim=-1)[0]
far_grads_stds = far_grads.std(dim=-1)
far_grads_entropy = entropy(far_grads)

summary(id_grads_mean)
summary(near_grads_mean)
summary(far_grads_mean)

print(id_grads_mean.shape)
print(np.vstack((id_grads_mean, id_grads_stds, id_grads_diff)).shape)

id_transformed = pca.fit_transform(
    np.vstack((id_grads_mean, id_grads_stds, id_grads_diff)).T
)

near_transformed = pca.transform(
    np.vstack((near_grads_mean, near_grads_stds, near_grads_diff)).T
)

far_transformed = pca.transform(
    np.vstack((far_grads_mean, far_grads_stds, far_grads_diff)).T
)


id_transformed = pca.fit_transform(id_grads)
near_transformed = pca.transform(near_grads)
far_transformed = pca.transform(far_grads)

# summary(id_grads_mean)
# summary(near_grads_mean)
# summary(far_grads_mean)

# summary(id_grads_max)
# summary(near_grads_max)
# summary(far_grads_max)


smoothing = 0.3
plot = lambda data, label: sns.kdeplot(
    data, bw_method=smoothing, label=label, linewidth=3
)

plot(id_grads_entropy, 'ID: ImageWoof')
plot(near_grads_entropy, 'Near-OOD: Stanford Dogs')
plot(far_grads_entropy, 'Far-OOD: Places365')
plt.xlabel('Entropy')
plt.title('entropy')
plt.legend()
plt.show()

plot(id_transformed[:, 0], 'ID: ImageWoof')
plot(near_transformed[:, 0], 'Near-OOD: Stanford Dogs')
plot(far_transformed[:, 0], 'Far-OOD: Places365')
plt.xlabel('First Principal Component')
plt.title('pca')
plt.legend()
plt.show()

plot(id_grads_stds, 'ID: ImageWoof')
plot(near_grads_stds, 'Near-OOD: Stanford Dogs')
plot(far_grads_stds, 'Far-OOD: Places365')
plt.xlabel('Standard deviation for a given image')
plt.title('Standard deviation of GradCAM saliencies')
plt.legend()
plt.show()

plot(id_grads_diff, 'id')
plot(near_grads_diff, 'near')
plot(far_grads_diff, 'far')
plt.xlabel('grads')
plt.title('Diff')
plt.legend()
plt.show()

plot(id_grads_mean, 'ID: ImageWoof')
plot(near_grads_mean, 'Near-OOD: Stanford Dogs')
plot(far_grads_mean, 'Far-OOD: Places365')
plt.xlabel('Mean saliency for a given image')
plt.title('Mean values of GradCAM saliencies')
plt.legend()
plt.show()

plot(id_grads_max, 'id')
plot(near_grads_max, 'near')
plot(far_grads_max, 'far')
plt.xlabel('Max value for a given image')
plt.title('Max values of GradCAM saliencies')
plt.legend()
plt.show()
