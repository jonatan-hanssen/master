import torch
import os, sys
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget as CO
from typing import Callable, List, Tuple, Optional
import numpy as np
import torchvision
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    get_dataloaders,
    display_pytorch_image,
    occlusion,
    numpify,
    lime_explanation,
    overlay_saliency,
    get_network,
    GradCAMWrapper,
)
import pickle
import matplotlib.cm as cm
import shap

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

X, y = shap.datasets.imagenet50()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


dogs = {
    0: 'Shih-Zu',
    1: 'Rhod. Ridgeback',
    2: 'Beagle',
    3: 'Eng. Foxhound',
    4: 'Border_terrier',
    5: 'Aus. Terrier',
    6: 'Golden_retriever',
    7: 'Old_English_sheepdog',
    8: 'Samoyed',
    9: 'Dingo',
}

transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)

id_name = sys.argv[1]
device = 'cuda'
batch_size = 16

dataloaders = get_dataloaders(id_name, batch_size=batch_size)

plt.rcParams.update({'font.size': 16})

# load the model

net = get_network(id_name)


def f(x):
    tmp = x.copy()
    if isinstance(tmp, torch.Tensor) == False:
        tmp = torch.from_numpy(tmp)
    tmp = nhwc_to_nchw(tmp)
    return net(tmp.cuda())


for i in range(3):
    id_batch = next(dataloaders['id'][0])
    ood_batch = next(dataloaders['far'][0])

    for batch in (id_batch, ood_batch):
        data = batch['data'].to(device)

        # define a masker that is used to mask out partitions of the input image.
        masker = shap.maskers.Image('blur(128,128)', data[0].shape)

        # create an explainer with model and image masker
        explainer = shap.Explainer(f, masker, output_names=list(dogs.values()))
        print(type(data))

        # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values

        plt.imshow(X[0] / 255)
        plt.show()
        plt.imshow(data[0])
        plt.show()
        data = data.permute(0, 2, 3, 1)

        print(data.shape)
        print(data.dtype)
        shap_values = explainer(
            data[0].unsqueeze(0),
            max_evals=100,
            batch_size=50,
            outputs=shap.Explanation.argsort.flip[:1],
        )
        print(shap_values)
        shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
        snake = shap_values.values.squeeze()[:, :, 0]
        plt.imshow(snake)
        plt.show()
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

        shap.image_plot(
            shap_values=shap_values.values,
            pixel_values=shap_values.data,
            labels=shap_values.output_names,
            true_labels=['hello hello'],
        )
        exit()
