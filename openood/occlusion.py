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

from skimage.segmentation import slic
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
    visualize_borders,
)
import pickle
import matplotlib.cm as cm
import shap
import captum

id_name = sys.argv[1]
device = 'cuda'
batch_size = 8

dataloaders = get_dataloaders(id_name, batch_size=batch_size)

plt.rcParams.update({'font.size': 16})


net = get_network(id_name)

id_batch = next(dataloaders['id'][0])
ood_batch = next(dataloaders['far'][0])

id_data = id_batch['data'].cuda()
ood_data = ood_batch['data'].cuda()

# display_pytorch_image(id_data[0])
# seg = slic(id_data[0].permute(1, 2, 0).cpu().numpy())
# visualize_borders(seg)
# plt.show()
id_preds = torch.argmax(net(id_data), dim=1)
ood_preds = torch.argmax(net(ood_data), dim=1)


ablator = captum.attr.GradientShap(net)
baselines = torch.randn(20, 3, 224, 224)
shap_attr_id = (
    ablator.attribute(id_data, baselines.cuda(), target=id_preds).mean(dim=1).cpu()
)
shap_attr_ood = (
    ablator.attribute(ood_data, baselines.cuda(), target=ood_preds).mean(dim=1).cpu()
)

ablator = captum.attr.LRP(net)
lrp_attr_id = ablator.attribute(id_data, target=id_preds).mean(dim=1).detach().cpu()
lrp_attr_ood = ablator.attribute(ood_data, target=ood_preds).mean(dim=1).detach().cpu()

segmentations = list()
for i in range(batch_size):
    seg = slic(id_data[i].permute(1, 2, 0).cpu().numpy())
    segmentations.append(seg)

id_segmentations = torch.from_numpy(np.stack(segmentations)).unsqueeze(dim=1).cuda()


segmentations = list()
for i in range(batch_size):
    seg = slic(ood_data[i].permute(1, 2, 0).cpu().numpy())
    segmentations.append(seg)

ood_segmentations = torch.from_numpy(np.stack(segmentations)).unsqueeze(dim=1).cuda()


ablator = captum.attr.FeatureAblation(net)
ablation_attr_id = (
    ablator.attribute(id_data, target=id_preds, feature_mask=id_segmentations)
    .detach()
    .cpu()
    .mean(dim=1)
)
print(ablation_attr_id)
ablation_attr_ood = (
    ablator.attribute(ood_data, target=ood_preds, feature_mask=ood_segmentations)
    .detach()
    .cpu()
    .mean(dim=1)
)


for i in range(batch_size):
    plt.subplot(241)
    display_pytorch_image(id_data[i])
    plt.subplot(242)
    plt.title(f'{torch.mean(shap_attr_id[i])}')
    plt.imshow(shap_attr_id[i])

    plt.subplot(243)
    plt.title(f'{torch.mean(lrp_attr_id[i])}')
    plt.imshow(lrp_attr_id[i])

    plt.subplot(244)
    plt.title(f'{torch.mean(ablation_attr_id[i])}')
    plt.imshow(ablation_attr_id[i])

    plt.subplot(245)
    display_pytorch_image(ood_data[i])

    plt.subplot(246)
    plt.title(f'{torch.mean(shap_attr_ood[i])}')
    plt.imshow(shap_attr_ood[i])

    plt.subplot(247)
    plt.title(f'{torch.mean(lrp_attr_ood[i])}')
    plt.imshow(lrp_attr_ood[i])

    plt.subplot(248)
    plt.title(f'{torch.mean(ablation_attr_ood[i])}')
    plt.imshow(ablation_attr_ood[i])

    plt.show()
