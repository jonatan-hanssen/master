from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM
from typing import Callable, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dataloaders

plt.rcParams.update({'font.size': 22})

device = 'cuda'

filepath = os.path.dirname(os.path.abspath(__file__))
config_root = os.path.join(filepath, 'configs')

data_root = './data'
id_name = 'cifar10'

preprocessor = get_default_preprocessor(id_name)

loader_kwargs = {
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 8,
}


class GradCAMNoRescale(GradCAM):
    # Class that removes rescaling and just the dim of the conv layer
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAMNoRescale, self).__init__(model, target_layers, reshape_transform)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


dataloaders = get_dataloaders(id_name)

# load the model

net = ResNet18_32x32(num_classes=10)
net.load_state_dict(
    torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
)
net.cuda()
net.eval()


def get_pca(dataloader):
    cams = list()
    pred_labels = list()
    preds = list()

    target_layers = [net.layer3[-1]]
    cam = GradCAMNoRescale(model=net, target_layers=target_layers)

    pbar = tqdm(dataloader[0], total=dataloader[1])

    for i, batch in enumerate(pbar):
        data = batch['data']
        data = data.to(device)

        gradcam = cam(input_tensor=data)
        pred, pred_label = torch.max(cam.outputs, dim=1)
        b, h, w = gradcam.shape
        gradcam = gradcam.reshape((b, h * w))
        cams.append(gradcam)
        pred_labels.append(pred_label.cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        if i > 100:
            pass
            # break

    pca = PCA(n_components=3)
    gradcams = np.concatenate(cams, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    preds = np.concatenate(preds, axis=0)

    preds = np.expand_dims(preds, axis=1)
    pred_labels = np.expand_dims(pred_labels, axis=1)

    data = np.hstack((gradcams, pred_labels, preds))

    pca_data = pca.fit_transform(data)

    return pca_data


id_cams = get_pca(dataloaders['id'])
near_cams = get_pca(dataloaders['near'])
far_cams = get_pca(dataloaders['far'])

scatter = lambda data, label: plt.scatter(data[:, 0], data[:, 1], label=label)

smoothing = 0.1
probs = lambda data, label: sns.kdeplot(
    data, bw_method=smoothing, label=label, linewidth=3
)

plt.subplot(131)
plt.title('PC 1')
probs(id_cams[:, 0], 'id')
probs(near_cams[:, 0], 'near')
probs(far_cams[:, 0], 'far')
plt.legend()

plt.subplot(132)
plt.title('PC 2')
probs(id_cams[:, 1], 'id')
probs(near_cams[:, 1], 'near')
probs(far_cams[:, 1], 'far')
plt.legend()

plt.subplot(133)
plt.title('PC 3')
probs(id_cams[:, 2], 'id')
probs(near_cams[:, 2], 'near')
probs(far_cams[:, 2], 'far')
plt.legend()
plt.show()
