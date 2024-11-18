import torch
import os
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget as CO
from typing import Callable, List, Tuple, Optional
import numpy as np
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
)
import pickle
import matplotlib.cm as cm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from typing import Callable, List, Tuple, Optional


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
            # print(np.min(cam))
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class GradCAMPlusPlusNoRescale(GradCAMPlusPlus):
    # Class that removes rescaling and just the dim of the conv layer
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAMPlusPlusNoRescale, self).__init__(
            model, target_layers, reshape_transform
        )

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
            # print(np.min(cam))
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return result


id_name = 'imagewoof'
device = 'cuda'
batch_size = 16

dataloaders = get_dataloaders(id_name, batch_size=batch_size)

# load the model

net = get_network(id_name)

repeats = 4
image_size = 224
block_size = image_size // repeats


for i in range(3):
    id_batch = next(dataloaders['id'][0])
    ood_batch = next(dataloaders['far'][0])

    outputs = list()
    for batch in (id_batch, ood_batch):
        data = batch['data'].to(device)

        target_layers = [net.layer4[-1]]
        camm = GradCAMPlusPlusNoRescale(model=net, target_layers=target_layers)

        preds = torch.argmax(net(data), dim=1)
        saliencies = occlusion(net, data, repeats=repeats)
        betas = lime_explanation(net, data, 64, repeats=repeats, kernel_width=0.75)
        cams = torch.from_numpy(camm(data))
        cam_block_size = image_size // cams.shape[-1]

        occlusions = (
            saliencies.reshape((batch_size, repeats, repeats))
            .repeat_interleave(block_size, dim=1)
            .repeat_interleave(block_size, dim=2)
        )

        gradcams = cams.repeat_interleave(cam_block_size, dim=1).repeat_interleave(
            cam_block_size, dim=2
        )

        limes = (
            betas.reshape((batch_size, repeats, repeats))
            .repeat_interleave(block_size, dim=1)
            .repeat_interleave(block_size, dim=2)
        )
        outputs.append([data, occlusions, limes, gradcams, batch['label'], preds])

    maxval_gradcam = None
    id_bool = True
    for i in range(16):
        if id_bool:
            print(True)
            values = outputs[0]
        else:
            values = outputs[1]

        img = values[0][i]
        occlusion_sal = values[1][i]
        lime_sal = values[2][i]
        gradcam_sal = values[3][i]
        label = values[4][i]
        pred = values[5][i]

        if not id_bool:
            maxval_gradcam = None

        plt.subplot(221)
        # plt.title(f'Original img, gt {hyperkvasir[label]} pred {hyperkvasir[pred]}')
        display_pytorch_image(img)
        plt.subplot(222)
        overlay_saliency(img, gradcam_sal, 'gradcam', maxval_gradcam)
        plt.subplot(223)
        overlay_saliency(img, lime_sal, 'lime')
        plt.subplot(224)
        overlay_saliency(img, occlusion_sal, 'occlusion')

        maxval_gradcam = np.abs(np.max(numpify(gradcam_sal)))

        plt.tight_layout()
        plt.show()

        id_bool = not id_bool
