from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.cluster import KMeans
from tqdm import tqdm
from time import time

from .base_postprocessor import BasePostprocessor

from pytorch_grad_cam import GradCAM
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
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class TestPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.APS_mode = False
        self.clusterers = None
        self.num_clusters = 16

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            target_layers = [net.layer3[-1]]

            cam = GradCAMNoRescale(model=net, target_layers=target_layers)

            print('Extracting gradcams from training')

            cams = list()
            preds = list()
            features = list()
            for i, batch in enumerate(
                tqdm(id_loader_dict['train'], desc='Setup: ', position=0, leave=True)
            ):
                data = batch['data'].cuda()
                data = data.float()
                with torch.no_grad():
                    _, feature = net(data, return_feature=True)

                features.append(feature.cpu().numpy())
                grayscale_cam = cam(input_tensor=data)

                n_classes = cam.outputs.shape[-1]

                _, pred = torch.max(cam.outputs, dim=1)
                preds.append(pred.cpu().numpy())

                b, h, w = grayscale_cam.shape
                cams.append(grayscale_cam.reshape((b, h * w)))
                if i > 20:
                    pass

            gradcams = np.concatenate(cams, axis=0)
            preds = np.concatenate(preds, axis=0)
            features = np.concatenate(features, axis=0)

            self.clusterers = list()

            print('Performing clustering on training')
            for i in range(n_classes):
                class_gradcams = gradcams[np.where(preds == i)]
                class_features = features[np.where(preds == i)]

                clusterer = KMeans(random_state=1, n_clusters=self.num_clusters)
                clusterer.fit(np.hstack([class_gradcams, class_features]))
                self.clusterers.append(clusterer)

            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        target_layers = [net.layer3[-1]]
        cam = GradCAMNoRescale(model=net, target_layers=target_layers)
        gradcams = cam(input_tensor=data)
        b, h, w = gradcams.shape
        gradcams = gradcams.reshape((b, h * w))

        with torch.no_grad():
            _, features = net(data, return_feature=True)

        features = features.cpu().numpy()

        _, preds = torch.max(cam.outputs, dim=1)
        preds = preds.cpu().numpy()
        n_classes = cam.outputs.shape[-1]

        distances = np.zeros((b, self.num_clusters))

        for i in range(n_classes):
            indices = np.where(preds == i)
            if len(indices[0]) == 0:
                continue
            distances[indices] = self.clusterers[i].transform(
                np.hstack([gradcams[indices], features[indices]])
            )

        score_ood = np.min(distances, axis=1) * -1

        return torch.from_numpy(preds), torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
