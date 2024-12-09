from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from typing import Callable, List, Tuple, Optional


class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model=None, target_layer=None, do_relu=False):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.do_relu = do_relu

        self.grads = None
        self.acts = None

        self.handles = list()

        self.handles.append(
            self.target_layer.register_full_backward_hook(self.grad_hook)
        )
        self.handles.append(self.target_layer.register_forward_hook(self.act_hook))

    def grad_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0]

    def act_hook(self, module, input, output):
        self.acts = output

    def forward(self, x, return_feature=False):
        batch_size = x.shape[0]

        if return_feature:
            preds, feature = self.model(x, return_feature=True)

        else:
            preds = self.model(x)

        self.model.zero_grad(set_to_none=True)

        idxs = torch.argmax(preds, dim=1)

        # backward pass, this gets gradients for each prediction
        torch.sum(preds[torch.arange(batch_size), idxs]).backward()

        average_gradients = self.grads.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        saliency = self.acts * average_gradients

        saliency = torch.sum(saliency, dim=1)
        if self.do_relu:
            saliency = torch.nn.functional.relu(saliency)

        if return_feature:
            return saliency.cpu().detach(), feature.cpu().detach()

        else:
            return saliency.cpu().detach()

    def __del__(self):
        for handle in self.handles:
            handle.remove()


class GradCAMNoRescale(HiResCAM):
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
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        result = np.mean(cam_per_target_layer, axis=1)
        return result


class NeoVIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            cams = list()

            target_layers = [net.layer4[-1]]
            # cam = GradCAMNoRescale(model=net, target_layers=target_layers)
            cam = GradCAMWrapper(model=net, target_layer=target_layers[0])

            self.w, self.b = net.get_fc()
            print('Extracting id training feature')
            feature_id_train = []
            for batch in tqdm(
                id_loader_dict['train'], desc='Setup: ', position=0, leave=True
            ):
                data = batch['data'].cuda()
                data = data.float()

                # grayscale_cam = cam(input_tensor=data)
                grayscale_cam, feature = cam(data, return_feature=True)

                feature_id_train.append(
                    feature.detach().cpu().numpy().astype('float32')
                )

                b, h, w = grayscale_cam.shape
                cams.append(grayscale_cam.reshape((b, h * w)))

            gradcam_id_train = np.concatenate(cams, axis=0).astype('float32')
            print(f'{gradcam_id_train.shape=}')
            feature_id_train = np.concatenate(feature_id_train, axis=0).astype(
                'float32'
            )
            stacked_feature_id_train = np.hstack([feature_id_train, gradcam_id_train])
            print(stacked_feature_id_train.shape)

            logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            self.u = np.concatenate([self.u, np.zeros(gradcam_id_train.shape[1])])
            print(f'{self.u.shape=}')
            print(f'{stacked_feature_id_train.shape=}')
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(stacked_feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim :]]).T
            )

            vlogit_id_train = norm(
                np.matmul(stacked_feature_id_train - self.u, self.NS), axis=-1
            )
            self.alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        # with torch.no_grad():
        #     _, feature_ood = net.forward(data, return_feature=True)

        target_layers = [net.layer4[-1]]
        # cam = GradCAMNoRescale(model=net, target_layers=target_layers)
        cam = GradCAMWrapper(model=net, target_layer=target_layers[0])
        # grayscale_cam = cam(input_tensor=data)
        grayscale_cam, feature_ood = cam(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        b, h, w = grayscale_cam.shape
        gradcam_ood = grayscale_cam.reshape((b, h * w))

        stacked_feature_ood = np.hstack([feature_ood, gradcam_ood])

        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = (
            norm(np.matmul(stacked_feature_ood - self.u, self.NS), axis=-1) * self.alpha
        )
        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
