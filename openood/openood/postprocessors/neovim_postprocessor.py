from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

from .xai_utils import GradCAMWrapper


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

                b, c, h, w = grayscale_cam.shape
                cams.append(grayscale_cam.reshape((b, c * h * w)))

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
        b, c, h, w = grayscale_cam.shape
        gradcam_ood = grayscale_cam.reshape((b, c * h * w))

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
