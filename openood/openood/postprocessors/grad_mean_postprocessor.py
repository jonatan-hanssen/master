from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from tqdm import tqdm
from time import time

from .base_postprocessor import BasePostprocessor


class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model=None, target_layer=None, do_relu=False, subtype=None):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.do_relu = do_relu

        self.subtype = subtype

        self.grads = None
        self.acts = None

        self.outputs = None

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

        self.outputs = preds

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
            return saliency.cpu().detach(), feature

        else:
            return saliency.cpu().detach()

    def __del__(self):
        for handle in self.handles:
            handle.remove()


class GradMeanPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.APS_mode = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        cam_wrapper = GradCAMWrapper(model=net, target_layer=net.layer4[-1])
        gradcams = cam_wrapper(data)
        b, h, w = gradcams.shape
        gradcams = gradcams.reshape((b, h * w))

        preds = torch.argmax(cam_wrapper.outputs, dim=-1).cpu()

        score_ood = torch.mean(gradcams, dim=-1)

        return preds, score_ood

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
