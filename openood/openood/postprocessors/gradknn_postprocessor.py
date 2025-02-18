from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


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

        idxs = torch.argsort(preds, dim=1)

        highest = preds[torch.arange(batch_size), idxs[:, -1]]
        # second_highest = preds[torch.arange(batch_size), idxs[:, -2]]
        # third_highest = preds[torch.arange(batch_size), idxs[:, -3]]

        self.model.zero_grad(set_to_none=True)
        # backward pass, this gets gradients for each prediction
        torch.sum(preds[torch.arange(batch_size), idxs[:, -1]]).backward(
            retain_graph=False
        )

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


class GradKNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(GradKNNPostprocessor, self).__init__(config)
        # self.args = self.config.postprocessor.postprocessor_args
        # self.args_dict = self.config.postprocessor.postprocessor_sweep
        # self.K = self.args.K

        self.activation_log = None
        self.setup_flag = False
        self.K = 50
        self.APS_mode = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()

            target_layers = [net.layer4[-1]]
            # cam = GradCAMNoRescale(model=net, target_layers=target_layers)
            wrapper = GradCAMWrapper(model=net, target_layer=target_layers[0])

            cams = list()

            for batch in tqdm(
                id_loader_dict['train'], desc='Setup: ', position=0, leave=True
            ):
                data = batch['data'].cuda()
                data = data.float()

                cam, feature = wrapper(data, return_feature=True)
                b, h, w = cam.shape
                cam = cam.reshape((b, h * w))
                activation_log.append(
                    normalizer(np.hstack((feature.data.cpu().numpy(), cam)))
                )

            self.activation_log = np.concatenate(activation_log, axis=0)
            print(self.activation_log.shape)
            self.index = faiss.IndexFlatL2(self.activation_log.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        target_layers = [net.layer4[-1]]
        wrapper = GradCAMWrapper(model=net, target_layer=target_layers[0])

        output, feature = net(data, return_feature=True)
        cam = wrapper(data)
        b, h, w = cam.shape
        cam = cam.reshape((b, h * w))
        feature_normed = normalizer(np.hstack((feature.data.cpu().numpy(), cam)))
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
