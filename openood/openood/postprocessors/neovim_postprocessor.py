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
            cam = GradCAMNoRescale(model=net, target_layers=target_layers)

            # with torch.no_grad():
            #     self.w, self.b = net.get_fc()
            #     print('Extracting id training feature')
            #     feature_id_train = []
            #     for batch in tqdm(
            #         id_loader_dict['train'], desc='Setup: ', position=0, leave=True
            #     ):
            #         data = batch['data'].cuda()
            #         data = data.float()
            #         _, feature = net(data, return_feature=True)
            #         feature_id_train.append(feature.cpu().numpy())
            #     feature_id_train = np.concatenate(feature_id_train, axis=0)
            #     logit_id_train = feature_id_train @ self.w.T + self.b

            self.w, self.b = net.get_fc()
            print('Extracting id training feature')
            feature_id_train = []
            for batch in tqdm(
                id_loader_dict['train'], desc='Setup: ', position=0, leave=True
            ):
                data = batch['data'].cuda()
                data = data.float()
                with torch.no_grad():
                    _, feature = net(data, return_feature=True)
                feature_id_train.append(feature.detach().cpu().numpy())

                grayscale_cam = cam(input_tensor=data)
                b, h, w = grayscale_cam.shape
                cams.append(grayscale_cam.reshape((b, h * w)))

            gradcam_id_train = np.concatenate(cams, axis=0)
            print(gradcam_id_train.shape)
            feature_id_train = np.concatenate(feature_id_train, axis=0)
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
        with torch.no_grad():
            _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()

        target_layers = [net.layer4[-1]]
        cam = GradCAMNoRescale(model=net, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=data)
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


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.io import read_image
# from PIL import Image
# from tqdm import tqdm
# from sklearn.covariance import EmpiricalCovariance
# from pytorch_grad_cam import GradCAM
# from typing import Callable, List, Tuple, Optional
# from src.lrp import LRPModel
#
#
# import matplotlib
#
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt
#
# from scipy.special import logsumexp, softmax
#
# from numpy.linalg import norm, pinv
#
# def fpr_recall(ind_conf, ood_conf, tpr):
#     num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
#     num_ood = len(ood_conf)
#     fpr = num_fp / max(1, num_ood)
#     return fpr, thresh
#
# def num_fp_at_recall(ind_conf, ood_conf, tpr):
#     num_ind = len(ind_conf)
#
#     if num_ind == 0 and len(ood_conf) == 0:
#         return 0, 0.0
#     if num_ind == 0:
#         return 0, np.max(ood_conf) + 1
#
#     recall_num = int(np.floor(tpr * num_ind))
#     thresh = np.sort(ind_conf)[-recall_num]
#     num_fp = np.sum(ood_conf >= thresh)
#     return num_fp, thresh
#
#
# class GradCAMNoRescale(GradCAM):
#     # Class that removes rescaling and just the dim of the conv layer
#     def __init__(self, model, target_layers, reshape_transform=None):
#         super(GradCAMNoRescale, self).__init__(
#             model,
#             target_layers,
#             reshape_transform
#         )
#
#     def compute_cam_per_layer(
#             self,
#             input_tensor: torch.Tensor,
#             targets: List[torch.nn.Module],
#             eigen_smooth: bool) -> np.ndarray:
#         activations_list = [a.cpu().data.numpy()
#                             for a in self.activations_and_grads.activations]
#         grads_list = [g.cpu().data.numpy()
#                       for g in self.activations_and_grads.gradients]
#         target_size = self.get_target_width_height(input_tensor)
#
#         cam_per_target_layer = []
#         # Loop over the saliency image from every layer
#         for i in range(len(self.target_layers)):
#             target_layer = self.target_layers[i]
#             layer_activations = None
#             layer_grads = None
#             if i < len(activations_list):
#                 layer_activations = activations_list[i]
#             if i < len(grads_list):
#                 layer_grads = grads_list[i]
#
#             cam = self.get_cam_image(input_tensor,
#                                      target_layer,
#                                      targets,
#                                      layer_activations,
#                                      layer_grads,
#                                      eigen_smooth)
#             cam = np.maximum(cam, 0)
#             # scaled = scale_cam_image(cam, target_size)
#             cam_per_target_layer.append(cam[:, None, :])
#
#         return cam_per_target_layer
#
#
#
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, path, transform=None):
#         self.root_dir = root_dir
#
#         if transform is None:
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ]
#             )
#         else:
#             self.transform = transform
#
#         self.data = list()  # list of tuples (image_file, label)
#
#         dir_path = os.path.join(root_dir, path)
#
#         for img_file in os.listdir(dir_path):
#             self.data.append(os.path.join(dir_path, img_file))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         img_path = self.data[index]
#         image = Image.open(img_path)  # Using PIL to open image
#
#         if image.mode != "RGB":
#             image = image.convert("RGB")
#
#         image = self.transform(image)
#
#         return image
#
#
# def save_features(model, loader, save_location, device="cuda"):
#     features = list()
#
#     hook_func = lambda model, input, output : features.append(output.squeeze().detach())
#
#     hook = model.avgpool.register_forward_hook(hook_func)
#
#     for i, data in enumerate(tqdm(loader)):
#         batch = data.to(device)
#
#         batch = batch.to(device)
#
#
#         model.eval()
#         with torch.no_grad():
#             model(batch)
#
#     hook.remove()
#
#     features = torch.cat(features)
#
#     torch.save(features, save_location)
#
# def save_gradcams(model, loader, save_location, device="cuda"):
#
#     cams = list()
#
#     target_layers = [model.layer4[-1]]
#     cam = GradCAMNoRescale(model=model, target_layers=target_layers)
#
#     for i, data in enumerate(tqdm(loader)):
#         batch = data.to(device)
#
#         batch = batch.to(device)
#
#         grayscale_cam = cam(input_tensor=batch)
#         b, h, w = grayscale_cam.shape
#         cams.append(grayscale_cam.reshape((b, h * w)))
#
#
#     cams = torch.tensor(np.concatenate(cams))
#     torch.save(cams, save_location)
#
# def save_lrps(model, loader, save_location):
#     cams = list()
#
#     # target_layers = [model.layer4[-1]]
#     lrp_model = LRPModel(model)
#
#     for i, data in enumerate(tqdm(loader)):
#         batch = data.to(device)
#
#         batch = batch.to(device)
#
#         relevances = lrp_model.forward(batch)
#
#         break
#
#
#
#
#
# def vim(model, feature_id_train, feature_id_val, feature_ood):
#     w = model.fc.weight.data.cpu().numpy()
#     b = model.fc.bias.data.cpu().numpy()
#
#     logit_id_train = feature_id_train @ w.T + b
#     logit_id_val = feature_id_val @ w.T + b
#     logit_ood = feature_ood @ w.T + b
#
#     logit_id_train = np.array(logit_id_train)
#     logit_id_val = np.array(logit_id_val)
#     logit_ood = np.array(logit_ood)
#
#     # print(logit_id_train)
#     # print(logit_id_val)
#     # print(logit_ood)
#
#
#     # origin point with moore penrose inverse
#     u = - (pinv(w) @ b)
#
#     DIM = 256
#
#     print("computing principal space...")
#     ec = EmpiricalCovariance(assume_centered=True)
#     ec.fit(feature_id_train - u)
#     eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#
#     null_space = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
#
#     print("computing alpha...")
#     vlogit_id_train = norm(np.matmul(feature_id_train - u, null_space), axis=-1)
#
#
#     # print(vlogit_id_train)
#     alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
#     print(f"{alpha=:.4f}")
#
#     vlogit_id_val = norm(np.matmul(feature_id_val - u, null_space), axis=-1) * alpha
#     energy_id_val = logsumexp(logit_id_val, axis=-1)
#     score_id = -vlogit_id_val + energy_id_val
#
#     energy_ood = logsumexp(logit_ood, axis=-1)
#     vlogit_ood = norm(np.matmul(feature_ood - u, null_space), axis=-1) * alpha
#     score_ood = -vlogit_ood + energy_ood
#     fpr_ood, _ = fpr_recall(score_id, score_ood, 0.95)
#
#     # plt.hist(vlogit_id_val, bins=100, density=True, histtype='step')
#     # plt.hist(vlogit_ood, bins=100, density=True, alpha=0.5, histtype='step')
#     # plt.show()
#
#     print(f"FPR95: {fpr_ood:.2%}")
#
# def neovim(model, feature_id_train, feature_id_val, feature_ood, gradcam_id_train, gradcam_id_val, gradcam_ood):
#     w = model.fc.weight.data.cpu().numpy()
#     b = model.fc.bias.data.cpu().numpy()
#
#     logit_id_train = feature_id_train @ w.T + b
#     logit_id_val = feature_id_val @ w.T + b
#     logit_ood = feature_ood @ w.T + b
#
#     logit_id_train = np.array(logit_id_train)
#     logit_id_val = np.array(logit_id_val)
#     logit_ood = np.array(logit_ood)
#
#     stacked_id_train = np.hstack([logit_id_train, gradcam_id_train.numpy()])
#     stacked_id_val = np.hstack([logit_id_val, gradcam_id_val.numpy()])
#     stacked_ood = np.hstack([logit_ood, gradcam_ood.numpy()])
#
#
#     DIM = (stacked_id_train.shape[-1] - 49) // 2
#
#     print("computing principal space...")
#     ec = EmpiricalCovariance(assume_centered=False)
#     ec.fit(stacked_id_train)
#     eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#
#     null_space = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
#
#     print("computing alpha...")
#     vlogit_id_train = norm(np.matmul(stacked_id_train, null_space), axis=-1)
#
#
#     alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
#     print(f"{alpha=:.4f}")
#
#     vlogit_id_val = norm(np.matmul(stacked_id_val, null_space), axis=-1) * alpha
#     energy_id_val = logsumexp(logit_id_val, axis=-1)
#     score_id = -vlogit_id_val + energy_id_val
#
#
#
#     energy_ood = logsumexp(logit_ood, axis=-1)
#     vlogit_ood = norm(np.matmul(stacked_ood, null_space), axis=-1) * alpha
#     score_ood = -vlogit_ood + energy_ood
#     fpr_ood, _ = fpr_recall(score_id, score_ood, 0.95)
#
#     # plt.title("logits neovim")
#     # plt.hist(vlogit_id_val, bins=100, density=True, histtype='step')
#     # plt.hist(vlogit_ood, bins=100, density=True, alpha=0.5, histtype='step')
#     # plt.show()
#
#     # plt.title("gradcams")
#     # plt.hist(gradcam_id_train.mean(dim=1), bins=100, density=True, histtype='step')
#     # plt.hist(gradcam_ood.mean(dim=1), bins=100, density=True, histtype='step')
#     # plt.show()
#
#     print(f"FPR95: {fpr_ood:.2%}")
#
#
#
