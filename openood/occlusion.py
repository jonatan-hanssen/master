import torch
import os
import torch.nn as nn
from tqdm import tqdm
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
from utils import (
    get_dataloaders,
    display_pytorch_image,
    occlusion,
    numpify,
    lime_explanation,
    overlay_saliency,
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

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return result


id_name = 'hyperkvasir'
device = 'cuda'
batch_size = 16

dataloaders = get_dataloaders(id_name, batch_size=batch_size)

# load the model

net = ResNet18_224x224(num_classes=6)
net.load_state_dict(
    torch.load(
        './results/hyperkvasir_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'
    )
)
net.cuda()
net.eval()

# id_name = 'cifar10'
# device = 'cuda'
#
# dataloaders = get_dataloaders(id_name)
#
# # load the model
#
# net = ResNet18_32x32(num_classes=10)
# net.load_state_dict(
#     torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
# )
# net.cuda()
# net.eval()

hyperkvasir = [
    'cecum',
    'retroflex-rectum',
    'polyps',
    'dyed-lifted-polyps',
    'ulcerative-colitis-grade-2',
    'bbps-0-1',
]


repeats = 8
image_size = 224
block_size = image_size // repeats


for key in ('id', 'near', 'far'):
    pbar = tqdm(dataloaders[key][0], total=dataloaders[key][1])

    target_layers = [net.layer4[-1]]
    camm = GradCAMNoRescale(model=net, target_layers=target_layers)

    for i, batch in enumerate(pbar):
        data = batch['data'].to(device)
        preds = torch.argmax(net(data), dim=1)
        saliencies = occlusion(net, data, repeats=repeats)
        betas = lime_explanation(net, data, 64, repeats=repeats, kernel_width=0.25)
        cams = torch.from_numpy(camm(data))
        print(cams.shape)
        cam_block_size = image_size // cams.shape[-1]

        saliency_imgs = (
            saliencies.reshape((batch_size, repeats, repeats))
            .repeat_interleave(block_size, dim=1)
            .repeat_interleave(block_size, dim=2)
        )

        cams = cams.repeat_interleave(cam_block_size, dim=1).repeat_interleave(
            cam_block_size, dim=2
        )

        beta_imgs = (
            betas.reshape((batch_size, repeats, repeats))
            .repeat_interleave(block_size, dim=1)
            .repeat_interleave(block_size, dim=2)
        )

        for img, sal, beta, cam, label, pred in zip(
            data, saliency_imgs, beta_imgs, cams, batch['label'], preds
        ):
            if label != 5 and label != 2:
                continue
            plt.subplot(221)
            plt.title(f'Original img, gt {hyperkvasir[label]} pred {hyperkvasir[pred]}')
            display_pytorch_image(img)
            plt.subplot(222)
            overlay_saliency(img, cam, 'gradcam')
            plt.subplot(223)
            overlay_saliency(img, beta, 'lime')
            plt.subplot(224)
            overlay_saliency(img, sal, 'occlusion')

            plt.tight_layout()
            plt.show()

        if i > 20:
            pass
            # break

    betas = torch.cat(all_betas, dim=0)
    print(betas.shape)
    betas_per[key] = betas

with open('saved_metrics/lime_betas1.pkl', 'wb') as file:
    pickle.dump(betas_per, file)
