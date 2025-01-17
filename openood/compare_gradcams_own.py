import torch
import os, sys
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, EigenCAM
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
    GradCAMWrapper,
)
import pickle
import matplotlib.cm as cm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from typing import Callable, List, Tuple, Optional

dogs = {
    0: 'Shih-Zu',
    1: 'Rhodesian_ridgeback',
    2: 'Beagle',
    3: 'English_foxhound',
    4: 'Border_terrier',
    5: 'Australian_terrier',
    6: 'Golden_retriever',
    7: 'Old_English_sheepdog',
    8: 'Samoyed',
    9: 'Dingo',
}


class GradCAMNoRescale(EigenCAM):
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


id_name = sys.argv[1]
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
        print(batch['label'])

        target_layers = [net.layer4[-1]]
        camm = GradCAMNoRescale(model=net, target_layers=target_layers)
        other_cam = GradCAMWrapper(
            model=net, target_layer=target_layers[0], do_relu=True
        )

        cams = torch.from_numpy(camm(data))
        self_cams = other_cam(data)

        outputs.append([data, cams, self_cams, batch['label'], camm.outputs])

    maxval_gradcam = None
    id_bool = True
    for i in range(16):
        if id_bool:
            values = outputs[0]
        else:
            values = outputs[1]

        img = values[0][i]
        gradcam_sal = values[1][i]
        self_gradcam_sal = values[2][i]
        label = values[3][i]
        pred = values[4][i]

        plt.subplot(311)
        # plt.title(f'Original img, gt {hyperkvasir[label]} pred {hyperkvasir[pred]}')
        display_pytorch_image(img)
        plt.subplot(312)
        plt.title('library')
        plt.imshow(gradcam_sal)
        # plt.imshow(gradcam_sal, vmax=1)
        plt.subplot(313)
        plt.title('self')
        plt.imshow(self_gradcam_sal)
        # plt.imshow(self_gradcam_sal, vmax=1)

        # plt.suptitle(f'pred={dogs[pred.item()]}')
        plt.tight_layout()
        plt.show()

        id_bool = not id_bool
