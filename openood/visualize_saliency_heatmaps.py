import torch
import os, sys
import torch.nn as nn
from tqdm import tqdm
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
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


# plt.rcParams.update({'font.size': 18})

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

        cam_wrapper = GradCAMWrapper(
            model=net, target_layer=net.layer4[-1], do_relu=True
        )

        cams = cam_wrapper(data)

        outputs.append(
            {
                'images': data,
                'saliencies': cams,
                'preds': cam_wrapper.outputs,
                'ground_truths': batch['label'],
            }
        )

    id_data, ood_data = outputs[0], outputs[1]

    opacity = 0.9
    interpolation = 'bilinear'
    subplotsize = 15
    normalize = True

    for i in range(16):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        id_img = id_data['images'][i]
        id_sal = id_data['saliencies'][i]
        id_pred = torch.argmax(id_data['preds'][i]).item()

        ood_img = ood_data['images'][i]
        ood_sal = ood_data['saliencies'][i]
        ood_pred = torch.argmax(ood_data['preds'][i]).item()

        axes[0][0].set_title(
            f'In Distribution,\n prediction: {dogs[id_pred]}', size=subplotsize
        )
        display_pytorch_image(id_img, ax=axes[0][0])

        axes[0][1].set_title(
            f'Out of Distribution,\n prediction: {dogs[ood_pred]}', size=subplotsize
        )
        display_pytorch_image(ood_img, ax=axes[0][1])

        axes[1][0].set_title(
            f'In Distribution,\n{"" if normalize else "un"}normalized saliencies',
            size=subplotsize,
        )
        overlay_saliency(
            id_img,
            id_sal,
            opacity=opacity,
            interpolation=interpolation,
            normalize=normalize,
            ax=axes[1][0],
        )

        axes[1][1].set_title(
            f'Out of Distribution,\n{"" if normalize else "un"}normalized saliencies',
            size=subplotsize,
        )
        im = overlay_saliency(
            ood_img,
            ood_sal,
            opacity=opacity,
            interpolation=interpolation,
            normalize=normalize,
            ax=axes[1][1],
        )

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.suptitle('GradCAM saliencies for ID and Far-OOD images', size=22)
        plt.show()
