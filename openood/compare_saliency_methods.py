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

plt.rcParams.update({'font.size': 16})

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
        cam_wrapper = GradCAMWrapper(net, target_layer=net.layer4[-1])

        cams = cam_wrapper(data)
        occlusions = occlusion(net, data, repeats=repeats)
        limes = lime_explanation(net, data, 64, repeats=repeats, kernel_width=0.75)

        preds = torch.argmax(cam_wrapper.outputs, dim=1)

        outputs.append([data, occlusions, limes, cams, batch['label'], preds])

    id_data = outputs[0]
    ood_data = outputs[1]

    normalize = False
    interpolation = 'bilinear'
    opacity = 100

    for i in range(16):
        id_img = id_data[0][i]
        id_occlusion_sal = id_data[1][i]
        id_lime_sal = id_data[2][i]
        id_gradcam_sal = id_data[3][i]
        label = id_data[4][i]
        id_pred = id_data[5][i]

        ood_img = ood_data[0][i]
        ood_occlusion_sal = ood_data[1][i]
        ood_lime_sal = ood_data[2][i]
        ood_gradcam_sal = ood_data[3][i]
        label = ood_data[4][i]
        ood_pred = ood_data[5][i]

        plt.subplot(241)
        display_pytorch_image(id_img)
        plt.title(f'ID Image,\nprediction: {dogs[id_pred.item()]}')
        plt.subplot(242)
        plt.title('GradCAM')
        overlay_saliency(
            id_img,
            id_gradcam_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )
        plt.subplot(243)
        plt.title('Lime')
        overlay_saliency(
            id_img,
            id_lime_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )
        plt.subplot(244)
        plt.title('Occlusion')
        overlay_saliency(
            id_img,
            id_occlusion_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
        )

        plt.subplot(245)
        plt.title(f'OOD Image,\nprediction: {dogs[ood_pred.item()]}')
        display_pytorch_image(ood_img)
        plt.subplot(246)
        plt.title('GradCAM')
        overlay_saliency(
            ood_img,
            ood_gradcam_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
            previous_maxval=torch.max(id_gradcam_sal),
        )
        plt.subplot(247)
        plt.title('Lime')
        overlay_saliency(
            ood_img,
            ood_lime_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
            previous_maxval=torch.max(id_lime_sal),
        )
        plt.subplot(248)
        plt.title('Occlusion')
        overlay_saliency(
            ood_img,
            ood_occlusion_sal,
            normalize=normalize,
            interpolation=interpolation,
            opacity=opacity,
            previous_maxval=torch.max(id_occlusion_sal),
        )
        # maxval_gradcam = np.abs(np.max(numpify(gradcam_sal)))

        plt.suptitle(
            f'ID and OOD unnormalized saliencies for different XAI methods', size=25
        )
        plt.tight_layout()
        plt.show()
