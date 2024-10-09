from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


def get_dataloaders(id_name: str):
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(filepath, 'configs')

    data_root = './data'
    id_name = 'cifar10'

    preprocessor = get_default_preprocessor(id_name)

    loader_kwargs = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 8,
    }

    dataloader_dict = get_id_ood_dataloader(
        id_name, data_root, preprocessor, data_split='val', **loader_kwargs
    )

    def combine_dataloaders(dictionary):
        for key in dictionary:
            for batch in dictionary[key]:
                yield batch

    get_length = lambda dictionary: sum([len(dictionary[key]) for key in dictionary])

    id_generator = combine_dataloaders(dataloader_dict['id'])
    near_generator = combine_dataloaders(dataloader_dict['ood']['near'])
    far_generator = combine_dataloaders(dataloader_dict['ood']['far'])

    id_length = get_length(dataloader_dict['id'])
    near_length = get_length(dataloader_dict['ood']['near'])
    far_length = get_length(dataloader_dict['ood']['far'])

    return {
        'id': (id_generator, id_length),
        'near': (near_generator, near_length),
        'far': (far_generator, far_length),
    }


def lime_explanation(net, batch, perturbations=100, mask_prob=0.5, block_size=4, device='cuda'):

    preds = net(batch)
    max_pred_ind = torch.argmax(preds, dim=1)

    kernel_width = 0.25
    kernel = lambda distances : torch.sqrt(torch.exp(-(distances**2)/kernel_width**2))

    all_betas = list()

    for image, pred_label in zip(batch, max_pred_ind):

        images = image.unsqueeze(0).expand(perturbations, -1, -1, -1)

        masked_images, masks = mask_image(images, block_size=4, mask_prob=mask_prob)

        network_preds = net(masked_images)[:, pred_label]

        original = torch.ones((1, masks.shape[-1]))

        cos = torch.nn.CosineSimilarity(dim=1)
        distances = 1 - cos(masks.float(), original.float())

        weights = kernel(distances)

        regressor = LinearRegression()

        regressor.fit(numpify(masks), numpify(network_preds), sample_weight=numpify(weights))

        betas = torch.tensor(regressor.coef_).unsqueeze(0)

        all_betas.append(betas)

        # print(betas)
        # ind = torch.argsort(betas, descending=True)
        # print(ind)
        #
        # betas = torch.zeros_like(betas)
        # print(betas.shape)
        # betas[ind[:20]] = 1
        #
        #
        # # Prepare the mask array
        # mask_array = (
        #     betas.reshape((8, 8))
        #     .repeat_interleave(block_size, dim=0)
        #     .repeat_interleave(block_size, dim=1)
        # ).unsqueeze(0).cuda()
        #
        # print(image.shape)
        # display_pytorch_image(image, mask=mask_array)

    return torch.cat(all_betas, dim=0)



        # return masked_images, masks

def numpify(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def mask_image(batch, block_size=4, mask_prob=0.5):

    batch_size = batch.shape[0]
    h = batch.shape[-1]

    repeats = h // block_size

    masks = (torch.rand(batch_size, repeats ** 2) > mask_prob).to(int)

    # Prepare the mask array
    mask_array = (
        masks.reshape((batch_size, repeats, repeats))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    mask_array = mask_array.unsqueeze(1)

    masked_images = batch * mask_array.to(batch.device)

    return masked_images, masks

def display_pytorch_image(image: torch.Tensor, mask: torch.Tensor = None):
    def inverse_normalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    image = inverse_normalize(
        tensor=image,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    if mask is not None:
        image = image * mask

    if image.device != 'cpu':
        image = image.cpu()

    plt.imshow(image.permute(1, 2, 0))
    plt.show()


