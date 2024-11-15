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
import numpy as np

from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet


def get_network(id_name: str):
    if id_name == 'cifar10':
        net = ResNet18_32x32(num_classes=10)
        net.load_state_dict(
            torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
        )

    elif id_name == 'cifar100':
        net = ResNet18_32x32(num_classes=100)
        net.load_state_dict(
            torch.load('./models/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
        )

    elif id_name == 'hyperkvasir':
        net = ResNet18_224x224(num_classes=6)
        net.load_state_dict(
            torch.load(
                './results/hyperkvasir_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'
            )
        )

    elif id_name == 'hyperkvasir_polyp':
        net = ResNet18_224x224(num_classes=4)
        net.load_state_dict(
            torch.load(
                './results/hyperkvasir_polyp_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'
            )
        )

    else:
        raise ValueError('No such dataset')

    net.cuda()
    net.eval()
    return net


def overlay_saliency(img, sal, desc):
    display_pytorch_image(img)

    if isinstance(sal, torch.Tensor):
        sal = numpify(sal)
    # sal = np.maximum(sal, 0)
    sal = sal / np.max(np.abs(sal))

    plt.imshow(sal, alpha=np.abs(sal), cmap='bwr', vmin=-1, vmax=1)
    plt.title(desc)
    plt.axis('off')


def get_dataloaders(id_name: str, batch_size: int = 16):
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(filepath, 'configs')

    data_root = './data'

    preprocessor = get_default_preprocessor(id_name)

    loader_kwargs = {
        'batch_size': batch_size,
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


def denormalize(tensor, mean, std):
    """
    Denormalizes the image tensor using the provided mean and std.

    Args:
    tensor (torch.Tensor): Normalized image tensor.
    mean (list): Mean values used for normalization.
    std (list): Standard deviation values used for normalization.

    Returns:
    torch.Tensor: Denormalized image tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_tensor_image(tensor):
    """
    Plots a PyTorch tensor image which was normalized with ImageNet mean and std.

    Args:
    tensor (torch.Tensor): Tensor representing the image. Expected shape: (3, H, W).
    """
    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # If the tensor is on GPU, move it to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Denormalize the tensor
    tensor = denormalize(tensor.clone(), mean, std)

    # Clip values to ensure they are within the range [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert tensor to numpy array
    img = tensor.numpy()

    # Reshape the tensor to have channels as the last dimension
    img = np.transpose(img, (1, 2, 0))

    # If the image is in the range [0, 1], convert to [0, 255]
    img = img * 255.0

    # Ensure the image is in uint8 format
    img = img.astype(np.uint8)

    # Plot the image
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.show()


def lime_explanation(
    net,
    batch,
    perturbations=100,
    mask_prob=0.5,
    repeats=8,
    kernel_width=0.25,
    device='cuda',
):
    preds = net(batch)
    max_pred_ind = torch.argmax(preds, dim=1)

    block_size = batch.shape[-1] // repeats

    kernel = lambda distances: torch.sqrt(torch.exp(-(distances**2) / kernel_width**2))

    all_betas = list()

    for image, pred_label in zip(batch, max_pred_ind):
        images = image.unsqueeze(0).expand(perturbations, -1, -1, -1)

        masked_images, masks = mask_image(
            images, block_size=block_size, mask_prob=mask_prob
        )

        with torch.no_grad():
            network_preds = net(masked_images)[:, pred_label]

        original = torch.ones((1, masks.shape[-1]))

        cos = torch.nn.CosineSimilarity(dim=1)
        distances = 1 - cos(masks.float(), original.float())

        weights = kernel(distances)

        regressor = LinearRegression()

        regressor.fit(
            numpify(masks), numpify(network_preds), sample_weight=numpify(weights)
        )

        betas = torch.tensor(regressor.coef_).unsqueeze(0)

        all_betas.append(betas)

    return torch.cat(all_betas, dim=0)


def occlusion(net, batch, repeats=8, device='cuda'):
    preds = net(batch)

    max_pred, max_pred_ind = torch.max(preds, dim=1)

    block_size = batch.shape[-1] // repeats

    saliencies = list()

    for image, pred_value, pred_label in zip(batch, max_pred, max_pred_ind):
        images = image.unsqueeze(0).expand(repeats**2, -1, -1, -1)

        masked_images = occlude_images(images, block_size=block_size)

        with torch.no_grad():
            network_preds = net(masked_images)[:, pred_label]
        saliencies.append((pred_value - network_preds).unsqueeze(0))

    return torch.cat(saliencies, dim=0)


def numpify(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def mask_image(batch, block_size=4, mask_prob=0.5):
    batch_size = batch.shape[0]
    h = batch.shape[-1]

    repeats = h // block_size

    masks = (torch.rand(batch_size, repeats**2) > mask_prob).to(int)

    # Prepare the mask array
    mask_array = (
        masks.reshape((batch_size, repeats, repeats))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    mask_array = mask_array.unsqueeze(1)

    masked_images = batch * mask_array.to(batch.device)

    return masked_images, masks


def occlude_images(batch, block_size=4):
    batch_size = batch.shape[0]
    h = batch.shape[-1]

    repeats = h // block_size

    masks = torch.where(torch.eye(repeats**2) == 1, 0, 1)

    # Prepare the mask array
    mask_array = (
        masks.reshape((batch_size, repeats, repeats))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    mask_array = mask_array.unsqueeze(1)

    masked_images = batch * mask_array.to(batch.device)

    return masked_images


def display_pytorch_image(image: torch.Tensor, mask: torch.Tensor = None):
    def inverse_normalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    image = inverse_normalize(
        tensor=torch.clone(image), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    if mask is not None:
        image = image * mask

    if image.device != 'cpu':
        image = image.cpu()

    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
