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


def lime_explanation(net, batch):
    pass


def mask_image(mask_tensor):
    # Define block size
    block_size = 4

    # Prepare the mask array
    mask_tensor = torch.tensor(mask_tensor, dtype=int)
    mask_array = (
        mask_tensor.reshape((2, 4, 4))
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    # Ensure the mask matches the image dimensions
    print(mask_array)


# Example usage:
# Define the mask tensor of 64 bits (8x8 regions)








