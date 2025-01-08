import torchvision.transforms as tvs_trans

from openood.preprocessors import BasePreprocessor
from openood.utils import Config

from torch import ones_like, triu, ones
from torch.nn import Module
import torchvision.transforms.functional as tf
import torch

INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR

default_preprocessing_dict = {
    'cifar10': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    },
    'cifar100': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    },
    'imagenet': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'imagenet200': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'aircraft': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    },
    'cub': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    },
    'hyperkvasir': {
        'mask': True,
        'pre_size': 224,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'hyperkvasir_polyp': {
        'mask': True,
        'pre_size': 224,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'imagewoof': {
        'pre_size': 224,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
}


class Mask(Module):
    def __init__(self, ratio=0.2, square_ratio=0.32):
        super().__init__()
        self.ratio = ratio
        self.square_ratio = square_ratio

    def forward(self, img):
        img = tvs_trans.PILToTensor()(img)
        c, h, w = img.shape
        mask = torch.ones(1, h, w, dtype=torch.uint8)
        n = int(h * self.ratio)
        square_n = int(h * self.square_ratio)

        # mask[:, -n:, :n] = tf.rotate(triu(ones(1, n, n, dtype=torch.uint8)), 0)
        mask[:, -square_n:, :square_n] = torch.zeros(
            1, square_n, square_n, dtype=torch.uint8
        )
        mask[:, -n:, -n:] = tf.rotate(triu(ones(1, n, n, dtype=torch.uint8)), 90)
        mask[:, :n, -n:] = tf.rotate(triu(ones(1, n, n, dtype=torch.uint8)), 180)
        mask[:, :n, :n] = tf.rotate(triu(ones(1, n, n, dtype=torch.uint8)), 270)

        return tvs_trans.ToPILImage()(img * mask)


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""

    def __init__(self, config: Config):
        if 'mask' in config:
            self.transform = tvs_trans.Compose(
                [
                    Convert('RGB'),
                    tvs_trans.Resize(config.pre_size, interpolation=INTERPOLATION),
                    tvs_trans.CenterCrop(config.img_size),
                    Mask(),
                    tvs_trans.ToTensor(),
                    tvs_trans.Normalize(*config.normalization),
                ]
            )
        else:
            self.transform = tvs_trans.Compose(
                [
                    Convert('RGB'),
                    tvs_trans.Resize(config.pre_size, interpolation=INTERPOLATION),
                    tvs_trans.CenterCrop(config.img_size),
                    tvs_trans.ToTensor(),
                    tvs_trans.Normalize(*config.normalization),
                ]
            )


class ImageNetCPreProcessor(BasePreprocessor):
    def __init__(self, mean, std):
        self.transform = tvs_trans.Compose(
            [
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean, std),
            ]
        )


def get_default_preprocessor(data_name: str):
    # TODO: include fine-grained datasets proposed in Vaze et al.?

    if data_name not in default_preprocessing_dict:
        raise NotImplementedError(f'The dataset {data_name} is not supported')

    config = Config(**default_preprocessing_dict[data_name])
    preprocessor = TestStandardPreProcessor(config)

    return preprocessor
