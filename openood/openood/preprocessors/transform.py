import torchvision.transforms as tvs_trans
from torch import ones_like, triu, ones, uint8
from torch.nn import Module
import torch
import torchvision.transforms.functional as tf

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'imagenet200': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'aircraft': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cub': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


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


# More transform classes shall be written here
