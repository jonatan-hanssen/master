import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    get_dataloaders,
    display_pytorch_image,
    numpify,
    overlay_saliency,
    get_network,
    get_saliency_generator,
)
from lime import lime_image
from torchvision import transforms
from skimage.segmentation import mark_boundaries

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--ood', '-o', type=str, default='near')
parser.add_argument(
    '--normalize', '-n', action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    '--interpolation',
    '-i',
    type=str,
    default='bilinear',
    choices=['bilinear', 'nearest', 'none'],
)

args = parser.parse_args()

id_name = args.dataset
device = 'cuda'

dataloaders = get_dataloaders(id_name, batch_size=8, full=False, shuffle=True)

# load the model
net = get_network(id_name)

saliency_dict = dict()

generator_func = get_saliency_generator(args.generator, net, args.repeats)

id_batch = next(dataloaders['id'][0])
ood_batch = next(dataloaders[args.ood][0])

id_images = id_batch['data'].to(device)

id_image = id_images[0]
print(id_image.shape)

explainer = lime_image.LimeImageExplainer()

from PIL import Image


def tensor_to_rgb_image(tensor):
    # Ensure the tensor is detached from the computation graph and moved to CPU
    tensor = tensor.detach().cpu()

    # Reverse normalization for ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std[:, None, None] + mean[:, None, None]

    # Clip the tensor to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy array and scale to 0-255
    image_np = tensor.permute(1, 2, 0).numpy()  # Change from CHW to HWC
    image_np = (image_np * 255).astype(np.uint8)

    return image_np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def predict(image):
    image = torch.tensor(image).permute(0, 3, 1, 2)
    image = image / 255.0
    image = normalize(image)
    image = image.to(device)
    pred = torch.nn.functional.softmax(net(image), dim=1)
    return pred.detach().cpu().numpy()


explanation = explainer.explain_instance(
    tensor_to_rgb_image(id_image), predict, top_labels=5, hide_color=0, num_samples=1000
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
)

plt.imshow(mask)
plt.show()
img_boundry1 = mark_boundaries(temp / 255.0, mask)
plt.imshow(img_boundry1)
plt.show()
