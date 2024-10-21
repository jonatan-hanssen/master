import torch
from openood.postprocessors.vim_postprocessor import VIMPostprocessor
import numpy as np
import pickle
import torchvision.transforms as transforms
from PIL import Image

from openood.postprocessors.lime_postprocessor import LimeVIMPostprocessor
from openood.postprocessors.cam_distance_postprocessor import CamDistancePostprocessor
from openood.evaluation_api import Evaluator
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet

# load the model

net = ResNet18_224x224(num_classes=6)
net.load_state_dict(
    torch.load(
        './results/hyperkvasir_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'
    )
)
net.cuda()
net.eval()

image_path = './data/images_largescale/hyperkvasir/lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2/007bcee7-a272-4e74-b10d-78d4d7816678.jpg'

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Load the image
image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode

# Apply the transformations to the image
x = transform(image).unsqueeze(0).cuda()

print(torch.argmax(net(x)))
