import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from PIL import Image
import seaborn as sns
from utils import (
    get_dataloaders,
    get_labels,
    display_pytorch_image,
    numpify,
    overlay_saliency,
    get_network,
    get_saliency_generator,
    GradCAMWrapper,
)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--repeats', '-r', type=int, default=4)
parser.add_argument('--skips', type=int, default=0)
parser.add_argument('--opacity', type=float, default=1.6)
parser.add_argument('--relu', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--full', '-f', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--ood', '-o', type=str, default='near')
parser.add_argument('--batch_size', '-b', type=int, default=8)
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
print(args)


if args.pgf:
    matplotlib.use('pgf')
    matplotlib.rcParams.update(
        {
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        }
    )


id_name = 'imagenet'
device = 'cuda'

dataloaders = get_dataloaders(id_name, batch_size=16, full=False, shuffle=False)
labels = get_labels(id_name)


# load the model
net = get_network(id_name)

# Load the image
image_path = '0.jpg'
image = Image.open(image_path)

# Define the transformation
preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize to 256 pixels (maintains aspect ratio)
        transforms.CenterCrop(224),  # Center crop to 224x224 (common ResNet input size)
        transforms.ToTensor(),  # Convert image to PyTorch tensor (scales to [0, 1])
        transforms.Normalize(  # Normalize with ResNet mean and std deviation
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Apply the transformations
image_tensor = preprocess(image)

# Add a batch dimension (1,)
husky = image_tensor.unsqueeze(0).to(device)


id_batch = next(dataloaders['id'][0])
id_images = id_batch['data'].to(device)

wrapper = GradCAMWrapper(model=net, target_layer=net.layer4[-1])

# husky_saliencies = wrapper(husky, class_to_backprog=250)
# flute_saliencies = wrapper(husky, class_to_backprog=558)
# plt.subplot(121)
#
# overlay_saliency(
#     husky[0],
#     husky_saliencies[0],
#     normalize=args.normalize,
#     interpolation=args.interpolation,
#     opacity=args.opacity,
# )
#
# plt.subplot(122)
#
# overlay_saliency(
#     husky[0],
#     flute_saliencies[0],
#     normalize=args.normalize,
#     interpolation=args.interpolation,
#     opacity=args.opacity,
# )
# plt.show()

for y in range(100):
    print(y)
    ids = [0, 32, 418, 429, 434, 448, 453, 587, 596]

    id_batch = next(dataloaders['id'][0])
    id_images = id_batch['data'].to(device)

    for i in range(len(ids)):
        plt.subplot(3, 3, i + 1)

        if not ids[i]:
            saliencies = wrapper(id_images)
        else:
            saliencies = wrapper(id_images, class_to_backprog=ids[i])

        overlay_saliency(
            id_images[10],
            saliencies[10],
            normalize=args.normalize,
            interpolation=args.interpolation,
            opacity=args.opacity,
        )

    plt.show()
