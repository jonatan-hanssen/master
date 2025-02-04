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
from torchvision.transforms import Resize, InterpolationMode
import pytorch_grad_cam

from torchvision import transforms
import captum
from sklearn.metrics import roc_curve, auc

from skimage.segmentation import slic
from skimage.segmentation import find_boundaries
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
    ResNet50,
)  # just a wrapper around the ResNet

from matplotlib.colors import LinearSegmentedColormap


def replace_layer_recursive(model, old_layer, new_layer):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


def replace_all_layer_type_recursive(model, old_layer_type, new_layer):
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer
        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


def find_layer_types_recursive(model, layer_types):
    def predicate(layer):
        return type(layer) in layer_types

    return find_layer_predicate_recursive(model, predicate)


def find_layer_predicate_recursive(model, predicate):
    result = []
    for name, layer in model._modules.items():
        if predicate(layer):
            result.append(layer)
        result.extend(find_layer_predicate_recursive(layer, predicate))
    return result


def get_network(id_name: str):
    if id_name == 'cifar10':
        net = ResNet18_32x32(num_classes=10)
        net.load_state_dict(
            torch.load(
                './models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
            )
        )

    elif id_name == 'imagenet':
        net = ResNet50(num_classes=1000)
        net.load_state_dict(torch.load('./models/resnet50_imagenet1k_v1.pth'))

    elif id_name == 'cifar100':
        net = ResNet18_32x32(num_classes=100)
        net.load_state_dict(
            torch.load(
                './models/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
            )
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
    elif id_name == 'imagewoof':
        net = ResNet18_224x224(num_classes=10)
        net.load_state_dict(
            torch.load(
                './results/imagewoof_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'
            )
        )

    else:
        raise ValueError('No such dataset')

    net.cuda()
    net.eval()
    return net


def calculate_auc(id_aggregate, ood_aggregate):
    values = torch.cat((id_aggregate, ood_aggregate)).numpy()
    labels = torch.cat(
        (torch.zeros_like(id_aggregate), torch.ones_like(ood_aggregate))
    ).numpy()

    fpr_list, tpr_list, thresholds = roc_curve(labels, -values)
    auroc = auc(fpr_list, tpr_list)
    return auroc


class GradCAMWrapper(torch.nn.Module):
    def __init__(
        self,
        model=None,
        target_layer=None,
        do_relu=False,
        subtype=None,
        normalize=False,
    ):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.do_relu = do_relu
        self.normalize = normalize

        self.subtype = subtype

        self.grads = None
        self.acts = None

        self.outputs = None

        self.handles = list()

        self.handles.append(
            self.target_layer.register_full_backward_hook(self.grad_hook)
        )
        self.handles.append(self.target_layer.register_forward_hook(self.act_hook))

    def grad_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0]

    def act_hook(self, module, input, output):
        self.acts = output

    def forward(self, x, return_feature=False):
        batch_size = x.shape[0]

        if return_feature:
            preds, feature = self.model(x, return_feature=True)

        else:
            preds = self.model(x)

        self.outputs = preds

        self.model.zero_grad(set_to_none=True)

        idxs = torch.argmax(preds, dim=1)

        # backward pass, this gets gradients for each prediction
        torch.sum(preds[torch.arange(batch_size), idxs]).backward()

        average_gradients = self.grads.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        saliency = self.acts * average_gradients

        saliency = torch.sum(saliency, dim=1)
        if self.do_relu:
            saliency = torch.nn.functional.relu(saliency)

        if self.normalize:
            mins = saliency.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            saliency -= mins
            maxes = saliency.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            saliency /= maxes

        if return_feature:
            return saliency.cpu().detach(), feature

        else:
            return saliency.cpu().detach()

    def __del__(self):
        for handle in self.handles:
            handle.remove()


def segmented_occlusion(net, batch, device='cuda'):
    batch_size = batch.shape[0]

    preds = torch.argmax(net(batch), dim=1)

    segmentations = list()
    for i in range(batch_size):
        seg = slic(batch[i].permute(1, 2, 0).cpu().numpy())
        segmentations.append(seg)

    segmentations = (
        torch.from_numpy(np.stack(segmentations)).unsqueeze(dim=1).to(device)
    )

    ablator = captum.attr.FeatureAblation(net)
    saliency = (
        ablator.attribute(batch, target=preds, feature_mask=segmentations)
        .detach()
        .cpu()
        .mean(dim=1)
    )

    return saliency


class EigenCAM(pytorch_grad_cam.EigenCAM):
    # Class that removes rescaling and just the dim of the conv layer
    def __init__(self, model, target_layers, reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layers, reshape_transform)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            # print(np.min(cam))
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return result


def get_saliency_generator(
    name: str,
    net: torch.nn.Module,
    repeats: int,
    just_mean: bool = False,
    do_relu: bool = False,
    normalize: bool = False,
) -> Callable:
    if name == 'gradcam':
        cam_wrapper = GradCAMWrapper(
            model=net, target_layer=net.layer4[-1], normalize=normalize
        )
        generator_func = cam_wrapper

    elif name == 'occlusion':
        generator_func = lambda data: occlusion(
            net, data, repeats=repeats, do_relu=do_relu
        )
    elif name == 'lime':
        generator_func = lambda data: lime_explanation(
            net, data, repeats=repeats, do_relu=do_relu
        )

    elif name == 'eigencam':
        cam_wrapper = EigenCAM(model=net, target_layers=[net.layer4[-1]])
        generator_func = lambda data: torch.tensor(cam_wrapper(data))

    elif name == 'eigengradcam':
        cam_wrapper = EigenGradCAM(model=net, target_layers=[net.layer4[-1]])
        generator_func = lambda data: torch.tensor(cam_wrapper(data))

    elif name == 'ablation':
        cam_wrapper = Ablation(model=net, target_layers=[net.layer4[-1]])
        generator_func = lambda data: torch.tensor(cam_wrapper(data))

    elif name == 'integratedgradients':
        if just_mean:

            def generator_func(data):
                targets = torch.argmax(net(data), dim=-1)
                lrp = captum.attr.IntegratedGradients(net)

                attributions = lrp.attribute(data, target=targets)

                attributions = torch.nn.functional.relu(attributions)

                attributions = attributions.sum(dim=1)

                return attributions.mean(dim=-1).mean(dim=-1).detach().cpu()

        else:

            def generator_func(data):
                targets = torch.argmax(net(data), dim=-1)
                lrp = captum.attr.IntegratedGradients(net)

                attributions = lrp.attribute(data, target=targets)

                attributions = torch.nn.functional.relu(attributions)

                attributions = attributions.sum(dim=1)

                return attributions

    elif name == 'segocc':
        if not just_mean:
            generator_func = lambda data: segmented_occlusion(net, data)
        else:

            def generator_func(data):
                saliencies = segmented_occlusion(net, data)
                means = torch.zeros(data.shape[0])
                for i, saliency in enumerate(saliencies):
                    means[i] = torch.unique(saliency).mean()
                return means

    elif name == 'seglime':
        if not just_mean:
            generator_func = lambda data: segmented_lime(
                net,
                data,
                perturbations=200,
                kernel_width=0.25,
                batch_size=100,
                do_relu=do_relu,
            )[0]
        else:

            def generator_func(data):
                betas = segmented_lime(net, data)[1]
                return torch.tensor([a.mean() for a in betas])

    else:
        raise TypeError('No such generator')

    return generator_func


class Ablation(pytorch_grad_cam.AblationCAM):
    # Class that removes rescaling and just the dim of the conv layer
    def __init__(self, model, target_layers, reshape_transform=None):
        super(Ablation, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[Callable],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        # Do a forward pass, compute the target scores, and cache the
        # activations
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32(
                [
                    target(output).cpu().item()
                    for target, output in zip(targets, outputs)
                ]
            )

        # Replace the layer with the ablation layer.
        # When we finish, we will replace it back, so the
        # original model is unchanged.
        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []
        # This is a "gradient free" method, so we don't need gradients here.
        with torch.no_grad():
            # Loop over each of the batch images and ablate activations for it.
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                new_scores = []
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                # Check which channels should be ablated. Normally this will be all channels,
                # But we can also try to speed this up by using a low
                # ratio_channels_to_ablate.
                channels_to_ablate = ablation_layer.activations_to_be_ablated(
                    activations[batch_index, :], self.ratio_channels_to_ablate
                )
                number_channels_to_ablate = len(channels_to_ablate)

                for i in range(0, number_channels_to_ablate, self.batch_size):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[: (number_channels_to_ablate - i)]

                    # Change the state of the ablation layer so it ablates the next channels.
                    # TBD: Move this into the ablation layer forward pass.
                    ablation_layer.set_next_batch(
                        input_batch_index=batch_index,
                        activations=self.activations,
                        num_channels_to_ablate=batch_tensor.size(0),
                    )
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[
                        batch_tensor.size(0) :
                    ]

                new_scores = self.assemble_ablation_scores(
                    new_scores,
                    original_scores[batch_index],
                    channels_to_ablate,
                    number_of_channels,
                )
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        # Returning the weights from new_scores
        return weights

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            # print(np.min(cam))
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class EigenGradCAM(pytorch_grad_cam.EigenGradCAM):
    # Class that removes rescaling and just the dim of the conv layer
    def __init__(self, model, target_layers, reshape_transform=None):
        super(EigenGradCAM, self).__init__(model, target_layers, reshape_transform)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            # print(np.min(cam))
            # cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return result


def fontsize(size):
    plt.rcParams.update({'font.size': size})


def overlay_saliency(
    img,
    sal,
    opacity=1,
    interpolation='nearest',
    normalize=False,
    previous_maxval=None,
    ax=None,
):
    display_pytorch_image(img, ax=ax)

    if interpolation == 'nearest':
        sal = Resize(img.shape[-2:], interpolation=InterpolationMode.NEAREST)(
            sal.unsqueeze(0)
        ).squeeze()
    elif interpolation == 'bilinear':
        sal = Resize(img.shape[-2:], interpolation=InterpolationMode.BILINEAR)(
            sal.unsqueeze(0)
        ).squeeze()
    elif interpolation == 'none':
        pass

    if isinstance(sal, torch.Tensor):
        sal = numpify(sal)

    sal -= np.min(sal)

    if normalize:
        sal = sal / np.max(np.abs(sal))

    if previous_maxval is None:
        max_val = np.max(sal)
    else:
        if previous_maxval > np.max(sal):
            max_val = previous_maxval
        else:
            max_val = np.max(sal)

    if ax is not None:
        ax.axis('off')
        if opacity > 1:
            return ax.imshow(sal, cmap='jet', vmax=max_val)
        else:
            return ax.imshow(sal, alpha=np.abs(sal) * opacity, cmap='jet', vmax=max_val)

    else:
        if opacity > 1 and opacity < 2:
            plt.imshow(sal, alpha=opacity - 1, cmap='jet', vmax=max_val)
        elif opacity > 2:
            plt.imshow(sal, cmap='jet', vmax=max_val)
        else:
            plt.imshow(sal, alpha=np.abs(sal) * opacity, cmap='jet', vmax=max_val)
        plt.axis('off')
        plt.axis('off')


def visualize_borders(segmentation: torch.Tensor, opacity_modifier: float = 0.8):
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()
    borders = find_boundaries(segmentation, mode='subpixel').astype('float32')
    borders = (
        transforms.Resize(
            segmentation.shape, interpolation=transforms.InterpolationMode.NEAREST
        )(torch.tensor(borders).unsqueeze(0))
        .squeeze()
        .numpy()
    )
    plt.imshow(borders, alpha=borders * opacity_modifier, cmap='grey')


def get_dataloaders(
    id_name: str, batch_size: int = 16, full: bool = False, shuffle: bool = False
):
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(filepath, 'configs')

    data_root = './data'

    preprocessor = get_default_preprocessor(id_name)

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
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

    if not full:
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

    else:
        id_generator = dataloader_dict['id']['test']
        near_generator = dataloader_dict['ood']['near']
        far_generator = dataloader_dict['ood']['far']

        return {
            'id': {id_name: id_generator},
            'near': near_generator,
            'far': far_generator,
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


def lime_explanation(
    net,
    batch,
    perturbations=100,
    mask_prob=0.5,
    repeats=8,
    kernel_width=0.25,
    do_relu=False,
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

    all_betas = torch.cat(all_betas, dim=0)
    if do_relu:
        all_betas = torch.nn.functional.relu(all_betas)

    return all_betas.reshape(-1, repeats, repeats)


def segmented_lime(
    net,
    batch,
    perturbations=100,
    mask_prob=0.5,
    kernel_width=0.25,
    batch_size=128,
    do_relu=False,
    device='cuda',
):
    num_images, c, h, w = batch.shape

    saliencies = torch.empty(num_images, h, w)

    preds = torch.argmax(net(batch), dim=1)

    all_betas = list()

    kernel = lambda distances: torch.sqrt(torch.exp(-(distances**2) / kernel_width**2))

    for i in range(num_images):
        image = batch[i]
        pred = preds[i]
        segmentation = torch.from_numpy(
            slic(reverse_imagenet_transform(image).permute(1, 2, 0).cpu().numpy())
        )

        num_segments = torch.max(segmentation)
        mask_tensor = (torch.rand(perturbations, num_segments) > mask_prob).to(int)

        masks = create_batch_masks(segmentation, mask_tensor)
        masks = masks.unsqueeze(1).to(device)

        stacked_image = image.unsqueeze(0).repeat(perturbations, 1, 1, 1)

        # for i, image in enumerate(stacked_image * masks):
        #     if i > 8:
        #         break
        #     plt.subplot(3, 3, i + 1)
        #     display_pytorch_image(image)
        # plt.show()

        masked_images = stacked_image * masks

        if perturbations <= batch_size:
            with torch.no_grad():
                network_preds = net(masked_images)[:, pred]

        else:
            network_preds = list()
            with torch.no_grad():
                for sub_batch in torch.split(masked_images, batch_size):
                    network_preds.append(net(sub_batch)[:, pred])

            network_preds = torch.cat(network_preds)
            print(network_preds.shape)

        original = torch.ones((1, mask_tensor.shape[-1]))

        cos = torch.nn.CosineSimilarity(dim=1)
        distances = 1 - cos(mask_tensor.float(), original.float())

        weights = kernel(distances)

        regressor = LinearRegression()

        regressor.fit(
            numpify(mask_tensor), numpify(network_preds), sample_weight=numpify(weights)
        )

        betas = torch.tensor(regressor.coef_)

        if do_relu:
            betas = torch.nn.functional.relu(betas)

        for j, v in enumerate(betas):
            saliencies[i][segmentation == j + 1] = v

        all_betas.append(betas)

    # for i in range(1, 6):
    #     plt.subplot(2, 5, i)
    #     display_pytorch_image(batch[i])
    #
    #     plt.subplot(2, 5, i * 2)
    #     plt.imshow(saliencies[i])
    # plt.show()

    return saliencies, all_betas


def create_batch_masks(segmentation, batch_segment_values):
    # segmentation is of shape (H, W)
    # batch_segment_values is of shape (N, num_segments)

    # Get batch size and number of segments
    N, num_segments = batch_segment_values.shape
    H, W = segmentation.shape

    # Initialize a mask tensor with ones
    batch_masks = torch.ones((N, H, W), dtype=torch.float32)

    # Create an expanded segmentation for comparison
    segmentation_expanded = segmentation.unsqueeze(0).expand(N, -1, -1)

    for segment_id in range(1, num_segments + 1):
        # Create a mask to determine where the zeroing should occur
        segment_mask = batch_segment_values[:, segment_id - 1].unsqueeze(1).unsqueeze(2)
        zero_mask = (segmentation_expanded == segment_id) * (segment_mask == 0)
        batch_masks[zero_mask] = 0

    return batch_masks


def reverse_imagenet_transform(tensor: torch.tensor):
    def inverse_normalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    return inverse_normalize(
        tensor=torch.clone(tensor),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def occlusion(net, batch, repeats=8, do_relu=False, device='cuda'):
    preds = net(batch)

    max_pred, max_pred_ind = torch.max(preds, dim=1)

    block_size = batch.shape[-1] // repeats

    saliencies = list()

    for image, pred_value, pred_label in zip(batch, max_pred, max_pred_ind):
        images = image.unsqueeze(0).expand(repeats**2, -1, -1, -1)

        masked_images = occlude_images(images, block_size=block_size)

        with torch.no_grad():
            network_preds = net(masked_images)[:, pred_label]
        saliencies.append((pred_value.detach() - network_preds.detach()).unsqueeze(0))

    saliencies = torch.cat(saliencies, dim=0)

    if do_relu:
        saliencies = torch.nn.functional.relu(saliencies)

    return saliencies.reshape(-1, repeats, repeats)


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


def display_pytorch_image(image: torch.Tensor, mask: torch.Tensor = None, ax=None):
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

    if ax is not None:
        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')
    else:
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
