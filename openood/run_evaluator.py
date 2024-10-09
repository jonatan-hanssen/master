import torch
from openood.postprocessors.vim_postprocessor import VIMPostprocessor
import numpy as np
import pickle

from openood.postprocessors.lime_postprocessor import LimeVIMPostprocessor
from openood.postprocessors.cam_distance_postprocessor import CamDistancePostprocessor
from openood.evaluation_api import Evaluator
from openood.networks import (
    ResNet18_32x32,
    ResNet18_224x224,
)  # just a wrapper around the ResNet

# load the model

net = ResNet18_32x32(num_classes=10)
net.load_state_dict(
    torch.load('./models/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
)
net.cuda()
net.eval()

# net = ResNet18_32x32(num_classes=100)
# net.load_state_dict(
#     torch.load('./models/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt')
# )
# net.cuda()
# net.eval()

# net = ResNet18_224x224(num_classes=200)
# net.load_state_dict(
#     torch.load(
#         './models/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt'
#     )
# )
# net.cuda()
# net.eval()

postprocessor_name = 'limevim'  # @param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}
postprocessor = LimeVIMPostprocessor(None)


evaluator = Evaluator(
    net,
    id_name='cifar10',  # the target ID dataset
    # id_name='imagenet200',  # the target ID dataset
    data_root='./data',  # change if necessary
    config_root=None,  # see notes above
    preprocessor=None,  # default preprocessing for the target ID dataset
    postprocessor_name=None,  # the postprocessor to use
    postprocessor=postprocessor,  # if you want to use your own postprocessor
    batch_size=200,  # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=2,
    data_split='val',  # added by me, split into val and test for development
)  # could use more num_workers outside colab


metrics = evaluator.eval_ood(fsood=False)

with open(f'saved_metrics/{postprocessor_name}.pkl', 'wb') as file:
    pickle.dump(evaluator.scores, file)
