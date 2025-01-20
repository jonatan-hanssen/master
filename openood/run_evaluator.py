import torch
from openood.postprocessors.vim_postprocessor import VIMPostprocessor
import numpy as np
import pickle
from utils import get_network
import argparse
import sys

from openood.postprocessors.lime_postprocessor import LimeVIMPostprocessor
from openood.postprocessors.gradknn_postprocessor import GradKNNPostprocessor
from openood.postprocessors.occlusion_postprocessor import OcclusionVIMPostprocessor
from openood.postprocessors.grad_mean_postprocessor import GradMeanPostprocessor
from openood.evaluation_api import Evaluator
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--postprocessor', '-p', type=str, default='vim')
parser.add_argument('--batch_size', '-b', type=int, default=200)

args = parser.parse_args(sys.argv[1:])

postprocessor = None
id_name = args.dataset
postprocessor_name = args.postprocessor  # @param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}

if postprocessor_name == 'occlusion':
    postprocessor = OcclusionVIMPostprocessor(None)
    postprocessor_name = None
if postprocessor_name == 'lime':
    postprocessor = LimeVIMPostprocessor(None)
    postprocessor_name = None

if postprocessor_name == 'gradknn':
    postprocessor = GradKNNPostprocessor(None)
    postprocessor_name = None

if postprocessor_name == 'gradmean':
    postprocessor = GradMeanPostprocessor(None)
    postprocessor_name = None


net = get_network(id_name)


print(f'ID Dataset: {id_name}')
print(f'Postprocessor: {args.postprocessor}')

evaluator = Evaluator(
    net,
    id_name=id_name,  # the target ID dataset
    # id_name='imagenet',  # the target ID dataset
    data_root='./data',  # change if necessary
    config_root=None,  # see notes above
    preprocessor=None,  # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name,  # the postprocessor to use
    postprocessor=postprocessor,  # if you want to use your own postprocessor
    batch_size=args.batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=2,
    data_split='val',  # added by me, split into val and test for development
    # bootstrap_seed=i,  # added by me, bootstrap validation
)  # could use more num_workers outside colab

metrics = evaluator.eval_ood(fsood=False)

# with open(f'saved_metrics/{postprocessor_name}.pkl', 'wb') as file:
#     pickle.dump(evaluator.scores, file)
