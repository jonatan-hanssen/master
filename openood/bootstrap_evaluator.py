import torch
from openood.postprocessors.vim_postprocessor import VIMPostprocessor
from openood.postprocessors.occlusion_postprocessor import OcclusionVIMPostprocessor
import numpy as np
import argparse, sys, pickle

from utils import get_network, get_aggregate_function, get_saliency_generator

from openood.postprocessors.lime_postprocessor import LimeVIMPostprocessor
from openood.postprocessors.gradknn_postprocessor import GradKNNPostprocessor
from openood.postprocessors.occlusion_postprocessor import OcclusionVIMPostprocessor
from openood.postprocessors.grad_mean_postprocessor import GradMeanPostprocessor
from openood.evaluation_api import Evaluator
from openood.postprocessors.saliency_aggregate_postprocessor import (
    SaliencyAggregatePostprocessor,
)

from openood.postprocessors.logit_mean_saliency_postprocessor import (
    LogitMeanSaliencyPostprocessor,
)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--postprocessor', '-p', type=str, default='vim')
parser.add_argument('--batch_size', '-b', type=int, default=200)
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--aggregator', '-a', type=str, default='Norm')
parser.add_argument('--split', '-s', type=str, default='val')

args = parser.parse_args(sys.argv[1:])
net = get_network(args.dataset)

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

if postprocessor_name == 'logitmean':
    postprocessor = LogitMeanSaliencyPostprocessor(None)
    postprocessor_name = None

if postprocessor_name == 'salagg':
    generator = get_saliency_generator(args.generator, net)
    aggregator = get_aggregate_function(args.aggregator)
    postprocessor = SaliencyAggregatePostprocessor(
        None, saliency_generator=generator, aggregator=aggregator
    )
    postprocessor_name = None


print(f'ID Dataset: {id_name}')
print(f'Postprocessor: {args.postprocessor}')

all_metrics = list()
all_scores = list()

for i in range(10):
    print(f'Run {i}')
    evaluator = Evaluator(
        net,
        id_name=id_name,  # the target ID dataset
        # id_name='imagenet200',  # the target ID dataset
        data_root='./data',  # change if necessary
        config_root=None,  # see notes above
        preprocessor=None,  # default preprocessing for the target ID dataset
        postprocessor_name=postprocessor_name,  # the postprocessor to use
        postprocessor=postprocessor,  # if you want to use your own postprocessor
        batch_size=200,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=2,
        data_split=args.split,  # added by me, split into val and test for development
        bootstrap_seed=i,  # added by me, bootstrap validation
    )  # could use more num_workers outside colab

    metrics = evaluator.eval_ood(fsood=False)
    all_metrics.append(metrics)
    all_scores.append(evaluator.scores)

if args.postprocessor == 'salagg':
    with open(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_{args.generator}_{args.aggregator}_bootstrapped.pkl',
        'wb',
    ) as file:
        pickle.dump([all_metrics, all_scores], file)
else:
    with open(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_bootstrapped.pkl', 'wb'
    ) as file:
        pickle.dump([all_metrics, all_scores], file)
