from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from tqdm import tqdm
from time import time
import captum

from .base_postprocessor import BasePostprocessor


class SaliencyAggregatePostprocessor(BasePostprocessor):
    def __init__(self, config, saliency_generator=None, aggregator=None):
        super().__init__(config)
        self.setup_flag = False
        self.APS_mode = False
        self.saliency_generator = saliency_generator
        self.aggregator = aggregator
        self.sigma = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            mean_aggregate_list = list()
            for loader in (id_loader_dict['val'], ood_loader_dict['val']):
                all_aggregates = list()
                for batch in tqdm(loader, desc='Setup: ', position=0, leave=True):
                    data = batch['data'].cuda()

                    saliencies = self.saliency_generator(data)
                    saliencies = saliencies.reshape(
                        saliencies.shape[0],
                        torch.prod(torch.tensor(saliencies.shape[1:])),
                    )

                    aggregate = self.aggregator(saliencies, dim=-1)
                    all_aggregates.append(aggregate)

                mean_aggregates = torch.cat(all_aggregates).mean()

                mean_aggregate_list.append(mean_aggregates)

            id_mean_saliency, ood_mean_saliency = mean_aggregate_list

            if id_mean_saliency > ood_mean_saliency:
                self.sigma = 1
            else:
                self.sigma = -1

            print(f'{self.sigma=}')
            print(f'{id_mean_saliency=}')
            print(f'{ood_mean_saliency=}')
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        max_logits, preds = torch.max(net(data), dim=-1)

        saliencies = self.saliency_generator(data)
        saliencies = saliencies.reshape(
            saliencies.shape[0],
            torch.prod(torch.tensor(saliencies.shape[1:])),
        )

        aggregate = self.aggregator(saliencies, dim=-1)

        score_ood = aggregate * self.sigma

        return preds, score_ood
