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


class SaliencyPlusLogitPostprocessor(BasePostprocessor):
    def __init__(self, config, saliency_generator=None, aggregator=None):
        super().__init__(config)
        self.setup_flag = False
        self.APS_mode = False
        self.saliency_generator = saliency_generator
        self.aggregator = aggregator

        self.logit_std = None
        self.saliency_std = None
        self.sign = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            all_max_logits = list()
            all_saliency_aggregates = list()

            for batch in tqdm(
                id_loader_dict['val'], desc='ID Setup: ', position=0, leave=True
            ):
                data = batch['data'].cuda()

                max_logits, targets = torch.max(net(data), dim=-1)
                saliencies = self.saliency_generator(data)
                saliencies = saliencies.reshape(
                    saliencies.shape[0],
                    torch.prod(torch.tensor(saliencies.shape[1:])),
                )

                aggregate = self.aggregator(saliencies, dim=-1)

                max_logits = max_logits.detach().cpu()

                all_max_logits.append(max_logits)
                all_saliency_aggregates.append(aggregate)

            self.logit_std = torch.cat(all_max_logits).std()
            self.saliency_std = torch.cat(all_saliency_aggregates).std()

            id_saliency_mean = torch.cat(all_saliency_aggregates).mean()

            all_saliency_aggregates = list()
            for batch in tqdm(
                ood_loader_dict['val'], desc='OOD Setup: ', position=0, leave=True
            ):
                data = batch['data'].cuda()

                saliencies = self.saliency_generator(data)
                saliencies = saliencies.reshape(
                    saliencies.shape[0],
                    torch.prod(torch.tensor(saliencies.shape[1:])),
                )

                aggregate = self.aggregator(saliencies, dim=-1)

                all_saliency_aggregates.append(aggregate)

            ood_saliency_mean = torch.cat(all_saliency_aggregates).mean()

            self.sign = 1 if id_saliency_mean > ood_saliency_mean else -1
            print(f'{self.saliency_std=}')
            print(f'{id_saliency_mean=}')
            print(f'{ood_saliency_mean=}')

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

        max_logits = max_logits.detach().cpu()
        preds = preds.detach().cpu()

        score_ood = max_logits / self.logit_std + self.sign * (
            aggregate / self.saliency_std
        )
        return preds, score_ood
