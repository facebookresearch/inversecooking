# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


class MaskedCrossEntropyCriterion(_WeightedLoss):

    def __init__(self, ignore_index=[-100], reduce=None):
        super(MaskedCrossEntropyCriterion, self).__init__()
        self.padding_idx = ignore_index
        self.reduce = reduce

    def forward(self, outputs, targets):
        lprobs = nn.functional.log_softmax(outputs, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        for idx in self.padding_idx:
            # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
            targets[targets == idx] = 0

        nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
        if self.reduce:
            nll_loss = nll_loss.sum()

        return nll_loss.squeeze()


def softIoU(out, target, e=1e-6, sum_axis=1):

    num = (out*target).sum(sum_axis, True)
    den = (out+target-out*target).sum(sum_axis, True) + e
    iou = num / den

    return iou


def update_error_types(error_types, y_pred, y_true):

    error_types['tp_i'] += (y_pred * y_true).sum(0).cpu().data.numpy()
    error_types['fp_i'] += (y_pred * (1-y_true)).sum(0).cpu().data.numpy()
    error_types['fn_i'] += ((1-y_pred) * y_true).sum(0).cpu().data.numpy()
    error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0).cpu().data.numpy()

    error_types['tp_all'] += (y_pred * y_true).sum().item()
    error_types['fp_all'] += (y_pred * (1-y_true)).sum().item()
    error_types['fn_all'] += ((1-y_pred) * y_true).sum().item()


def compute_metrics(ret_metrics, error_types, metric_names, eps=1e-10, weights=None):

    if 'accuracy' in metric_names:
        ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))
    if 'jaccard' in metric_names:
        ret_metrics['jaccard'].append(error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all'] + eps))
    if 'dice' in metric_names:
        ret_metrics['dice'].append(2*error_types['tp_all'] / (2*(error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all']) + eps))
    if 'f1' in metric_names:
        pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
        rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
        f1_perclass = 2*(pre * rec) / (pre + rec + eps)
        if 'f1_ingredients' not in ret_metrics.keys():
            ret_metrics['f1_ingredients'] = [np.average(f1_perclass, weights=weights)]
        else:
            ret_metrics['f1_ingredients'].append(np.average(f1_perclass, weights=weights))

        pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
        rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
        f1 = 2*(pre * rec) / (pre + rec + eps)
        ret_metrics['f1'].append(f1)
