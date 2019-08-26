# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch


def postprocess_pr(pr_outs):
    _, pr_outs = torch.max(pr_outs, dim=1)
    return [p.item() for p in pr_outs.detach().cpu()]


def postprocess_gt(gt_labels):
    return [l.item() for l in gt_labels.detach().cpu()]
