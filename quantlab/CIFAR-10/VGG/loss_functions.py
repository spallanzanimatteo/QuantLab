# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn


class HingeLoss(nn.Module):

    def __init__(self, num_classes=10, margin=1.0, p=2.0):
        super(HingeLoss, self).__init__()
        self.num_classes = num_classes
        self.margin      = margin
        self.p           = p

    def forward(self, pr_outs, gt_labels):
        # one-hot encode labels
        gt_outs = torch.Tensor(gt_labels.unsqueeze(1).size(0), self.num_classes).to(pr_outs.device)
        gt_outs.fill_(-1.)
        gt_outs = gt_outs.scatter_(1, gt_labels.unsqueeze(1), 1.)
        # hinge loss
        loss = self.margin - torch.mul(pr_outs, gt_outs)
        loss = torch.max(loss.cpu(), torch.Tensor([0.]))
        loss = torch.pow(loss, self.p)
        loss = torch.mean(loss)
        return loss
