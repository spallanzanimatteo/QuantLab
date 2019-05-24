# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn


class HingeLoss(nn.Module):

    def __init__(self, num_classes, margin=1.0, p=2.0):
        super(HingeLoss, self).__init__()
        self.num_classes = num_classes
        self.margin      = margin
        self.p           = p

    def forward(self, inputs, labels):
        labels_one_hot = torch.Tensor(labels.unsqueeze(1).size(0), self.num_classes).to(labels.device)
        labels_one_hot.fill_(-1.)
        labels_one_hot = labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.)
        out            = self.margin - torch.mul(inputs, labels_one_hot)
        out            = torch.max(out.cpu(), torch.Tensor([0.]))
        out            = torch.pow(out, self.p)
        return torch.mean(out)
