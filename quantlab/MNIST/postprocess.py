# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import math
import torch


class Meter(object):
    def __init__(self):
        # main metric is classification error
        self.start_metric = math.inf
        self.topk         = (1,)
        self.maxk         = max(self.topk)
        self.bar_suffix   = None
        self.numel        = None
        self.loss         = None
        self.avg_loss     = None
        self.correct      = None
        self.avg_accuracy = None
        self.avg_metric   = None

        self.reset()

    def reset(self):
        self.numel        = 0
        self.loss         = 0.
        self.avg_loss     = 0.
        self.correct      = [0 for k in self.topk]
        self.avg_accuracy = [0. for k in self.topk]

    def update(self, outputs, labels, loss):
        batch_size  = outputs.size(0)
        self.numel += batch_size
        self.loss  += loss * batch_size
        with torch.no_grad():
            _, topk_preds = torch.topk(outputs, self.maxk, dim=1)
            correct       = torch.eq(labels.view(batch_size, -1).expand_as(topk_preds), topk_preds)
            for i, k in enumerate(self.topk):
                self.correct[i] += correct[:, :k].sum().item()
        self.avg_loss     = self.loss / self.numel
        self.avg_accuracy = [(100. * c) / self.numel for c in self.correct]
        # main metric is classification error
        self.avg_metric   = 100. - self.avg_accuracy[0]

    def is_better(self, valid_metric, best_metric):
        # compare classification errors
        return valid_metric < best_metric

    def print_metric(self, metric):
        # main metric is classification error
        print("Best accuracy: \t\t{:6.2f}%%".format(100. - metric))

    def bar(self):
        return '| Loss: {loss:6.4f} | Accuracy: {top:6.2f}%%'.format(
                loss=self.avg_loss,
                top=self.avg_accuracy[0])
