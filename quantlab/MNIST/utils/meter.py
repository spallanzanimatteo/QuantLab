# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import math


class Meter(object):
    def __init__(self, pp_pr, pp_gt):
        self.n_tracked    = None
        self.loss         = None
        self.avg_loss     = None
        # main metric is classification error
        self.pp_pr        = pp_pr
        self.pp_gt        = pp_gt
        self.start_metric = 100.
        self.correct      = None
        self.avg_metric   = None
        self.reset()

    def reset(self):
        self.n_tracked  = 0
        self.loss       = 0.
        self.avg_loss   = 0.
        self.correct    = 0
        self.avg_metric = self.start_metric

    def update(self, pr_outs, gt_labels, loss, track_metric=False):
        gt_labels       = self.pp_gt(gt_labels)
        batch_size      = len(gt_labels)
        self.n_tracked += batch_size
        # update loss
        self.loss      += loss * batch_size
        self.avg_loss   = self.loss / self.n_tracked
        if track_metric:
            # update main metric
            pr_labels = self.pp_pr(pr_outs)
            assert len(pr_labels) == len(gt_labels), 'Number of predictions and number of ground truths do not match!'
            for i in range(len(pr_labels)):
                self.correct += pr_labels[i] == gt_labels[i]
            self.avg_metric = 100. * (1. - self.correct / self.n_tracked)

    def is_better(self, current_metric, best_metric):
        # compare classification errors
        return current_metric < best_metric

    def bar(self):
        return '| Loss: {loss:8.5f} | Accuracy: {acc:6.2f}%%'.format(
                loss=self.avg_loss,
                acc=100. - self.avg_metric)
