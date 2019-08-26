# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import numpy as np
import torch

from quantlab.COCO.utils.utils import xywh2xyxy, bbox_iou


def compute_ap(recall_curve, precision_curve):
    """ Compute the average precision, given the recall and precision curves.
    Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    Returns
        The average precision.
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall_curve, [1.]))
    mpre = np.concatenate(([0.], precision_curve, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # then integrate
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, pr_conf, pr_cls, gt_cls):
    """ Compute the average precision, given the recall and precision curves.
    Arguments
        tp: True positives (list).
        pr_conf: Objectness value from 0-1 (list).
        pr_cls: Predicted object classes (list).
        gt_cls: True object classes (list).
    Returns
        The per-class average precision.
    """
    # sort by objectness confidence
    i = np.argsort(-pr_conf)
    tp, pr_conf, pr_cls = tp[i], pr_conf[i], pr_cls[i]
    # create recall-precision curve and compute AP (for each class)
    r, p, ap = [], [], []
    unique_classes = np.unique(gt_cls)
    for c in unique_classes:
        i = pr_cls == c
        n_p = i.sum()               # number of predicted annotations of class `c`    (TPs+FPs)
        n_gt = (gt_cls == c).sum()  # number of ground-truth annotations of class `c` (TPs+FNs)
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            r.append(0.)
            p.append(0.)
            ap.append(0.)
        else:
            # accumulate TPs and FPs
            tpc = (tp[i]).cumsum()
            fpc = (1 - tp[i]).cumsum()
            # recall (TPs / (TPs+FNs))
            recall_curve = tpc / n_gt
            r.append(recall_curve[-1])
            # precision (TPs / (TPs+FPs))
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
    return np.array(ap).mean()


class Meter(object):
    def __init__(self, pp_pr, pp_gt):
        self.n_tracked    = None
        self.loss         = None
        self.avg_loss     = None
        # main metric is mean Average Precision (mAP)
        self.pp_pr        = pp_pr
        self.pp_gt        = pp_gt
        self.start_metric = 0.
        self.stats        = None
        self.avg_metric   = None
        self.reset()

    def reset(self):
        self.n_tracked  = 0
        self.loss       = 0.
        self.avg_loss   = 0.
        self.stats      = []
        self.avg_metric = self.start_metric

    def _update_stats(self, pr_labels, gt_labels, iou_thres=0.5):
        """Compute detection statistics for the current batch.

        For each image in the batch, compare the list of predicted annotations
        with the list of ground-truth annotations.
        """
        for i, pr in enumerate(pr_labels):
            gt = gt_labels[i]
            na = len(gt)
            # na = 0 if gt is None else len(gt)
            gt_cls = gt[:, 0].tolist() if na else []  # target class
            if na:
                if pr is None:
                    img_stats = ([], torch.Tensor(), torch.Tensor(), gt_cls)
                else:
                    correct = [0] * len(pr)
                    gt_boxes = xywh2xyxy(gt[:, 1:5])
                    detected = []
                    for j, anno in enumerate(pr):
                        pr_box = anno[:4]
                        pr_cls = anno[-1]
                        if pr_cls.item() not in gt_cls:
                            # class of predicted annotation not amongst classes of ground-truth annotations
                            continue
                        # find ground-truth annotation that has best iou wrt current annotation
                        m = (gt[:, 0] == pr_cls).nonzero().view(-1)  # class match!
                        iou, box_id = bbox_iou(pr_box, gt_boxes[m]).max(0)
                        cond_iou = iou > iou_thres
                        cond_available = m[box_id] not in detected
                        if cond_iou and cond_available:
                            correct[j] = 1
                            detected.append(m[box_id])
                        if len(detected) == na:
                            # all ground-truth annotations have already been located
                            break
                    img_stats = (correct, pr[:, 4], pr[:, 6], gt_cls)
                self.stats.append(img_stats)

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
            assert len(pr_labels) == len(gt_labels), 'Number of predictions and number of ground-truths do not match!'
            self._update_stats(pr_labels, gt_labels)
            stats = [np.concatenate(s, 0) for s in list(zip(*self.stats))]
            if len(stats):
                self.avg_metric = 100. * ap_per_class(*stats)
            else:
                self.avg_metric = 0.

    def is_better(self, current_metric, best_metric):
        # compare mAP
        return current_metric > best_metric

    def bar(self):
        return '| Loss: {loss:8.5f} | mAP: {map:6.2f}%%'.format(
                loss=self.avg_loss,
                map=self.avg_metric)
