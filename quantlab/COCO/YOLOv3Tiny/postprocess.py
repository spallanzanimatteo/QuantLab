# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch

from ..utils.utils import xywh2xyxy, bbox_iou


def clip_boxes(boxes):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=1)


def postprocess_pr(pr_outs, conf_thres=0.001, overlap_thres=0.5):
    """Restructure YOLOv3Tiny tensors into lists, then filter out non-maximal
    (redundant) annotations from the predictions."""
    # pr_outs = [[bs, grid_positions, 85], [bs, 4*grid_positions, 85]]
    # when its two components are concatenated, we get a tensor [bs, 5*gridpositions, 85], which `bs` "slices"
    # have to be "stripped" to remove redundant components
    # strip each slice (corresponding to a single image in the batch) to get sequences of (possibly) different lengths:
    # the natural data structure to use to collect these sequences is a list
    pr_outs = [p.view(p.size(0), -1, p.size(-1)) for p in pr_outs]
    pr_outs = torch.cat(pr_outs, 1).detach().cpu()
    pr_labels = [None] * len(pr_outs)
    for img_id, pr in enumerate(pr_outs):
        # filter out irrelevant predictions
        pr_cls_prob, pr_cls_id = pr[:, 5:].max(1)
        pr[:, 4] *= pr_cls_prob
        i = (pr[:, 4] > conf_thres) & torch.isfinite(pr).all(1)
        pr = pr[i]
        if len(pr) == 0:
            continue
        pr_cls_prob = pr_cls_prob[i]
        pr_cls_id = pr_cls_id[i].unsqueeze(1).float()
        pr[:, :4] = xywh2xyxy(pr[:, :4])
        pr = torch.cat((pr[:, :5], pr_cls_prob.unsqueeze(1), pr_cls_id), 1)
        pr = pr[(-pr[:, 4]).argsort()]
        detections = []
        for c in pr[:, -1].unique():
            pr_anno_c = pr[pr[:, -1] == c]
            n = len(pr_anno_c)
            if n == 1:
                detections.append(pr_anno_c)
                continue
            elif n > 100:
                pr_anno_c = pr_anno_c[:100]
            while len(pr_anno_c) > 0:
                if len(pr_anno_c) == 1:
                    detections.append(pr_anno_c)
                    break
                redundant = bbox_iou(pr_anno_c[0], pr_anno_c) > overlap_thres
                weights = pr_anno_c[redundant, 4:5]
                pr_anno_c[0, :4] = (weights * pr_anno_c[redundant, 0:4]).sum(0) / weights.sum()
                detections.append(pr_anno_c[0:1])  # keep leading dimension 1 for 1D tensor
                pr_anno_c = pr_anno_c[~redundant]
        if len(detections) > 0:
            detections = torch.cat(detections)
            clip_boxes(detections[:, :4])
            pr_labels[img_id] = detections[(-detections[:, 4]).argsort()]
    return pr_labels


def postprocess_gt(gt_labels):
    gt_labels = gt_labels.detach().cpu()
    bs = gt_labels[0, 0].to(torch.int)
    gt_labels = [gt_labels[gt_labels[:, 1] == i, 2:] for i in range(bs)]
    return gt_labels
