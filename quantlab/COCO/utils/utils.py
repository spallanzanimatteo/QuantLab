# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch


def xywh2xyxy(xywh):
    """Convert bounding box encoding from [x, y, w, h] to [x1, y1, x2, y2]."""
    # the input tensor must have four components in the last dimension
    xyxy = torch.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def xyxy2xywh(xyxy):
    """Convert bounding box enconding from [x1, y1, x2, y2] to [x, y, w, h]."""
    # the input tensor must have four components in the last dimension
    xywh = torch.zeros_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def bbox_iou(box1, box2, xyxy=True):
    """Compute Intersection over Union (IoU) of two given bounding boxes."""
    # get x1, x2, y1, y2 coordinates for each box
    box2 = box2.t()
    if not xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = xywh2xyxy(box1)
        b2_x1, b2_y1, b2_x2, b2_y2 = xywh2xyxy(box2)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # intersection area
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-16
    return inter_area / union_area


def bbox_iou_wh(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = w1 * h1 + w2 * h2 - inter_area + 1e-16
    return inter_area / union_area
