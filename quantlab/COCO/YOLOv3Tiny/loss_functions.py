# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn

from ..utils.utils import bbox_iou_wh


class YOLOv3Loss(nn.Module):
    def __init__(self, net):
        super(YOLOv3Loss, self).__init__()
        self.net = net
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 5
        self.iou_thres = 0.5

    def _pr_outs(self, pr, layer):
        # revert lines 10-11 in `forward` method of `YOLOv3Layer`
        pr_x = pr[..., 0] * layer.nx
        pr_y = pr[..., 1] * layer.ny
        pr_w = pr[..., 2] * layer.nx
        pr_h = pr[..., 3] * layer.ny
        # revert line 8 in `forward` method of `YOLOv3Layer`
        pr_x2 = pr_x - layer.og.to(pr)[..., 0]
        pr_x2 = torch.clamp(pr_x2, min=0+1e-5, max=1-1e-5)
        pr_x3 = torch.log(pr_x2 / (1. - pr_x2) + 1e-5)
        pr_y2 = pr_y - layer.og.to(pr)[..., 1]
        pr_y2 = torch.clamp(pr_y2, min=0+1e-5, max=1-1e-5)
        pr_y3 = torch.log(pr_y2 / (1. - pr_y2) + 1e-5)
        # revert line 9 in `forward` method of `YOLOv3Layer`
        pr_w2 = pr_w / layer.ag.to(pr)[..., 0]
        pr_w2 = torch.clamp(pr_w2, min=0+1e-5)
        pr_w3 = torch.log(pr_w2)
        pr_h2 = pr_h / layer.ag.to(pr)[..., 1]
        pr_h2 = torch.clamp(pr_h2, min=0+1e-5)
        pr_h3 = torch.log(pr_h2)
        return pr_x3, pr_y3, pr_w3, pr_h3

    def _gt_outs(self, bs, cuda, layer, iou_thres, gt_labels):
        # initialize masks and ground-truth outputs data structure
        bs = bs
        na = layer.na
        ny = layer.ny
        nx = layer.nx
        nc = layer.nc
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
        obj_mask = ByteTensor(bs, na, ny, nx).fill_(0)
        noobj_mask = ByteTensor(bs, na, ny, nx).fill_(1)
        gt_x = FloatTensor(bs, na, ny, nx).fill_(0)
        gt_y = FloatTensor(bs, na, ny, nx).fill_(0)
        gt_w = FloatTensor(bs, na, ny, nx).fill_(0)
        gt_h = FloatTensor(bs, na, ny, nx).fill_(0)
        gt_conf = FloatTensor(bs, na, ny, nx).fill_(0)
        gt_cls = FloatTensor(bs, na, ny, nx, nc).fill_(0)
        # convert ground-truth boxes annotations to format coherent with YOLOv3 predictions (cells grid)
        gt_labels = gt_labels.clone()
        img_id, gt_cls_id = gt_labels[:, 1:3].long().t()
        gt_boxes = gt_labels[:, 3:7]
        gt_boxes[:, 0:4:2] *= nx
        gt_boxes[:, 1:4:2] *= ny
        boxes_xy = gt_boxes[:, 0:2]
        boxes_wh = gt_boxes[:, 2:4]
        gx, gy = boxes_xy.t()
        gw, gh = boxes_wh.t()
        gi, gj = boxes_xy.long().t()
        # find anchors with best iou
        gt_ious_a = torch.stack([bbox_iou_wh(a, boxes_wh) for a in layer.ag.squeeze()]).t()
        _, gt_best_a_id = gt_ious_a.max(1)
        obj_mask[img_id, gt_best_a_id, gj, gi] = 1
        noobj_mask[img_id, gt_best_a_id, gj, gi] = 0
        # YOLOv3 paper, end of pag.1-beginning of pag.2
        for gt_anno_id, gt_anno_ious_a in enumerate(gt_ious_a):
            noobj_mask[img_id[gt_anno_id], gt_anno_ious_a > iou_thres, gj[gt_anno_id], gi[gt_anno_id]] = 0
        # ground-truth outputs (boxes): invert as indicated in YOLOv3 paper, almost end of pag.1
        gx = gx - gx.floor()
        gx[gx == 0.] += 1e-5
        gx = torch.log(gx / (1. - gx))
        gy = gy - gy.floor()
        gy[gy == 0.] += 1e-5
        gy = torch.log(gy / (1. - gy))
        gw = torch.log(gw / layer.ag.squeeze()[gt_best_a_id, 0] + 1e-5)
        gh = torch.log(gh / layer.ag.squeeze()[gt_best_a_id, 1] + 1e-5)
        gt_x[img_id, gt_best_a_id, gj, gi] = gx
        gt_y[img_id, gt_best_a_id, gj, gi] = gy
        gt_w[img_id, gt_best_a_id, gj, gi] = gw
        gt_h[img_id, gt_best_a_id, gj, gi] = gh
        # ground-truth outputs (confidence and classes)
        gt_conf = obj_mask.float()
        gt_cls[img_id, gt_best_a_id, gj, gi, gt_cls_id] = 1
        return obj_mask, noobj_mask, gt_x, gt_y, gt_w, gt_h, gt_conf, gt_cls

    def forward(self, pr_outs, gt_labels):
        loss = 0
        for i, pr in enumerate(pr_outs):
            layer = self.net.yololayers[i]
            pr_x, pr_y, pr_w, pr_h = self._pr_outs(pr, layer)
            pr_conf = pr[..., 4]
            pr_cls = pr[..., 5:]
            # transform ground-truth labels in ground-truth output data structures
            obj_mask, noobj_mask, gt_x, gt_y, gt_w, gt_h, gt_conf, gt_cls = self._gt_outs(bs=pr.size(0),
                                                                                          cuda=pr.is_cuda,
                                                                                          layer=layer,
                                                                                          iou_thres=self.iou_thres,
                                                                                          gt_labels=gt_labels)
            l_x = self.mse_loss(pr_x[obj_mask], gt_x[obj_mask])
            l_y = self.mse_loss(pr_y[obj_mask], gt_y[obj_mask])
            l_w = self.mse_loss(pr_w[obj_mask], gt_w[obj_mask])
            l_h = self.mse_loss(pr_h[obj_mask], gt_h[obj_mask])
            l_conf_obj = self.bce_loss(pr_conf[obj_mask], gt_conf[obj_mask])
            l_conf_noobj = self.bce_loss(pr_conf[noobj_mask], gt_conf[noobj_mask])
            l_conf = self.obj_scale * l_conf_obj + self.noobj_scale * l_conf_noobj
            l_cls = self.bce_loss(pr_cls[obj_mask], gt_cls[obj_mask])
            l_t = l_x + l_y + l_w + l_h + l_conf + l_cls
            loss += l_t
        return loss
