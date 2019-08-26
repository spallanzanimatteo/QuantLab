# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

import torch
import torch.nn as nn


# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# Use the following templates for the `net` and `thermostat` configurations:
#
# "net": {
#   "class": "YOLOv3TinyBaseline",
#   "params": {},
#   "pretrained": null,
#   "loss_function": {
#     "class": "YOLOv3Loss",
#     "params": {}
#   }
# }
#
# "thermostat": {
#   "class": "YOLOv3TinyBaseline",
#   "params": {
#     "noise_schemes": {},
#     "bindings":      []
#   }
# }

class YOLOv3Layer(nn.Module):

    def __init__(self, anchors, num_classes):
        super(YOLOv3Layer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)
        self.nx = 0  # cells in x direction
        self.ny = 0  # cells in y direction
        self.ng = torch.Tensor([0, 0]).type(torch.int)  # cells in grid
        self.pxc = torch.Tensor([0, 0]).type(torch.int)  # pixels per cell (in x and y directions)
        self.ag = None
        self.og = None
        self.nc = num_classes

    def compute_grids(self, img_size=416, ng=(13, 13), device='cpu', dtype=torch.float32):
        nx, ny = ng
        self.nx = nx
        self.ny = ny
        self.ng = torch.Tensor(ng).to(device)
        self.pxc = img_size / self.ng
        # anchors grid
        cxa = self.anchors.to(device) / self.pxc  # anchors sizes (in grid cells instead of pixels)
        self.ag = cxa.view(1, self.na, 1, 1, 2).to(device).type(dtype).detach()
        # cells offsets grid
        oy, ox = torch.meshgrid([torch.arange(self.ny), torch.arange(self.nx)])
        self.og = torch.stack((ox, oy), 2).view(1, 1, self.ny, self.nx, 2).to(device).type(dtype).detach()

    def forward(self, x, img_size):
        bs = x.size(0)
        ny = x.size(-2)
        nx = x.size(-1)
        if (self.nx, self.ny) != (nx, ny):
            self.compute_grids(img_size=img_size, ng=(nx, ny), device=x.device, dtype=x.dtype)
        x = x.view(bs, self.na, 4+1+self.nc, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        # activate predictions
        x[..., 0:2] = torch.sigmoid(x[..., 0:2]) + self.og.to(x)
        x[..., 2:4] = torch.exp(x[..., 2:4]) * self.ag.to(x)
        x[..., 0:4:2] = x[..., 0:4:2] / self.nx
        x[..., 1:4:2] = x[..., 1:4:2] / self.ny
        x[..., 4:5] = torch.sigmoid(x[..., 4:5])
        x[..., 5:] = torch.sigmoid(x[..., 5:])
        return x


class YOLOv3TinyBaseline(nn.Module):

    def __init__(self):
        super(YOLOv3TinyBaseline, self).__init__()
        self.nc = 80
        self.img_size = 416
        self.anchors = [[(81, 82), (135, 169), (344, 319)],
                        [(23, 27), (37, 58),   (81, 82)]]
        na_coarse = len(self.anchors[0])
        na_fine   = len(self.anchors[1])
        self.yololayers = []
        self.phi01_conv  = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi01_bn    = nn.BatchNorm2d(num_features=16, momentum=0.9)
        self.phi01_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi02_mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.phi02_conv  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi02_bn    = nn.BatchNorm2d(num_features=32, momentum=0.9)
        self.phi02_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi03_mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.phi03_conv  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi03_bn    = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.phi03_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi04_mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.phi04_conv  = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi04_bn    = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.phi04_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi05_mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.phi05_conv  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi05_bn    = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.phi05_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi06_mp    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.phi06_conv  = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi06_bn    = nn.BatchNorm2d(num_features=512, momentum=0.9)
        self.phi06_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi07_zp    = nn.ZeroPad2d((0, 1, 0, 1))
        self.phi07_mp    = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.phi07_conv  = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi07_bn    = nn.BatchNorm2d(num_features=1024, momentum=0.9)
        self.phi07_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi08_conv  = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi08_bn    = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.phi08_act   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # coarse-grained branch
        self.phi09a_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi09a_bn   = nn.BatchNorm2d(num_features=512, momentum=0.9)
        self.phi09a_act  = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi10a_conv = nn.Conv2d(in_channels=512, out_channels=na_coarse*(4+1+self.nc), kernel_size=1, stride=1, padding=0, bias=True)
        self.phi10a_yolo = YOLOv3Layer(self.anchors[0], num_classes=self.nc)
        # fine-grained branch
        self.phi09b_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi09b_bn   = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.phi09b_act  = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi10b_up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.phi10b_conv = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.phi10b_bn   = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.phi10b_act  = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.phi11b_conv = nn.Conv2d(in_channels=256, out_channels=na_fine*(4+1+self.nc), kernel_size=1, stride=1, padding=0, bias=True)
        self.phi11b_yolo = YOLOv3Layer(self.anchors[1], num_classes=self.nc)
        # yolo layers list (for loss function)
        self.yololayers  = [self.phi10a_yolo, self.phi11b_yolo]

    def forward(self, x):
        img_size = max(x.size()[-2:])
        if img_size != self.img_size:
            self.img_size = img_size
        pr_outs = []
        x = self.phi01_conv(x)
        x = self.phi01_bn(x)
        x = self.phi01_act(x)
        x = self.phi02_mp(x)
        x = self.phi02_conv(x)
        x = self.phi02_bn(x)
        x = self.phi02_act(x)
        x = self.phi03_mp(x)
        x = self.phi03_conv(x)
        x = self.phi03_bn(x)
        x = self.phi03_act(x)
        x = self.phi04_mp(x)
        x = self.phi04_conv(x)
        x = self.phi04_bn(x)
        x = self.phi04_act(x)
        x = self.phi05_mp(x)
        x = self.phi05_conv(x)
        x = self.phi05_bn(x)
        x_phi05 = self.phi05_act(x)  # route layer (entry)
        x = self.phi06_mp(x_phi05)
        x = self.phi06_conv(x)
        x = self.phi06_bn(x)
        x = self.phi06_act(x)
        x = self.phi07_zp(x)
        x = self.phi07_mp(x)
        x = self.phi07_conv(x)
        x = self.phi07_bn(x)
        x = self.phi07_act(x)
        x = self.phi08_conv(x)
        x = self.phi08_bn(x)
        x_phi08 = self.phi08_act(x)
        x = self.phi09a_conv(x_phi08)  # coarse-grained branch
        x = self.phi09a_bn(x)
        x = self.phi09a_act(x)
        x = self.phi10a_conv(x)
        x = self.phi10a_yolo(x, self.img_size)  # coarse-grained prediction
        pr_outs.append(x)
        x = self.phi09b_conv(x_phi08)  # fine-grained branch
        x = self.phi09b_bn(x)
        x = self.phi09b_act(x)
        x = self.phi10b_up(x)
        x = torch.cat([x, x_phi05], 1)  # route layer (exit)
        x = self.phi10b_conv(x)
        x = self.phi10b_bn(x)
        x = self.phi10b_act(x)
        x = self.phi11b_conv(x)
        x = self.phi11b_yolo(x, self.img_size)  # fine-grained prediction
        pr_outs.append(x)
        return pr_outs

    def forward_with_tensor_stats(self, x):
        stats = []
        img_size = max(x.size()[-2:])
        if img_size != self.img_size:
            self.img_size = img_size
        pr_outs = []
        x = self.phi01_conv(x)
        x = self.phi01_bn(x)
        x = self.phi01_act(x)
        x = self.phi02_mp(x)
        x = self.phi02_conv(x)
        x = self.phi02_bn(x)
        x = self.phi02_act(x)
        x = self.phi03_mp(x)
        x = self.phi03_conv(x)
        x = self.phi03_bn(x)
        x = self.phi03_act(x)
        x = self.phi04_mp(x)
        x = self.phi04_conv(x)
        x = self.phi04_bn(x)
        x = self.phi04_act(x)
        x = self.phi05_mp(x)
        x = self.phi05_conv(x)
        x = self.phi05_bn(x)
        x_phi05 = self.phi05_act(x)  # route layer (entry)
        x = self.phi06_mp(x_phi05)
        x = self.phi06_conv(x)
        x = self.phi06_bn(x)
        x = self.phi06_act(x)
        x = self.phi07_zp(x)
        x = self.phi07_mp(x)
        x = self.phi07_conv(x)
        x = self.phi07_bn(x)
        x = self.phi07_act(x)
        x = self.phi08_conv(x)
        x = self.phi08_bn(x)
        x_phi08 = self.phi08_act(x)
        x = self.phi09a_conv(x_phi08)  # coarse-grained branch
        x = self.phi09a_bn(x)
        x = self.phi09a_act(x)
        x = self.phi10a_conv(x)
        x = self.phi10a_yolo(x, self.img_size)  # coarse-grained prediction
        pr_outs.append(x)
        x = self.phi09b_conv(x_phi08)  # fine-grained branch
        x = self.phi09b_bn(x)
        x = self.phi09b_act(x)
        x = self.phi10b_up(x)
        x = torch.cat([x, x_phi05], 1)  # route layer (exit)
        x = self.phi10b_conv(x)
        x = self.phi10b_bn(x)
        x = self.phi10b_act(x)
        x = self.phi11b_conv(x)
        x = self.phi11b_yolo(x, self.img_size)  # fine-grained prediction
        pr_outs.append(x)
        return stats, pr_outs
