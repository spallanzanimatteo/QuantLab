# Copyright (c) 2019 UniMoRe, Matteo Spallanzani
# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import torch.nn as nn


# In order for the baselines to be launched with the same logic as quantized
# models, an empty quantization scheme and an empty thermostat schedule need
# to be configured.
# Use the following templates for the `net` and `thermostat` configurations:
#
# "net": {
#   "class": "MobileNetv2Baseline",
#   "params": {"capacity": 1, "expansion": 6},
#   "pretrained": null,
#   "loss_fn": {
#     "class": "CrossEntropyLoss",
#     "params": {}
#   }
# }
#
# "thermostat": {
#   "class": "MobileNetv2Baseline",
#   "params": {
#     "noise_schemes": {},
#     "bindings":      []
#   }
# }

class MobileNetv2Baseline(nn.Module):
    """MobileNetv2 Convolutional Neural Network."""
    def __init__(self, capacity=1, expansion=6):
        super().__init__()
        c0 = 3
        t0 = int(32 * capacity) * 1
        c1 = int(16 * capacity)
        t1 = c1 * expansion
        c2 = int(24 * capacity)
        t2 = c2 * expansion
        c3 = int(32 * capacity)
        t3 = c3 * expansion
        c4 = int(64 * capacity)
        t4 = c4 * expansion
        c5 = int(96 * capacity)
        t5 = c5 * expansion
        c6 = int(160 * capacity)
        t6 = c6 * expansion
        c7 = int(320 * capacity)
        c8 = max(int(1280 * capacity), 1280)
        # first block
        self.phi01_conv = nn.Conv2d(c0, t0, kernel_size=3, stride=2, padding=1, bias=False)
        self.phi01_bn   = nn.BatchNorm2d(t0)
        self.phi01_act  = nn.ReLU6(inplace=True)
        self.phi02_conv = nn.Conv2d(t0, t0, kernel_size=3, stride=1, padding=1, groups=t0, bias=False)
        self.phi02_bn   = nn.BatchNorm2d(t0)
        self.phi02_act  = nn.ReLU6(inplace=True)
        self.phi03_conv = nn.Conv2d(t0, c1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi03_bn   = nn.BatchNorm2d(c1)
        # second block
        self.phi04_conv = nn.Conv2d(c1, t1, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi04_bn   = nn.BatchNorm2d(t1)
        self.phi04_act  = nn.ReLU6(inplace=True)
        self.phi05_conv = nn.Conv2d(t1, t1, kernel_size=3, stride=2, padding=1, groups=t1, bias=False)
        self.phi05_bn   = nn.BatchNorm2d(t1)
        self.phi05_act  = nn.ReLU6(inplace=True)
        self.phi06_conv = nn.Conv2d(t1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi06_bn   = nn.BatchNorm2d(c2)
        self.phi07_conv = nn.Conv2d(c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi07_bn   = nn.BatchNorm2d(t2)
        self.phi07_act  = nn.ReLU6(inplace=True)
        self.phi08_conv = nn.Conv2d(t2, t2, kernel_size=3, stride=1, padding=1, groups=t2, bias=False)
        self.phi08_bn   = nn.BatchNorm2d(t2)
        self.phi08_act  = nn.ReLU6(inplace=True)
        self.phi09_conv = nn.Conv2d(t2, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi09_bn   = nn.BatchNorm2d(c2)
        # third block
        self.phi10_conv = nn.Conv2d(c2, t2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi10_bn   = nn.BatchNorm2d(t2)
        self.phi10_act  = nn.ReLU6(inplace=True)
        self.phi11_conv = nn.Conv2d(t2, t2, kernel_size=3, stride=2, padding=1, groups=t2, bias=False)
        self.phi11_bn   = nn.BatchNorm2d(t2)
        self.phi11_act  = nn.ReLU6(inplace=True)
        self.phi12_conv = nn.Conv2d(t2, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi12_bn   = nn.BatchNorm2d(c3)
        self.phi13_conv = nn.Conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi13_bn   = nn.BatchNorm2d(t3)
        self.phi13_act  = nn.ReLU6(inplace=True)
        self.phi14_conv = nn.Conv2d(t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi14_bn   = nn.BatchNorm2d(t3)
        self.phi14_act  = nn.ReLU6(inplace=True)
        self.phi15_conv = nn.Conv2d(t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi15_bn   = nn.BatchNorm2d(c3)
        self.phi16_conv = nn.Conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi16_bn   = nn.BatchNorm2d(t3)
        self.phi16_act  = nn.ReLU6(t3)
        self.phi17_conv = nn.Conv2d(t3, t3, kernel_size=3, stride=1, padding=1, groups=t3, bias=False)
        self.phi17_bn   = nn.BatchNorm2d(t3)
        self.phi17_act  = nn.ReLU6(inplace=True)
        self.phi18_conv = nn.Conv2d(t3, c3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi18_bn   = nn.BatchNorm2d(c3)
        # fourth block
        self.phi19_conv = nn.Conv2d(c3, t3, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi19_bn   = nn.BatchNorm2d(t3)
        self.phi19_act  = nn.ReLU6(inplace=True)
        self.phi20_conv = nn.Conv2d(t3, t3, kernel_size=3, stride=2, padding=1, groups=t3, bias=False)
        self.phi20_bn   = nn.BatchNorm2d(t3)
        self.phi20_act  = nn.ReLU6(inplace=True)
        self.phi21_conv = nn.Conv2d(t3, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi21_bn   = nn.BatchNorm2d(c4)
        self.phi22_conv = nn.Conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi22_bn   = nn.BatchNorm2d(t4)
        self.phi22_act  = nn.ReLU6(inplace=True)
        self.phi23_conv = nn.Conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi23_bn   = nn.BatchNorm2d(t4)
        self.phi23_act  = nn.ReLU6(inplace=True)
        self.phi24_conv = nn.Conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi24_bn   = nn.BatchNorm2d(c4)
        self.phi25_conv = nn.Conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi25_bn   = nn.BatchNorm2d(t4)
        self.phi25_act  = nn.ReLU6(inplace=True)
        self.phi26_conv = nn.Conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi26_bn   = nn.BatchNorm2d(t4)
        self.phi26_act  = nn.ReLU6(inplace=True)
        self.phi27_conv = nn.Conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi27_bn   = nn.BatchNorm2d(c4)
        self.phi28_conv = nn.Conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi28_bn   = nn.BatchNorm2d(t4)
        self.phi28_act  = nn.ReLU6(inplace=True)
        self.phi29_conv = nn.Conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi29_bn   = nn.BatchNorm2d(t4)
        self.phi29_act  = nn.ReLU6(inplace=True)
        self.phi30_conv = nn.Conv2d(t4, c4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi30_bn   = nn.BatchNorm2d(c4)
        # fifth block
        self.phi31_conv = nn.Conv2d(c4, t4, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi31_bn   = nn.BatchNorm2d(t4)
        self.phi31_act  = nn.ReLU6(inplace=True)
        self.phi32_conv = nn.Conv2d(t4, t4, kernel_size=3, stride=1, padding=1, groups=t4, bias=False)
        self.phi32_bn   = nn.BatchNorm2d(t4)
        self.phi32_act  = nn.ReLU6(inplace=True)
        self.phi33_conv = nn.Conv2d(t4, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi33_bn   = nn.BatchNorm2d(c5)
        self.phi34_conv = nn.Conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi34_bn   = nn.BatchNorm2d(t5)
        self.phi34_act  = nn.ReLU6(inplace=True)
        self.phi35_conv = nn.Conv2d(t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi35_bn   = nn.BatchNorm2d(t5)
        self.phi35_act  = nn.ReLU6(inplace=True)
        self.phi36_conv = nn.Conv2d(t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi36_bn   = nn.BatchNorm2d(c5)
        self.phi37_conv = nn.Conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi37_bn   = nn.BatchNorm2d(t5)
        self.phi37_act  = nn.ReLU6(inplace=True)
        self.phi38_conv = nn.Conv2d(t5, t5, kernel_size=3, stride=1, padding=1, groups=t5, bias=False)
        self.phi38_bn   = nn.BatchNorm2d(t5)
        self.phi38_act  = nn.ReLU6(inplace=True)
        self.phi39_conv = nn.Conv2d(t5, c5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi39_bn   = nn.BatchNorm2d(c5)
        # sixth block
        self.phi40_conv = nn.Conv2d(c5, t5, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi40_bn   = nn.BatchNorm2d(t5)
        self.phi40_act  = nn.ReLU6(inplace=True)
        self.phi41_conv = nn.Conv2d(t5, t5, kernel_size=3, stride=2, padding=1, groups=t5, bias=False)
        self.phi41_bn   = nn.BatchNorm2d(t5)
        self.phi41_act  = nn.ReLU6(inplace=True)
        self.phi42_conv = nn.Conv2d(t5, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi42_bn   = nn.BatchNorm2d(c6)
        self.phi43_conv = nn.Conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi43_bn   = nn.BatchNorm2d(t6)
        self.phi43_act  = nn.ReLU6(inplace=True)
        self.phi44_conv = nn.Conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi44_bn   = nn.BatchNorm2d(t6)
        self.phi44_act  = nn.ReLU6(inplace=True)
        self.phi45_conv = nn.Conv2d(t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi45_bn   = nn.BatchNorm2d(c6)
        self.phi46_conv = nn.Conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi46_bn   = nn.BatchNorm2d(t6)
        self.phi46_act  = nn.ReLU6(inplace=True)
        self.phi47_conv = nn.Conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi47_bn   = nn.BatchNorm2d(t6)
        self.phi47_act  = nn.ReLU6(inplace=True)
        self.phi48_conv = nn.Conv2d(t6, c6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi48_bn   = nn.BatchNorm2d(c6)
        # seventh block
        self.phi49_conv = nn.Conv2d(c6, t6, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi49_bn   = nn.BatchNorm2d(t6)
        self.phi49_act  = nn.ReLU6(inplace=True)
        self.phi50_conv = nn.Conv2d(t6, t6, kernel_size=3, stride=1, padding=1, groups=t6, bias=False)
        self.phi50_bn   = nn.BatchNorm2d(t6)
        self.phi50_act  = nn.ReLU6(inplace=True)
        self.phi51_conv = nn.Conv2d(t6, c7, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi51_bn   = nn.BatchNorm2d(c7)
        # classifier
        self.phi52_conv = nn.Conv2d(c7, c8, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi52_bn   = nn.BatchNorm2d(c8)
        self.phi52_act  = nn.ReLU6(inplace=True)
        self.phi53_avg  = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.phi53_fc   = nn.Linear(c8, 1000)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, withStats=False):
        # first block
        x     = self.phi01_conv(x)
        x     = self.phi01_bn(x)
        x     = self.phi01_act(x)
        x     = self.phi02_conv(x)
        x     = self.phi02_bn(x)
        x     = self.phi02_act(x)
        x     = self.phi03_conv(x)
        x     = self.phi03_bn(x)
        # second block
        x     = self.phi04_conv(x)
        x     = self.phi04_bn(x)
        x     = self.phi04_act(x)
        x     = self.phi05_conv(x)
        x     = self.phi05_bn(x)
        x     = self.phi05_act(x)
        x     = self.phi06_conv(x)
        x     = self.phi06_bn(x)
        x_res = self.phi07_conv(x)
        x_res = self.phi07_bn(x_res)
        x_res = self.phi07_act(x_res)
        x_res = self.phi08_conv(x_res)
        x_res = self.phi08_bn(x_res)
        x_res = self.phi08_act(x_res)
        x_res = self.phi09_conv(x_res)
        x_res = self.phi09_bn(x_res)
        x     = x + x_res
        # third block
        x     = self.phi10_conv(x)
        x     = self.phi10_bn(x)
        x     = self.phi10_act(x)
        x     = self.phi11_conv(x)
        x     = self.phi11_bn(x)
        x     = self.phi11_act(x)
        x     = self.phi12_conv(x)
        x     = self.phi12_bn(x)
        x_res = self.phi13_conv(x)
        x_res = self.phi13_bn(x_res)
        x_res = self.phi13_act(x_res)
        x_res = self.phi14_conv(x_res)
        x_res = self.phi14_bn(x_res)
        x_res = self.phi14_act(x_res)
        x_res = self.phi15_conv(x_res)
        x_res = self.phi15_bn(x_res)
        x     = x + x_res
        x_res = self.phi16_conv(x)
        x_res = self.phi16_bn(x_res)
        x_res = self.phi16_act(x_res)
        x_res = self.phi17_conv(x_res)
        x_res = self.phi17_bn(x_res)
        x_res = self.phi17_act(x_res)
        x_res = self.phi18_conv(x_res)
        x_res = self.phi18_bn(x_res)
        x     = x + x_res
        # fourth block
        x     = self.phi19_conv(x)
        x     = self.phi19_bn(x)
        x     = self.phi19_act(x)
        x     = self.phi20_conv(x)
        x     = self.phi20_bn(x)
        x     = self.phi20_act(x)
        x     = self.phi21_conv(x)
        x     = self.phi21_bn(x)
        x_res = self.phi22_conv(x)
        x_res = self.phi22_bn(x_res)
        x_res = self.phi22_act(x_res)
        x_res = self.phi23_conv(x_res)
        x_res = self.phi23_bn(x_res)
        x_res = self.phi23_act(x_res)
        x_res = self.phi24_conv(x_res)
        x_res = self.phi24_bn(x_res)
        x     = x + x_res
        x_res = self.phi25_conv(x)
        x_res = self.phi25_bn(x_res)
        x_res = self.phi25_act(x_res)
        x_res = self.phi26_conv(x_res)
        x_res = self.phi26_bn(x_res)
        x_res = self.phi26_act(x_res)
        x_res = self.phi27_conv(x_res)
        x_res = self.phi27_bn(x_res)
        x     = x + x_res
        x_res = self.phi28_conv(x)
        x_res = self.phi28_bn(x_res)
        x_res = self.phi28_act(x_res)
        x_res = self.phi29_conv(x_res)
        x_res = self.phi29_bn(x_res)
        x_res = self.phi29_act(x_res)
        x_res = self.phi30_conv(x_res)
        x_res = self.phi30_bn(x_res)
        x     = x + x_res
        # fifth block
        x     = self.phi31_conv(x)
        x     = self.phi31_bn(x)
        x     = self.phi31_act(x)
        x     = self.phi32_conv(x)
        x     = self.phi32_bn(x)
        x     = self.phi32_act(x)
        x     = self.phi33_conv(x)
        x     = self.phi33_bn(x)
        x_res = self.phi34_conv(x)
        x_res = self.phi34_bn(x_res)
        x_res = self.phi34_act(x_res)
        x_res = self.phi35_conv(x_res)
        x_res = self.phi35_bn(x_res)
        x_res = self.phi35_act(x_res)
        x_res = self.phi36_conv(x_res)
        x_res = self.phi36_bn(x_res)
        x     = x + x_res
        x_res = self.phi37_conv(x)
        x_res = self.phi37_bn(x_res)
        x_res = self.phi37_act(x_res)
        x_res = self.phi38_conv(x_res)
        x_res = self.phi38_bn(x_res)
        x_res = self.phi38_act(x_res)
        x_res = self.phi39_conv(x_res)
        x_res = self.phi39_bn(x_res)
        x     = x + x_res
        # sixth block
        x     = self.phi40_conv(x)
        x     = self.phi40_bn(x)
        x     = self.phi40_act(x)
        x     = self.phi41_conv(x)
        x     = self.phi41_bn(x)
        x     = self.phi41_act(x)
        x     = self.phi42_conv(x)
        x     = self.phi42_bn(x)
        x_res = self.phi43_conv(x)
        x_res = self.phi43_bn(x_res)
        x_res = self.phi43_act(x_res)
        x_res = self.phi44_conv(x_res)
        x_res = self.phi44_bn(x_res)
        x_res = self.phi44_act(x_res)
        x_res = self.phi45_conv(x_res)
        x_res = self.phi45_bn(x_res)
        x     = x + x_res
        x_res = self.phi46_conv(x)
        x_res = self.phi46_bn(x_res)
        x_res = self.phi46_act(x_res)
        x_res = self.phi47_conv(x_res)
        x_res = self.phi47_bn(x_res)
        x_res = self.phi47_act(x_res)
        x_res = self.phi48_conv(x_res)
        x_res = self.phi48_bn(x_res)
        x     = x + x_res
        # seventh block
        x     = self.phi49_conv(x)
        x     = self.phi49_bn(x)
        x     = self.phi49_act(x)
        x     = self.phi50_conv(x)
        x     = self.phi50_bn(x)
        x     = self.phi50_act(x)
        x     = self.phi51_conv(x)
        x     = self.phi51_bn(x)
        # classifier
        x     = self.phi52_conv(x)
        x     = self.phi52_bn(x)
        x     = self.phi52_act(x)
        x     = self.phi53_avg(x)
        x     = x.view(x.size(0), -1)
        x     = self.phi53_fc(x)
        
        if withStats:
            stats = []
            return stats, x

        return x

    def forward_with_tensor_stats(self, x):
        stats, x = self.forward(x, withStats=True)
        return stats, x
